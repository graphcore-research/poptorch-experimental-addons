# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Static sparse-dense matrix multiplication.

These APIs are based on a block COO format, created as:
```python
torch.sparse_coo_tensor(indices, values, size=(n_rows, n_cols))
#   indices -- shape (2, nnz_blocks) -- row and column block indices
#   values -- shape (nnz_blocks, block_size, block_size)
```

To transpose & convert to dense correctly, use the included functions
`block_coo_transpose()` and `block_coo_to_dense()`.
"""

from dataclasses import dataclass

import numpy as np
import poptorch
import torch
from torch import Tensor

# Low-level API


def block_coo_transpose(tensor: Tensor) -> Tensor:
    """The 2D matrix transpose of a block sparse tensor."""
    return tensor.permute(1, 0, 3, 2).coalesce()


def block_coo_to_dense(tensor: Tensor) -> Tensor:
    """Convert a block sparse tensor to dense."""
    if tensor.sparse_dim() != tensor.dense_dim():
        raise ValueError(
            f"Block COO requires sparse_dim ({tensor.sparse_dim()})"
            f" == dense_dim ({tensor.dense_dim()})"
        )

    rank = tensor.sparse_dim()

    # interleave blocks & block sizes
    permutation = tuple(np.stack([range(rank), range(rank, 2 * rank)], -1).flatten())
    blocks, block_size = np.array_split(tensor.shape, [rank])
    shape = tuple(blocks * block_size)

    return tensor.to_dense().permute(permutation).reshape(shape)


def block_coo_spmm(sparse: Tensor, dense: Tensor, mode: str) -> Tensor:
    """
    A block-COO matrix multiplication.

    - if `mode=="sparse_dense"`, `y = sparse @ dense`
    - if `mode=="dense_sparse"`, `y = dense @ sparse`

    sparse -- torch.sparse_coo_tensor -- should be coalesced, dimensions
              `(blocks_row, blocks_col, block_size_row, block_size_col)`
    """
    if poptorch.isRunningOnIpu():
        return block_coo_spmm_ipu(sparse, dense, mode)
    return block_coo_spmm_gs(sparse, dense, mode)


def block_coo_spmm_gs(sparse: Tensor, dense: Tensor, mode: str) -> Tensor:
    """
    Implement a block sparse COO matmul "manually" using gather-multiply-scatter.

    See `block_coo_spmm()`.
    """
    if sparse.layout != torch.sparse_coo:
        raise ValueError(f"Expected sparse_coo `lhs`, actual layout: {sparse.layout}")

    if mode == "sparse_dense":
        lhs = sparse
        rhs = dense
    elif mode == "dense_sparse":
        lhs = block_coo_transpose(sparse)
        rhs = dense.T
    else:
        raise ValueError(
            "Expected block_coo_spmm_gs mode either 'sparse_dense' or 'dense_sparse'"
        )

    blocks_out, blocks_in, block_size_out, block_size_in = lhs.shape
    elements_in, batch_size = rhs.shape
    elements_out = blocks_out * block_size_out
    indices_out, indices_in = lhs.indices()

    if elements_in != blocks_in * block_size_in:
        raise ValueError(
            f"Block sparse input size ({blocks_in} blocks * {block_size_in})"
            f" does not match dense input size ({elements_in})"
        )

    # 1. Gather
    inputs = rhs.view(blocks_in, block_size_in, batch_size)[indices_in]
    # 2. Multiply
    products = lhs.values() @ inputs
    # 3. Scatter
    output = torch.scatter_add(
        torch.zeros(blocks_out, block_size_out, batch_size, dtype=rhs.dtype),
        dim=0,
        index=indices_out[:, None, None].expand(-1, block_size_out, batch_size),
        src=products,
    ).view(elements_out, batch_size)

    return output.T if mode == "dense_sparse" else output


def block_coo_spmm_ipu(sparse: Tensor, dense: Tensor, mode: str) -> Tensor:
    """
    IPU-only spmm using `popsparse::static_::matMul`.

    See `block_coo_spmm()`.
    """

    if mode not in ["sparse_dense", "dense_sparse"]:
        raise ValueError(
            "Expected block_coo_spmm_ipu mode either 'sparse_dense' or 'dense_sparse'"
        )
    if sparse.layout != torch.sparse_coo:
        raise ValueError(
            "block_coo_spmm_ipu requires `sparse` to be torch.sparse_coo_tensor"
            f", actual {sparse.layout}"
        )
    if sparse.sparse_dim() != 2 or sparse.dense_dim() != 2:
        raise ValueError(
            f"block_coo_spmm_ipu requires `sparse` to be 2D block sparse, with"
            f" sparse_dim ({sparse.sparse_dim()}) == 2"
            f", dense_dim ({sparse.dense_dim()}) == 2",
        )
    if sparse.shape[2] != sparse.shape[3]:
        raise ValueError(
            "block_coo_spmm_ipu requires square blocks i.e. shape (*, *, B, B)"
            f", actual: {sparse.shape}"
        )
    if len(dense.shape) != 2:
        raise ValueError(
            "block_coo_spmm_ipu supports only 2D dense operands"
            f", actual shape: {sparse.shape}"
        )

    if mode == "sparse_dense":
        output_shape = (sparse.shape[0] * sparse.shape[2], dense.shape[1])
    elif mode == "dense_sparse":
        output_shape = (dense.shape[0], sparse.shape[1] * sparse.shape[3])

    row_indices, col_indices = sparse.indices().numpy()
    nzvalues = sparse.values().numpy().flatten()

    y: Tensor
    (y,) = poptorch.custom_op(
        [dense],
        name="StaticSparseMatmul",
        domain="ai.graphcore",
        domain_version=1,
        example_outputs=[
            torch.zeros(output_shape, dtype=dense.dtype, device=dense.device)
        ],
        attributes=dict(
            mode=mode,
            n_rows=sparse.shape[0] * sparse.shape[2],
            n_cols=sparse.shape[1] * sparse.shape[3],
            block_size=sparse.shape[2],
            # Poplar expects (row, col) element indices, not block indices
            rows=(sparse.shape[2] * row_indices).tolist(),
            cols=(sparse.shape[3] * col_indices).tolist(),
            values=nzvalues.tolist(),
        ),
    )
    return y


# High-level API


@dataclass
class StaticSparseMatrix:
    """Convenience wrapper for `dense @ sparse` or `sparse @ dense`."""

    matrix: Tensor

    def __matmul__(self, rhs: Tensor) -> Tensor:
        return block_coo_spmm(self.matrix, rhs, mode="sparse_dense")

    def __rmatmul__(self, lhs: Tensor) -> Tensor:
        return block_coo_spmm(self.matrix, lhs, mode="dense_sparse")


class StaticSparseLinear(torch.nn.Module):
    """
    A linear layer with frozen sparse parameters.

    weight -- shape `(out_features, in_features)`
    """

    def __init__(self, weight: Tensor):
        super().__init__()
        self.weight = weight

    def forward(self, x: Tensor) -> Tensor:
        return (StaticSparseMatrix(self.weight) @ x.T).T


def magnitude_prune(
    matrix: Tensor, block_size: int, density: float, ord: int = 2
) -> Tensor:
    """
    Basic single-tensor block magnitude pruning.

    Groups `matrix` into blocks of size `block_size x block_size`, then
    creates a sparse COO tensor retaining the largest-magnitude blocks
    such that `result.values().nelement() / matrix.nelement() <= density`.

    returns -- Tensor -- a COO block-sparse tensor of shape
               `(blocks_row, blocks_col, block_size, block_size)`
    """
    blocks = matrix.reshape(
        matrix.shape[0] // block_size,
        block_size,
        matrix.shape[1] // block_size,
        block_size,
    )
    norms = torch.linalg.norm(blocks.float(), ord=ord, dim=(1, 3))
    nnz_blocks = int(density * matrix.nelement() / block_size**2)
    topk = torch.topk(norms.view(-1), nnz_blocks).indices
    indices = torch.stack([topk // norms.shape[1], topk % norms.shape[1]])
    return torch.sparse_coo_tensor(
        indices=indices,
        values=blocks[indices[0], :, indices[1], :],
        size=blocks.permute(0, 2, 1, 3).shape,
    ).coalesce()


__all__ = [
    "block_coo_transpose",
    "block_coo_to_dense",
    "block_coo_spmm",
    "block_coo_spmm_gs",
    "block_coo_spmm_ipu",
    "StaticSparseMatrix",
    "StaticSparseLinear",
    "magnitude_prune",
]
