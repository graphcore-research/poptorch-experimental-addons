# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import numpy as np
import poptorch
import pytest
import torch
from torch import Tensor

import poptorch_experimental_addons as pea


@dataclass
class Problem:
    sparse: Tensor
    dense: Tensor
    mode: str
    expected_output: Tensor

    @classmethod
    def generate_sparse_dense(
        cls,
        block_size: int,
        out_blocks: int,
        in_blocks: int,
        batch_size: int,
        density: float,
        dtype: torch.dtype,
    ) -> "Problem":
        nnz_blocks = int(density * in_blocks * out_blocks)
        lhs = torch.sparse_coo_tensor(
            indices=torch.tensor(np.indices((out_blocks, in_blocks))).reshape(2, -1)[
                :, torch.randperm(out_blocks * in_blocks)[:nnz_blocks]
            ],
            values=torch.randn(size=(nnz_blocks, block_size, block_size), dtype=dtype),
            size=(out_blocks, in_blocks, block_size, block_size),
        ).coalesce()
        rhs = torch.randn((in_blocks * block_size, batch_size), dtype=dtype)
        output = pea.sparse.block_coo_to_dense(lhs).float() @ rhs.float()
        return cls(sparse=lhs, dense=rhs, mode="sparse_dense", expected_output=output)

    def transpose(self) -> "Problem":
        return Problem(
            sparse=pea.sparse.block_coo_transpose(self.sparse),
            dense=self.dense.T,
            mode="dense_sparse" if self.mode == "sparse_dense" else "sparse_dense",
            expected_output=self.expected_output.T,
        )

    def __str__(self) -> str:
        return (
            f"Problem(sparse {self.sparse.shape}, dense {self.dense.shape},"
            f" mode: {self.mode}, dtype: {self.dense.dtype})"
        )


@pytest.mark.parametrize(
    "shape,block_size,seed",
    [
        ((2, 3), 1, 100),
        ((3, 5, 7), 4, 200),
    ],
)
def test_coo_methods(shape: Tuple[int], block_size: int, seed: int) -> None:
    torch.manual_seed(seed)
    nnz_blocks = int(np.prod(shape)) // 2
    array = torch.sparse_coo_tensor(
        indices=torch.tensor(np.indices(shape).reshape(len(shape), -1))[
            :, torch.randperm(int(np.prod(shape)))[:nnz_blocks]
        ],
        values=1000
        + torch.arange(nnz_blocks * block_size ** len(shape)).reshape(
            (-1,) + (block_size,) * len(shape)
        ),
        size=shape + (block_size,) * len(shape),
    )
    dense = pea.sparse.block_coo_to_dense(array)
    assert dense.shape == tuple(s * block_size for s in shape)
    assert torch.sum(dense != 0) == nnz_blocks * block_size ** len(shape)

    if len(shape) == 2:
        torch.testing.assert_close(
            pea.sparse.block_coo_to_dense(pea.sparse.block_coo_transpose(array)),
            dense.T,
        )


class LambdaModule(torch.nn.Module):
    def __init__(self, fn: Callable[..., Any]):
        super().__init__()
        self.fn = fn

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        return self.fn(*args, **kwargs)


@pytest.mark.parametrize(
    "block_size,shape,batch_size,density,dtype,seed",
    [
        (1, (3, 5), 7, 0.4, torch.float, 1000),
        (4, (12, 10), 8, 0.2, torch.half, 2000),
        (1, (128, 128), 1, 0.01, torch.float, 3000),
    ],
)
def test_block_coo_spmm(
    block_size: int,
    shape: Tuple[int, int],
    batch_size: int,
    density: float,
    dtype: torch.dtype,
    seed: int,
) -> None:
    torch.manual_seed(seed)
    base_problem = Problem.generate_sparse_dense(
        block_size=block_size,
        out_blocks=shape[0],
        in_blocks=shape[1],
        batch_size=batch_size,
        density=density,
        dtype=dtype,
    )
    for problem in [base_problem, base_problem.transpose()]:
        outputs = {}
        if dtype != torch.half:
            outputs["gather_scatter"] = pea.sparse.block_coo_spmm_gs(
                problem.sparse, problem.dense, problem.mode
            )
        options = poptorch.Options()
        options.useIpuModel(True)
        outputs["ipu"] = poptorch.inferenceModel(
            LambdaModule(
                lambda dense: pea.sparse.block_coo_spmm_ipu(
                    problem.sparse, dense, problem.mode
                )
            ),
            options=options,
        )(problem.dense)

        for name, output in outputs.items():
            torch.testing.assert_close(
                output.float(),
                problem.expected_output,
                rtol=0,
                atol={torch.float: 1e-5, torch.half: 1e-2}[dtype],
                msg=f"{name} (shape {output.shape}) vs dense"
                f" (shape {problem.expected_output.shape})"
                f", for {problem}",
            )


def test_high_level_api() -> None:
    torch.manual_seed(4000)
    block_size = 2
    out_size = 3 * block_size
    in_size = 5 * block_size

    sparse = pea.sparse.magnitude_prune(
        torch.randn(out_size, in_size),
        block_size=block_size,
        density=0.75,
    )
    dense_in = torch.randn(7, in_size)
    expected_output = dense_in @ pea.sparse.block_coo_to_dense(sparse).T

    def check(output: Tensor) -> None:
        torch.testing.assert_close(output, expected_output, atol=1e-5, rtol=0)

    check(pea.sparse.StaticSparseLinear(sparse)(dense_in))
    check((pea.sparse.StaticSparseMatrix(sparse) @ dense_in.T).T)
    check(
        dense_in @ pea.sparse.StaticSparseMatrix(pea.sparse.block_coo_transpose(sparse))
    )
