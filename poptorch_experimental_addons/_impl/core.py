# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Top-level utilities.
"""

from pathlib import Path
from typing import Any, Tuple

import poptorch
import torch
from torch import Tensor


class _AutogradProxy(torch.autograd.Function):
    @staticmethod
    def forward(  # type:ignore[override]
        ctx: Any, fwd: Tensor, proxy: Tensor
    ) -> Tensor:
        return fwd

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: Any, grad: Tensor
    ) -> Tuple[None, Tensor]:
        return None, grad


def autograd_proxy(fwd: Tensor, proxy: Tensor) -> Tensor:
    """
    Return one tensor in the forward pass, using a separate tensor for the
    backward pass.

    Typically, used `y = autograd_proxy(f(x), g(x))`, in which case the forward pass
    uses `f`, such that `y = f(x)`, and the backward pass uses `g`, such that
    `dy/dx = dg/dx`.

    For example, a straight-through estimator for `round`:
    ```python
    y = autograd_proxy(torch.round(x), x)
    ```

    Note that `fwd`, `proxy` and the output all have the same shape.
    """
    if fwd.shape != proxy.shape:
        raise ValueError(
            f"autograd_proxy expects both arguments to have the same shape, actual:"
            f"fwd.shape: {fwd.shape}, proxy.shape: {proxy.shape}"
        )
    y: Tensor
    if poptorch.isRunningOnIpu():
        (y,) = poptorch.custom_op(
            [fwd, proxy],
            name="AutogradProxy",
            domain="ai.graphcore",
            domain_version=1,
            example_outputs=[fwd],
        )
    else:
        y = _AutogradProxy.apply(fwd, proxy)
    return y


def distance_matrix(tensor1: Tensor, tensor2: Tensor, p: int) -> Tensor:
    """
    p-norm broadcasted pairwise distance between two collections of vectors.

    Computes p-norm reduction along trailing dimension of
    tensor1[:,None,:] - tensor2[None,:,:] without materializing the intermediate
    broadcasted difference, for memory optimization.

    tensor1 -- shape (M, K)
    tensor2 -- shape (N, K)
    returns --  shape (M, N)
    """
    if p not in [1, 2]:
        raise NotImplementedError("distance_matrix implemented only for p=1,2")

    if tensor1.dim() != 2 or tensor2.dim() != 2:
        raise ValueError(
            "distance_matrix requires 2-dimensional inputs"
            f" `tensor1` (dim = {tensor1.dim()}) and `tensor2` (dim = {tensor2.dim()})"
        )

    if tensor1.shape[-1] != tensor2.shape[-1]:
        raise ValueError(
            "distance_matrix requires rightmost dimension of same size"
            f" for `tensor1` ({tensor1.shape[-1]}) and `tensor2` ({tensor2.shape[-1]})"
        )

    y: Tensor
    if poptorch.isRunningOnIpu():
        (y,) = poptorch.custom_op(
            name=f"L{p}Distance",
            domain_version=1,
            domain="ai.graphcore",
            inputs=[tensor1, tensor2],
            example_outputs=[
                torch.zeros(
                    dtype=tensor1.dtype,
                    size=[tensor1.shape[0], tensor2.shape[0]],
                    device=tensor1.device,
                )
            ],
            attributes=dict(root_path=str(Path(__file__).parent.parent)),
        )
    else:
        y = torch.cdist(tensor1, tensor2, p=p)

    return y


__all__ = ["autograd_proxy", "distance_matrix"]
