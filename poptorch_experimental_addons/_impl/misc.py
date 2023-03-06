# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Small standalone utilities that don't belong in larger groups.
"""

from typing import Any, Tuple

import poptorch
import torch
from torch import Tensor


class _CustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, fwd: Tensor, bwd: Tensor) -> Tensor:  # type:ignore[override]
        return fwd

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: Any, grad: Tensor
    ) -> Tuple[None, Tensor]:
        return None, grad


def custom_grad(fwd: Tensor, bwd: Tensor) -> Tensor:
    """
    Return one tensor in the forward pass, using a separate tensor for the
    backward pass.

    Typically, used `y = custom_grad(f(x), g(x))`, in which case the forward pass
    uses `f`, such that `y = f(x)`, and the backward pass uses `g`, such that
    `dy/dx = dg/dx`.

    For example, a straight-through estimator for `round`:
    ```python
    y = custom_grad(torch.round(x), x)
    ```

    Note that `fwd`, `bwd` and the output all have the same shape.
    """
    if fwd.shape != bwd.shape:
        raise ValueError(
            f"custom_grad expects both arguments to have the same shape"
            f", actual: fwd.shape: {fwd.shape}, bwd.shape: {bwd.shape}"
        )
    y: Tensor
    if poptorch.isRunningOnIpu():
        (y,) = poptorch.custom_op(
            [fwd, bwd],
            name="CustomGradient",
            domain="ai.graphcore.pea",
            domain_version=1,
            example_outputs=[fwd],
        )
    else:
        y = _CustomGrad.apply(fwd, bwd)
    return y


def scaling(x: Tensor, fwd_scale: float, bwd_scale: float) -> Tensor:
    """
    Scale `x` by `fwd_scale` in the forward pass, and the gradient by `bwd_scale`
    in the backward pass.
    """
    return custom_grad(x * fwd_scale, x * bwd_scale)


__all__ = ["custom_grad", "scaling"]
