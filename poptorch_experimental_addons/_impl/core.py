# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Top-level utilities.
"""

from typing import Any, Tuple

import poptorch
import torch
from torch import Tensor


class _CustomGrad(torch.autograd.Function):
    @staticmethod
    def forward(  # type:ignore[override]
        ctx: Any, fwd: Tensor, fwd_surrogate: Tensor
    ) -> Tensor:
        return fwd

    @staticmethod
    def backward(  # type:ignore[override]
        ctx: Any, grad: Tensor
    ) -> Tuple[None, Tensor]:
        return None, grad


def custom_grad(fwd: Tensor, fwd_surrogate: Tensor) -> Tensor:
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

    Note that `fwd`, `fwd_surrogate` and the output all have the same shape.
    """
    if fwd.shape != fwd_surrogate.shape:
        raise ValueError(
            f"custom_grad expects both arguments to have the same shape, actual:"
            f"fwd.shape: {fwd.shape}, fwd_surrogate.shape: {fwd_surrogate.shape}"
        )
    y: Tensor
    if poptorch.isRunningOnIpu():
        (y,) = poptorch.custom_op(
            [fwd, fwd_surrogate],
            name="CustomGradient",
            domain="ai.graphcore.pea",
            domain_version=1,
            example_outputs=[fwd],
        )
    else:
        y = _CustomGrad.apply(fwd, fwd_surrogate)
    return y


__all__ = ["custom_grad"]
