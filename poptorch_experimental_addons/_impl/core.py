# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Top-level utilities.
"""

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
            domain="ai.graphcore.pea",
            domain_version=1,
            example_outputs=[fwd],
        )
    else:
        y = _AutogradProxy.apply(fwd, proxy)
    return y


__all__ = ["autograd_proxy"]
