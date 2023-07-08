# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Top-level utilities.
"""

from pathlib import Path
from typing import Any, Optional, Tuple

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
        y = _AutogradProxy.apply(fwd, proxy)  # type:ignore[no-untyped-call]
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
            attributes=dict(root_path=str(Path(__file__).parent.parent.absolute())),
        )
    else:
        y = torch.cdist(tensor1, tensor2, p=p)

    return y


def quantise_fpx(
    x: Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    rounding: str = "stochastic",
    fwd: bool = True,
    bwd: Optional[bool] = None,
) -> Tensor:
    """
    Quantise the values in a tensor to a lower-precision floating point format.
    Note that this is not a cast; the returned tensor has the same dtype as the input.

        quantise_fpx(tensor(0.2), exponent_bits=2, mantissa_bits=1, rounding="nearest")
           => 0.25

    By default, quantise in the forward pass and return no gradient.

    exponent_bits, mantissa_bits -- define the FP format (total bits = 1 (sign) + E + M)

    rounding -- either "nearest" or "stochastic"

    fwd -- whether to quantise the forward value

    bwd -- whether to generate & whether to quantise the gradient:
           bwd=None  -- no gradient
           bwd=False -- unquantised gradient (straight-through estimator)
           bwd=True  -- quantised gradient
    """
    if rounding not in ["nearest", "stochastic"]:
        raise ValueError(
            "Expected quantise(rounding=?) to be 'nearest' or 'stochastic'"
            f", actual '{rounding}'"
        )

    if poptorch.isRunningOnIpu():
        max_exponent_bits = 5
        max_mantissa_bits = 10
    else:
        max_exponent_bits = 8
        max_mantissa_bits = 23
    if exponent_bits > max_exponent_bits:
        raise ValueError(
            f"quantise_fpx(exponent_bits={exponent_bits}) not supported, maximum"
            f" number of exponent bits is {max_exponent_bits}"
        )
    if mantissa_bits > max_mantissa_bits:
        raise ValueError(
            f"quantise_fpx(mantissa_bits={mantissa_bits}) not supported, maximum"
            f" number of mantissa bits is {max_mantissa_bits}"
        )

    q: Tensor
    if poptorch.isRunningOnIpu():
        (q,) = poptorch.custom_op(
            name="SimulatedQuant",
            domain_version=1,
            domain="ai.graphcore",
            inputs=[x],
            example_outputs=[x],
            attributes=dict(
                root_path=str(Path(__file__).parent.parent.absolute()),
                exponent_bits=exponent_bits,
                mantissa_bits=mantissa_bits,
                rounding=rounding,
                fwd=fwd,
                bwd={True: "quantise", False: "ste", None: "stop"}[bwd],
            ),
        )
        return q

    def _quantise(x: Tensor) -> Tensor:
        max_exponent = 2 ** (exponent_bits - 1) - 1
        absmax = 2**max_exponent * (2 - 2**-mantissa_bits)
        downscale = 2.0 ** (126 - max_exponent)
        mask = torch.tensor(
            2 ** (23 - mantissa_bits) - 1, dtype=torch.int32, device=x.device
        )
        offset = (
            torch.randint(  # type:ignore[call-overload]
                0, mask + 1, x.shape, dtype=torch.int32, device=x.device
            )
            if rounding == "stochastic"
            else mask // 2
        )
        # Manually clip to max
        # Then scale down (to generate appropriate subnormals) & mask off mantissa bits
        q = x.to(torch.float32)
        q = torch.clip(x, -absmax, absmax)
        q /= downscale
        q = ((q.view(torch.int32) + offset) & ~mask).view(torch.float32)
        q *= downscale
        q = q.to(x.dtype)
        return q

    class F(torch.autograd.Function):
        @staticmethod
        def forward(  # type:ignore[override]
            ctx: torch.autograd.function.FunctionCtx, xx: Tensor
        ) -> Tensor:
            return _quantise(xx) if fwd else xx.clone()

        @staticmethod
        def backward(  # type:ignore[override]
            ctx: torch.autograd.function.FunctionCtx, grad_y: Tensor
        ) -> Optional[Tensor]:
            if bwd is not None:
                return _quantise(grad_y) if bwd else grad_y
            return None

    q = F.apply(x)  # type:ignore[no-untyped-call]
    return q


def quantise_fpx_ste(
    x: Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    rounding: str = "stochastic",
) -> Tensor:
    """
    Quantise the forward value while leaving the gradient unchanged, as a
    straight-through estimator.

    See `quantise_fpx` for more detail.
    """
    return quantise_fpx(
        x,
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        rounding=rounding,
        fwd=True,
        bwd=False,
    )


def quantise_fpx_grad(
    x: Tensor,
    exponent_bits: int,
    mantissa_bits: int,
    rounding: str = "stochastic",
) -> Tensor:
    """
    Quantise the gradient while leaving the forward value unchanged.

    See `quantise_fpx` for more detail.
    """
    return quantise_fpx(
        x,
        exponent_bits=exponent_bits,
        mantissa_bits=mantissa_bits,
        rounding=rounding,
        fwd=False,
        bwd=True,
    )


__all__ = [
    "autograd_proxy",
    "distance_matrix",
    "quantise_fpx",
    "quantise_fpx_ste",
    "quantise_fpx_grad",
]
