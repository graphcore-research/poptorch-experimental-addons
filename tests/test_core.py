# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import collections
from typing import Callable, Dict, Tuple

import poptorch
import pytest
import torch
from torch import Tensor, nn

import poptorch_experimental_addons as pea


def run_forward_and_backward(
    fn: Callable[..., Dict[str, Tensor]],
    inputs: Dict[str, Tensor],
    device: str,
    grad_outputs: Dict[str, Tensor] = {},
    patterns: Dict[str, bool] = {},
) -> Dict[str, Tensor]:
    class TestModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for k, v in inputs.items():
                self.register_parameter(k, nn.Parameter(v.clone()))

        def forward(self) -> Tuple[Dict[str, Tensor], Tensor]:
            outputs = fn(**{k: getattr(self, k) for k in inputs})
            loss = poptorch.identity_loss(
                sum(
                    torch.sum(v * grad_outputs.get(k, torch.ones_like(v)).to(v.device))
                    for k, v in outputs.items()
                ),
                reduction="none",
            )
            return outputs, loss

    module = TestModule()
    optimiser = torch.optim.SGD(module.parameters(), 1.0)
    if device == "ipu":
        options = poptorch.Options()
        options.useIpuModel(not poptorch.ipuHardwareIsAvailable())
        options._popart.setPatterns(patterns)
        step = poptorch.trainingModel(module, options, optimiser)
        output, _ = step()
        step.copyWeightsToHost()
    else:
        optimiser.zero_grad()
        output, loss = module()
        loss.backward()
        optimiser.step()
    with torch.no_grad():
        return dict(
            **output,
            **{f"grad_{k}": inputs[k] - getattr(module, k) for k in inputs},
        )


@pytest.mark.parametrize("device", ["cpu", "ipu"])
def test_autograd_proxy(device: str) -> None:
    outputs = run_forward_and_backward(
        lambda x: dict(y=pea.autograd_proxy(torch.round(x), 3 * x)),
        dict(x=torch.tensor(5.7)),
        grad_outputs=dict(y=torch.tensor(100.0)),
        patterns=dict(AutogradProxyOpPattern=True),
        device=device,
    )
    torch.testing.assert_close(outputs["y"], torch.tensor(6.0))
    torch.testing.assert_close(outputs["grad_x"], torch.tensor(300.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("p", [1, 2])
def test_distance_matrix(p: int, dtype: torch.dtype) -> None:
    torch.manual_seed(1234)
    M, N, K = 10, 30, 5
    tensor1 = torch.randn(size=(M, K), dtype=dtype)
    tensor2 = torch.randn(size=(N, K), dtype=dtype)

    output_ipu = run_forward_and_backward(
        lambda tensor1, tensor2: dict(y=pea.distance_matrix(tensor1, tensor2, p)),
        dict(tensor1=tensor1, tensor2=tensor2),
        device="ipu",
    )
    output_torch = run_forward_and_backward(
        lambda tensor1, tensor2: dict(
            y=(tensor1[:, None] - tensor2[None, :]).norm(p=p, dim=-1)
        ),
        dict(tensor1=tensor1, tensor2=tensor2),
        device="cpu",
    )

    atol = {torch.float32: 1e-5, torch.float16: 2e-3}[dtype]
    rtol = {torch.float32: 2e-6, torch.float16: 2e-3}[dtype]

    torch.testing.assert_close(
        output_ipu["y"],
        output_torch["y"],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        output_ipu["grad_tensor1"],
        output_torch["grad_tensor1"],
        rtol=rtol,
        atol=atol,
    )
    torch.testing.assert_close(
        output_ipu["grad_tensor2"],
        output_torch["grad_tensor2"],
        rtol=rtol,
        atol=atol,
    )


def test_quantisation_cpu() -> None:
    assert set(
        pea.quantise_fpx(
            torch.linspace(-4, 4, steps=100),
            exponent_bits=2,
            mantissa_bits=1,
            rounding="nearest",
        ).tolist()
    ) == {sx for x in [0, 0.25, 0.5, 0.75, 1, 1.5, 2, 3] for sx in [x, -x]}

    assert set(
        pea.quantise_fpx(
            torch.linspace(-10, 10, steps=1000),
            exponent_bits=3,
            mantissa_bits=0,
            rounding="nearest",
        )
        .abs()
        .tolist()
    ) == {0, 0.125, 0.25, 0.5, 1, 2, 4, 8}


@pytest.mark.parametrize("device", ["cpu", "ipu"])
def test_quantisation_stochastic_rounding(device: str) -> None:
    if device == "ipu" and not poptorch.ipuHardwareIsAvailable():
        pytest.skip("quantise_fpx() requires IPU hardware")

    n = 10000
    x, grad_y = -1.35, 3.0
    out = run_forward_and_backward(
        lambda x: dict(
            y=pea.quantise_fpx(
                x, exponent_bits=2, mantissa_bits=1, rounding="stochastic", bwd=True
            )
        ),
        dict(x=torch.full((n,), x)),
        grad_outputs=dict(y=torch.full((n,), grad_y)),
        device=device,
    )

    y_count = collections.Counter(out["y"].tolist())
    assert y_count.keys() == {-1.5, -1.0}
    expected_ratio = (1.35 - 1.0) / 0.5
    nearest_ratio = y_count[-1.5] / sum(y_count.values())
    std_x3 = 3 * (expected_ratio * (1 - expected_ratio) / n) ** 0.5
    assert expected_ratio - std_x3 < nearest_ratio < expected_ratio + std_x3

    assert collections.Counter(out["grad_x"].tolist()) == {3.0: n}


@pytest.mark.parametrize("device", ["cpu", "ipu"])
def test_quantisation_variants(device: str) -> None:
    if device == "ipu" and not poptorch.ipuHardwareIsAvailable():
        pytest.skip("quantise_fpx() requires IPU hardware")

    # Note: some gymnastics here to get everything into one test, for efficiency
    def _fn(**args: Tensor) -> Dict[str, Tensor]:
        return {
            f"y{suffix}": quantise(  # type:ignore[operator]
                args[f"x{suffix}"],
                exponent_bits=5,
                mantissa_bits=2,
                rounding="nearest",
            )
            for suffix, quantise in [
                ("", pea.quantise_fpx),
                ("_ste", pea.quantise_fpx_ste),
                ("_grad_only", pea.quantise_fpx_grad),
            ]
        }

    x = torch.linspace(-1e5, 1e5, steps=1000)
    grad_y = torch.flip(x, (0,))  # different grads, same range
    out = run_forward_and_backward(
        _fn,
        {name: x for name in ["x", "x_ste", "x_grad_only"]},
        grad_outputs={name: grad_y for name in ["y", "y_ste", "y_grad_only"]},
        device=device,
    )

    def assert_quantised(t: Tensor) -> None:
        assert len(set(t.tolist())) <= 256
        torch.testing.assert_close(t.max(), torch.tensor(57344.0))
        torch.testing.assert_close(t.min(), torch.tensor(-57344.0))

    # quantise_fpx
    assert_quantised(out["y"])
    assert torch.equal(out["grad_x"], torch.zeros_like(x))

    # quantise_fpx_ste
    assert_quantised(out["y_ste"])
    assert torch.equal(out["grad_x_ste"], grad_y)

    # quantise_fpx_grad
    assert torch.equal(out["y_grad_only"], x)
    assert_quantised(out["grad_x_grad_only"])


def test_quantisation_invalid_settings() -> None:
    with pytest.raises(ValueError, match="exponent_bits=9"):
        pea.quantise_fpx(torch.zeros((1,)), exponent_bits=9, mantissa_bits=2)
    with pytest.raises(ValueError, match="mantissa_bits=24"):
        pea.quantise_fpx(torch.zeros((1,)), exponent_bits=8, mantissa_bits=24)
    with pytest.raises(ValueError, match="exponent_bits=6"):
        run_forward_and_backward(
            lambda x: dict(y=pea.quantise_fpx(x, exponent_bits=6, mantissa_bits=2)),
            dict(x=torch.zeros((1,))),
            device="ipu",
        )
