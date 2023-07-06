# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable, Dict, cast

import poptorch
import pytest
import torch
from torch import Tensor, nn

import poptorch_experimental_addons as pea


def run_forward_and_backward(
    fn: Callable[..., Tensor],
    args: Dict[str, Tensor],
    patterns: Dict[str, bool],
    device: str,
) -> Dict[str, Tensor]:
    class TestModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            for k, v in args.items():
                self.register_parameter(k, nn.Parameter(v.clone()))

        def forward(self) -> Dict[str, Tensor]:
            output = fn(**{k: getattr(self, k) for k in args})
            loss: Tensor = poptorch.identity_loss(output, reduction="sum")
            return dict(output=output, loss=loss)

    module = TestModule()
    optimiser = torch.optim.SGD(module.parameters(), 1.0)
    if device == "ipu":
        options = poptorch.Options()
        options.useIpuModel(True)
        options._popart.setPatterns(patterns)
        step = poptorch.trainingModel(module, options, optimiser)
        output = step()
        step.copyWeightsToHost()
    else:
        optimiser.zero_grad()
        output = module()
        output["loss"].backward()
        optimiser.step()
    return dict(
        **output,
        **{f"grad_{k}": args[k] - getattr(module, k).detach() for k in args},
    )


@pytest.mark.parametrize("device", ["cpu", "ipu"])
def test_autograd_proxy(device: str) -> None:
    outputs = run_forward_and_backward(
        lambda x: pea.autograd_proxy(torch.round(x), 3 * x),
        dict(x=torch.tensor(5.7)),
        patterns=dict(AutogradProxyOpPattern=True),
        device=device,
    )
    torch.testing.assert_close(outputs["loss"], torch.tensor(6.0))
    torch.testing.assert_close(outputs["grad_x"], torch.tensor(3.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
@pytest.mark.parametrize("p", [1, 2])
def test_distance_matrix(p: int, dtype: torch.dtype) -> None:
    torch.manual_seed(1234)
    M, N, K = 10, 30, 5
    tensor1 = torch.randn(size=(M, K), dtype=dtype)
    tensor2 = torch.randn(size=(N, K), dtype=dtype)

    output_ipu = run_forward_and_backward(
        lambda tensor1, tensor2: pea.distance_matrix(tensor1, tensor2, p),
        dict(tensor1=tensor1, tensor2=tensor2),
        patterns={},
        device="ipu",
    )
    output_torch = run_forward_and_backward(
        lambda tensor1, tensor2: cast(
            torch.Tensor, (tensor1[:, None] - tensor2[None, :]).norm(p=p, dim=-1)
        ),
        dict(tensor1=tensor1, tensor2=tensor2),
        patterns={},
        device="cpu",
    )

    atol = {torch.float32: 1e-5, torch.float16: 2e-3}[dtype]
    rtol = {torch.float32: 2e-6, torch.float16: 2e-3}[dtype]

    torch.testing.assert_close(
        output_ipu["output"],
        output_torch["output"],
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
