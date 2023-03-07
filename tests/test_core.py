# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Callable, Dict

import poptorch
import pytest
import torch
from torch import Tensor, nn

import poptorch_experimental_addons as pea

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


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

        def forward(self) -> Tensor:
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
    assert_close(outputs["loss"], torch.tensor(6.0))
    assert_close(outputs["grad_x"], torch.tensor(3.0))


@pytest.mark.parametrize("p", [1, 2])
def test_distance_matrix(p: int) -> None:
    M, N, K = 10, 30, 50
    tensor1 = 10 + 20 * torch.randn(size=(M, K), dtype=torch.float32)
    tensor2 = -10 + 10 * torch.randn(size=(N, K), dtype=torch.float32)

    output_ipu = run_forward_and_backward(
        lambda tensor1, tensor2: pea.distance_matrix(tensor1, tensor2, p),
        dict(tensor1=tensor1, tensor2=tensor2),
        patterns={},
        device="ipu",
    )
    output_torch = run_forward_and_backward(
        lambda tensor1, tensor2: torch.cdist(tensor1, tensor2, p),
        dict(tensor1=tensor1, tensor2=tensor2),
        patterns={},
        device="cpu",
    )

    assert_close(output_ipu["output"], output_torch["output"])
    assert_close(output_ipu["grad_tensor1"], output_torch["grad_tensor1"])
    assert_close(output_ipu["grad_tensor2"], output_torch["grad_tensor2"])
