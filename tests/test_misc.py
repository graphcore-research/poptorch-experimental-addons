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
            return loss

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
        output.backward()
        optimiser.step()
    return dict(
        output=output,
        **{f"grad_{k}": args[k] - getattr(module, k).detach() for k in args},
    )


@pytest.mark.parametrize("device", ["cpu", "ipu"])
def test_custom_grad(device: str) -> None:
    outputs = run_forward_and_backward(
        lambda x: pea.misc.custom_grad(torch.round(x), 3 * x),
        dict(x=torch.tensor(5.7)),
        patterns=dict(CustomGradientOpPatten=True),
        device=device,
    )
    assert_close(outputs["output"], torch.tensor(6.0))
    assert_close(outputs["grad_x"], torch.tensor(3.0))
