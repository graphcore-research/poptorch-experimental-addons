from typing import Any, Tuple
import poptorch
import torch
import pytest

import poptorch_experimental_addons as pea

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


class BroadcastedPairwiseDistanceTester(torch.nn.Module):
    def __init__(self, tensor1: torch.Tensor, tensor2: torch.Tensor, p: int):
        super().__init__()
        self.p = p
        self.tensor1 = torch.nn.Parameter(tensor1.contiguous())
        self.tensor2 = torch.nn.Parameter(tensor2.contiguous())

    def forward(self) -> Tuple[Any, Any]:
        dist = pea.ops.broadcasted_pairwise_distance(
            self.tensor1, self.tensor2, p=self.p
        )

        return dist, poptorch.identity_loss(dist, reduction="sum")


class BroadcastedPairwiseDistanceSimulator(torch.nn.Module):
    def __init__(self, tensor1: torch.Tensor, tensor2: torch.Tensor, p: int):
        super().__init__()
        self.p = p
        self.tensor1 = torch.nn.Parameter(tensor1.contiguous())
        self.tensor2 = torch.nn.Parameter(tensor2.contiguous())

    def forward(self) -> Tuple[Any, Any]:
        dist = torch.cdist(self.tensor1, self.tensor2, p=self.p)

        return dist, torch.sum(dist)


@pytest.mark.parametrize("p", [1, 2])
def test_broadcasted_pairwise_distance(p: int) -> None:
    torch.manual_seed(1135)
    M = 10
    N = 30
    K = 50
    tensor1 = 10 + 20 * torch.randn(size=(M, K), dtype=torch.float32)
    tensor2 = -10 + 10 * torch.randn(size=(N, K), dtype=torch.float32)

    model = BroadcastedPairwiseDistanceTester(tensor1, tensor2, p=p)
    optimizer = poptorch.optim.SGD(model.parameters(), lr=1.0)
    options = poptorch.Options()
    options.useIpuModel(True)
    options.anchorTensor("grad_tensor1", "Gradient___tensor1")
    options.anchorTensor("grad_tensor2", "Gradient___tensor2")
    options._Popart.setPatterns({"OpToIdentity": True})

    ipu_model = poptorch.trainingModel(model, options, optimizer)
    torch_model = BroadcastedPairwiseDistanceSimulator(tensor1, tensor2, p=p)

    ipu_distance, _ = ipu_model()
    torch_distance, torch_loss = torch_model()

    assert_close(ipu_distance.detach().cpu(), torch_distance.detach().cpu())

    torch_loss.backward()
    torch_grad1 = torch_model.tensor1.grad
    torch_grad2 = torch_model.tensor2.grad
    ipu_grad1 = ipu_model.getAnchoredTensor("grad_tensor1")
    ipu_grad2 = ipu_model.getAnchoredTensor("grad_tensor2")

    assert_close(ipu_grad1.detach().cpu(), torch_grad1.detach().cpu())
    assert_close(ipu_grad2.detach().cpu(), torch_grad2.detach().cpu())
