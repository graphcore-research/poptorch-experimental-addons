# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import random
from copy import deepcopy
from typing import Any, Tuple

import einops
import numpy as np
import poptorch
import torch
import torch.nn as nn
from poptorch.enums import CommGroupType

import poptorch_experimental_addons as pea

from .utils import _apply_replica_grouping

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class _AllReduceCrossReplicaTester(torch.nn.Module):
    def __init__(self, X: torch.Tensor, replication_factor: int):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor

    def forward(self) -> Tuple[Any, Any]:
        out = pea.collectives.all_reduce_cross_replica_sum(
            self.X, self.replication_factor
        )
        return out, poptorch.identity_loss(out.pow(2), reduction="sum")


class _AllReduceSimulator(torch.nn.Module):
    def __init__(self, X: torch.Tensor, replication_factor: int):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor

    def forward(self) -> Tuple[Any, Any]:
        out = einops.reduce(self.X, "r n d -> n d", "sum")
        out = einops.repeat(out, "n d -> r n d", r=self.replication_factor)
        loss = out.pow(2).sum(dim=list(range(out.ndim))[1:])
        return torch.vstack([*out]), loss


def test_all_reduce() -> None:
    set_seed(112358)
    num_ipus = 2

    X = einops.rearrange(torch.arange(32, dtype=torch.float32), "(d n) -> n d", n=4)

    sim = _AllReduceSimulator(deepcopy(X), replication_factor=num_ipus)
    out_true, loss = sim()
    # loss.sum().backward()  # sum losses across IPUs to generate gradients
    loss.mean().backward()  # average losses across IPUs to generate gradients
    grad_true = einops.rearrange(sim.X.grad, "r n d -> (r n) d")

    options = poptorch.Options()
    options.replicationFactor(num_ipus)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options.anchorTensor("grad_X", "Gradient___X")
    options._Popart.setPatterns({"OpToIdentity": True})

    model = _AllReduceCrossReplicaTester(deepcopy(X), num_ipus)
    optimizer = poptorch.optim.SGD(model.parameters(), lr=1.0)
    model = poptorch.trainingModel(model, options, optimizer)
    _apply_replica_grouping(model, CommGroupType.Orthogonal, 1)

    out_actual, _ = model()
    out_actual = out_actual.detach().cpu()

    assert_close(out_actual, out_true, rtol=0, atol=0)

    grad_actual = model.getAnchoredTensor("grad_X")  # type: ignore
    assert_close(grad_actual, grad_true, rtol=0, atol=0)
