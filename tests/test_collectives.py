# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import random
from copy import deepcopy
from typing import Any, Callable, Tuple

import einops
import numpy as np
import poptorch
import pytest
import torch
import torch.nn as nn
from poptorch.enums import CommGroupType

import poptorch_experimental_addons as pea

from . import utils

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class _CollectiveCrossReplicaTester(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        replication_factor: int,
        cross_replica_op: Callable[[torch.Tensor, int], torch.Tensor],
    ):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor
        self.cross_replica_op = cross_replica_op

    def forward(self) -> Tuple[Any, Any]:
        out = self.cross_replica_op(self.X, self.replication_factor)
        return out, poptorch.identity_loss(out.pow(2), reduction="sum")


class _CollectiveSimulator(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        replication_factor: int,
        simulator_op: Callable[[torch.Tensor, int], torch.Tensor],
    ):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor
        self.simulator_op = simulator_op

    def forward(self) -> Tuple[Any, Any]:
        out = self.simulator_op(self.X, self.replication_factor)
        loss = out.pow(2).sum(dim=list(range(out.ndim))[1:])
        return torch.vstack([*out]), loss


def _all_gather_simulator_op(X: torch.Tensor, replication_factor: int) -> torch.Tensor:
    return einops.repeat(X, "s n d -> r s n d", r=replication_factor)  # r == s


def _all_reduce_simulator_op(X: torch.Tensor, replication_factor: int) -> torch.Tensor:
    out = einops.reduce(X, "r n d -> n d", "sum")
    return einops.repeat(out, "n d -> r n d", r=replication_factor)


def _all_to_all_simulator_op(X: torch.Tensor, replication_factor: int) -> torch.Tensor:
    return einops.rearrange(X, "r s d -> s r d", r=replication_factor)


_op_mapping = {
    pea.collectives.all_gather_cross_replica: _all_gather_simulator_op,
    pea.collectives.all_reduce_cross_replica_sum: _all_reduce_simulator_op,
    pea.collectives.all_to_all_single_cross_replica: _all_to_all_simulator_op,
}


def generate_input() -> torch.Tensor:
    return einops.rearrange(torch.arange(32, dtype=torch.float32), "(d n)  -> n d", n=4)


def simulate_collective(
    X: torch.Tensor, op: Callable[[torch.Tensor, int], torch.Tensor], num_ipus: int
) -> Tuple[Any, Any, Any]:
    sim = _CollectiveSimulator(
        deepcopy(X), replication_factor=num_ipus, simulator_op=op
    )
    lr = 1.0
    if op == _all_reduce_simulator_op:
        lr /= num_ipus
    optimizer = torch.optim.SGD(sim.parameters(), lr=lr)
    optimizer.zero_grad()
    out, loss = sim()
    loss.mean().backward()
    grad = einops.rearrange(sim.X.grad, "r n d -> (r n) d")
    optimizer.step()
    if op in set([_all_gather_simulator_op, _all_to_all_simulator_op]):
        grad *= num_ipus  # type: ignore
    return out, grad, sim.X.data


def run_collective(
    X: torch.Tensor, op: Callable[[torch.Tensor, int], torch.Tensor], num_ipus: int
) -> Tuple[Any, Any, Any]:
    options = poptorch.Options()
    options.replicationFactor(num_ipus)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options.anchorTensor("grad_X", "Gradient___X")
    options._Popart.setPatterns({"OpToIdentity": True})

    col = _CollectiveCrossReplicaTester(
        X, replication_factor=num_ipus, cross_replica_op=op
    )
    optimizer = poptorch.optim.SGD(col.parameters(), lr=1.0)
    col = poptorch.trainingModel(col, options, optimizer)
    utils._apply_replica_grouping(col, CommGroupType.Orthogonal, 1)
    out, _ = col()
    out = out.detach().cpu()
    grad = col.getAnchoredTensor("grad_X")  # type: ignore
    return out, grad, col.X.data


@pytest.mark.parametrize("op", list(_op_mapping.keys()))
def test_collective(op: Callable[[torch.Tensor, int], torch.Tensor]) -> None:
    X = generate_input()
    num_ipus = 2
    actual = run_collective(X, op, num_ipus)
    expected = simulate_collective(X, _op_mapping[op], num_ipus)
    list(map(assert_close, actual, expected))
