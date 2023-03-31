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
from poptorch_experimental_addons.collectives import ReplicaGroupingInfo

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
        rg_info: ReplicaGroupingInfo,
        cross_replica_op: Callable[[torch.Tensor, ReplicaGroupingInfo], torch.Tensor],
    ):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=rg_info.group_size).contiguous()
        )
        self.rg_info = rg_info
        self.cross_replica_op = cross_replica_op

    def forward(self) -> Tuple[Any, Any]:
        out = self.cross_replica_op(self.X, self.rg_info)
        return out, poptorch.identity_loss(out.pow(2), reduction="sum")


class _CollectiveSimulator(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        rg_info: ReplicaGroupingInfo,
        simulator_op: Callable[[torch.Tensor, ReplicaGroupingInfo], torch.Tensor],
    ):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(g n) d -> g n d", g=rg_info.group_size).contiguous()
        )
        self.rg_info = rg_info
        self.simulator_op = simulator_op

    def forward(self) -> Tuple[Any, Any]:
        self.Xr = einops.repeat(self.X, "g n d -> r g n d", r=self.rg_info.num_groups)
        self.Xr.register_hook(utils._store_grad(self.Xr))
        out = self.simulator_op(self.Xr, self.rg_info)
        loss = out.pow(2).sum(dim=list(range(out.ndim))[1:])
        return torch.vstack([*torch.vstack([*out])]), loss


def _all_gather_simulator_op(
    X: torch.Tensor, rg_info: ReplicaGroupingInfo
) -> torch.Tensor:
    # X repeated along first dimension, sharded along second dimension
    return einops.repeat(X, "r s n d -> r g s n d", g=rg_info.group_size)  # g == s
    # return einops.repeat(out, "g s n d -> r g s n d", r=rg_info.num_groups)  # g == s
    # return einops.repeat(X, "s n d -> r s n d", r=group_size)  # r == s


def _all_reduce_simulator_op(
    X: torch.Tensor, rg_info: pea.collectives.ReplicaGroupingInfo
) -> torch.Tensor:
    out = einops.reduce(X, "r g n d -> r n d", "sum")
    return einops.repeat(out, "r n d -> r g n d", g=rg_info.group_size)


def _all_to_all_simulator_op(
    X: torch.Tensor, rg_info: ReplicaGroupingInfo
) -> torch.Tensor:
    return einops.rearrange(X, "r g s d -> r s g d", g=rg_info.group_size)  # g == s


_op_mapping = {
    pea.collectives.all_gather_cross_replica: _all_gather_simulator_op,
    pea.collectives.all_reduce_cross_replica_sum: _all_reduce_simulator_op,
    pea.collectives.all_to_all_single_cross_replica: _all_to_all_simulator_op,
}


def generate_input() -> torch.Tensor:
    X = einops.rearrange(torch.arange(32, dtype=torch.float32), "(d n)  -> n d", n=4)
    return X


def simulate_collective(
    X: torch.Tensor,
    op: Callable[[torch.Tensor, ReplicaGroupingInfo], torch.Tensor],
    rg_info: ReplicaGroupingInfo,
) -> Tuple[Any, Any, Any]:
    sim = _CollectiveSimulator(deepcopy(X), rg_info=rg_info, simulator_op=op)
    lr = 1.0
    if op == _all_reduce_simulator_op:
        lr /= rg_info.num_groups
    if op in set([_all_gather_simulator_op, _all_to_all_simulator_op]):
        lr /= rg_info.group_size
    optimizer = torch.optim.SGD(sim.parameters(), lr=lr)
    optimizer.zero_grad()
    out, loss = sim()
    loss.mean().backward()
    grad = einops.rearrange(sim.Xr.grad, "r g n d -> (r g n) d")
    optimizer.step()
    if op == _all_reduce_simulator_op:
        grad *= rg_info.group_size  # type: ignore
    if op in set([_all_gather_simulator_op, _all_to_all_simulator_op]):
        grad *= rg_info.num_groups  # type: ignore
    return out, grad, sim.X.data


def run_collective(
    X: torch.Tensor,
    op: Callable[[torch.Tensor, ReplicaGroupingInfo], torch.Tensor],
    rg_info: ReplicaGroupingInfo,
) -> Tuple[Any, Any, Any]:
    options = poptorch.Options()
    options.replicationFactor(rg_info.num_replicas)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options.anchorTensor("grad_X", "Gradient___X")
    options._Popart.setPatterns({"OpToIdentity": True})

    col = _CollectiveCrossReplicaTester(X, rg_info=rg_info, cross_replica_op=op)
    optimizer = poptorch.optim.SGD(col.parameters(), lr=1.0)
    col = poptorch.trainingModel(col, options, optimizer)
    utils._apply_replica_grouping(col, CommGroupType.Orthogonal, rg_info.num_groups)
    out, _ = col()
    out = out.detach().cpu()
    grad = col.getAnchoredTensor("grad_X")  # type: ignore
    return out, grad, col.X.data


@pytest.mark.parametrize("op", list(_op_mapping.keys()))
def test_collective(
    op: Callable[[torch.Tensor, ReplicaGroupingInfo], torch.Tensor]
) -> None:
    if op == pea.collectives.all_to_all_single_cross_replica:
        num_ipus = 2
    else:
        num_ipus = 8
    group_size = 2
    X = generate_input()
    rg_info = ReplicaGroupingInfo(num_ipus, 1, group_size)
    actual = run_collective(X, op, rg_info)
    expected = simulate_collective(X, _op_mapping[op], rg_info)
    list(map(assert_close, actual, expected))
