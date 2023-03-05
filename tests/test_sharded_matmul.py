# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import random
from copy import deepcopy
from typing import Any, Callable, Tuple
from enum import Enum
from functools import partial

import einops
import numpy as np
import poptorch
import pytest
import torch
import torch.nn as nn
from poptorch.enums import CommGroupType

import poptorch_experimental_addons as pea

import utils

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


class Sharding(Enum):
    Replicated = 1
    Row = 2
    Column = 3


def _rowcolrow_simulator(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int, num_chunks: int
) -> torch.Tensor:
    _, X_local_outer_dim, _ = X.shape
    _, _, Y_local_outer_dim = Y.shape
    X = einops.rearrange(X, "r m (k c) -> r m k c", k=num_chunks)
    Y = einops.rearrange(Y, "r (k c) n -> r c k n", k=num_chunks)

    out = torch.zeros(
        (
            replication_factor,
            X_local_outer_dim,
            Y_local_outer_dim * replication_factor,
        )
    )
    for i in range(num_chunks):
        Yg = Y[:, :, i]
        Yg = einops.repeat(Yg, "s c n -> r s c n", r=replication_factor)
        Yg = einops.rearrange(Yg, "r s c n -> r c (n s)")
        Xp = X[:, :, i]
        out += Xp @ Yg
    return out


def _colrowrep_simulator(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int
) -> torch.Tensor:
    out = einops.einsum(X, Y, "r n d, r d m -> r n m")
    out = einops.reduce(out, "r n m -> n m", "sum")
    return out


def _repcolcol_simulator(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int
) -> torch.Tensor:
    out = einops.einsum(X, Y, "n d, r d m - > r n m")
    return out


_sharding_transform_map = {
    Sharding.Replicated: lambda x: x,
    Sharding.Row: partial(einops.rearrange, pattern="(r n) d -> r n d"),
    Sharding.Column: partial(einops.rearrange(pattern="d (m r) -> r d m")),
}


_op_mapping = {
    pea.sharded.rowcolrow_sharded_matmul: {
        "sharding": (Sharding.Row, Sharding.Column),
        "simulator": _rowcolrow_simulator,
        "kwargs": {"num_chunks": 2},
    },
    pea.sharded.repcolcol_sharded_matmul: {
        "sharding": (Sharding.Replicated, Sharding.Column),
        "simulator": _colrowrep_simulator,
        "kwargs": {},
    },
    pea.sharded.colrowrep_sharded_matmul: {
        "sharding": (Sharding.Column, Sharding.Row),
        "simulator": _repcolcol_simulator,
        "kwargs": {},
    },
}


class _ShardedMatmulTester(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        replication_factor: int,
        sharded_op: Callable,
    ):
        super().__init__()
        X_sharding, Y_sharding = _op_mapping[sharded_op]["sharding"]

        X_trf = _sharding_transform_map[X_sharding]
        self.X = nn.Parameter(X_trf(X, r=replication_factor).contiguous())

        Y_trf = _sharding_transform_map[Y_sharding]
        self.X = nn.Parameter(Y_trf(Y, r=replication_factor).contiguous())
        self.replication_factor = replication_factor
        self.op_kwargs = _op_mapping[sharded_op]["kwargs"]
        self.sharded_op = sharded_op

    def forward(self) -> Tuple[Any, Any]:
        out = self.sharded_op(self.X, self.Y, self.replication_factor, **self.op_kwargs)
        return out, poptorch.identity_loss(out**2, reduction="mean")


class _ShardedMatmulSimulator(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        replication_factor: int,
        sharded_op: Callable,
    ):
        super().__init__()
        X_sharding, Y_sharding = _op_mapping[sharded_op]["sharding"]
        simulator_op = _op_mapping[sharded_op]["simulator"]

        X_trf = _sharding_transform_map[X_sharding]
        self.X = nn.Parameter(X_trf(X, r=replication_factor).contiguous())

        Y_trf = _sharding_transform_map[Y_sharding]
        self.X = nn.Parameter(Y_trf(Y, r=replication_factor).contiguous())
        self.replication_factor = replication_factor
        self.op_kwargs = _op_mapping[sharded_op]["kwargs"]
        self.simulator = simulator_op

    def forward(self) -> Tuple[Any, Any]:
        out = self.simulator_op(
            self.X, self.Y, self.replication_factor, **self.op_kwargs
        )
        loss = out.pow(2).mean(dim=list(range(out.ndim))[1:])
        return torch.vstack([*out]), loss


def generate_inputs() -> Tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(112358)
    X = torch.randn(6, 4, dtype=torch.float32)
    Y = torch.randn(4, 8, dtype=torch.float32)
    return X, Y


def simulate_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, op: Callable, num_ipus: int
) -> Tuple[Any, Any, Any]:
    simulator = _ShardedMatmulSimulator(
        deepcopy(X),
        deepcopy(Y),
        replication_factor=num_ipus,
        num_chunks=2,
        matmul_op=op,
    )
    out, loss = simulator()
    loss.mean().backward()
    grad_X = einops.rearrange(simulator.X.grad, "r n d -> (r n) d")
    grad_Y = einops.rearrange(simulator.Y.grad, "r d m -> d (m r)")
    return out, grad_X, grad_Y


def run_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, op: Callable, num_ipus: int
) -> Tuple[Any, Any, Any]:
    options = poptorch.Options()
    options.replicationFactor(num_ipus)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options.anchorTensor("grad_X", "Gradient___X")
    options.anchorTensor("grad_Y", "Gradient___Y")
    options._Popart.setPatterns({"OpToIdentity": True})
    tester = _ShardedMatmulTester(
        deepcopy(X), deepcopy(Y), replication_factor=num_ipus, sharded_op=op
    )
    optimizer = poptorch.optim.SGD(tester.parameters(), lr=0.0)
    tester = poptorch.trainingModel(tester, options, optimizer)
    utils._apply_replica_grouping(tester, CommGroupType.Orthogonal, 1)
    out, _ = tester()
    out = out.detach().cpu()
    grad_X = tester.getAnchoredTensor("grad_X")  # type: ignore
    grad_Y = tester.getAnchoredTensor("grad_Y")  # type: ignore
    grad_Y = einops.rearrange(grad_Y, "(r d) m -> d (m r)", r=num_ipus)
    return out, grad_X, grad_Y


@pytest.mark.parametrize("op", list(_op_mapping.keys()))
def test_sharded_matmul(op: Callable) -> None:
    X, Y = generate_inputs()
    num_ipus = 2
    actual = run_sharded_matmul(X, Y, op, num_ipus)
    expected = simulate_sharded_matmul(X, Y, op, num_ipus)
    map(assert_close, actual, expected)


if __name__ == "__main__":
    test_sharded_matmul()
