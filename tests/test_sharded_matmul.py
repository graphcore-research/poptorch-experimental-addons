# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from copy import deepcopy
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Tuple

import einops
import poptorch
import pytest
import torch
import torch.nn as nn
from poptorch.enums import CommGroupType

import poptorch_experimental_addons as pea

from . import utils

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
    out = einops.repeat(out, "n m -> r n m", r=replication_factor)
    return out


def _repcolcol_simulator(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int
) -> torch.Tensor:
    out = einops.einsum(X, Y, "n d, r d m -> r n m")
    return out


_sharding_transform_map: Dict[Any, Any] = {
    Sharding.Replicated: {
        "in": lambda x, r: x,
        "out_sim": lambda x: x,
        "out_test": lambda x, r: x,
    },
    Sharding.Row: {
        "in": partial(einops.rearrange, pattern="(r n) d -> r n d"),
        "out_sim": partial(einops.rearrange, pattern="r n d -> (r n) d"),
        "out_test": lambda x, r: x,
    },
    Sharding.Column: {
        "in": partial(einops.rearrange, pattern="d (m r) -> r d m"),
        "out_sim": partial(einops.rearrange, pattern="r d m -> d (m r)"),
        "out_test": partial(einops.rearrange, pattern="(r d) m -> d (m r)"),
    },
}


_op_mapping: Dict[Any, Any] = {
    pea.sharded.rowcolrow_sharded_matmul: {
        "sharding": (Sharding.Row, Sharding.Column, Sharding.Row),
        "simulator": _rowcolrow_simulator,
        "kwargs": {"num_chunks": 2},
    },
    pea.sharded.repcolcol_sharded_matmul: {
        "sharding": (Sharding.Replicated, Sharding.Column, Sharding.Column),
        "simulator": _repcolcol_simulator,
        "kwargs": {},
    },
    pea.sharded.colrowrep_sharded_matmul: {
        "sharding": (Sharding.Column, Sharding.Row, Sharding.Replicated),
        "simulator": _colrowrep_simulator,
        "kwargs": {},
    },
}


class _ShardedMatmulTester(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        replication_factor: int,
        sharded_op: Callable[[Any], torch.Tensor],
    ):
        super().__init__()
        self.X_sharding, self.Y_sharding, self.out_sharding = _op_mapping[sharded_op][
            "sharding"
        ]

        X_trf = _sharding_transform_map[self.X_sharding]["in"]
        self.X = nn.Parameter(X_trf(X, r=replication_factor).contiguous())

        Y_trf = _sharding_transform_map[self.Y_sharding]["in"]
        self.Y = nn.Parameter(Y_trf(Y, r=replication_factor).contiguous())
        self.replication_factor = replication_factor
        self.op_kwargs = _op_mapping[sharded_op]["kwargs"]
        self.sharded_op = sharded_op

    def forward(self) -> Tuple[Any, Any]:
        out = self.sharded_op(
            self.X, self.Y, self.replication_factor, **self.op_kwargs
        )  # type: ignore
        return out, poptorch.identity_loss(out**2, reduction="mean")


class _ShardedMatmulSimulator(torch.nn.Module):
    def __init__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        replication_factor: int,
        sharded_op: Callable[[Any], torch.Tensor],
    ):
        super().__init__()
        self.X_sharding, self.Y_sharding, self.out_sharding = _op_mapping[sharded_op][
            "sharding"
        ]
        simulator_op = _op_mapping[sharded_op]["simulator"]

        X_trf = _sharding_transform_map[self.X_sharding]["in"]
        self.X = nn.Parameter(X_trf(X, r=replication_factor).contiguous())

        Y_trf = _sharding_transform_map[self.Y_sharding]["in"]
        self.Y = nn.Parameter(Y_trf(Y, r=replication_factor).contiguous())
        self.replication_factor = replication_factor
        self.op_kwargs = _op_mapping[sharded_op]["kwargs"]
        self.simulator_op = simulator_op

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
    X: torch.Tensor, Y: torch.Tensor, op: Callable[[Any], torch.Tensor], num_ipus: int
) -> Tuple[Any, Any, Any, Any, Any]:
    simulator = _ShardedMatmulSimulator(
        deepcopy(X),
        deepcopy(Y),
        replication_factor=num_ipus,
        sharded_op=op,
    )
    lr = 1.0
    if simulator.out_sharding == Sharding.Replicated:
        lr /= num_ipus
        optimizer = torch.optim.SGD(simulator.parameters(), lr=lr)
    else:
        params = [
            {"params": simulator.X, "lr": lr},
            {"params": simulator.Y, "lr": lr},
        ]
        if simulator.X_sharding == Sharding.Replicated:
            params[0]["lr"] *= num_ipus  # type: ignore
        if (
            simulator.X_sharding == Sharding.Row
            and simulator.Y_sharding == Sharding.Column
        ):
            params[0]["lr"] /= num_ipus  # type: ignore
        optimizer = torch.optim.SGD(params, lr=lr)
    optimizer.zero_grad()
    out, loss = simulator()
    loss.mean().backward()
    grad_X = _sharding_transform_map[simulator.X_sharding]["out_sim"](simulator.X.grad)
    grad_Y = _sharding_transform_map[simulator.Y_sharding]["out_sim"](simulator.Y.grad)
    if simulator.X_sharding == Sharding.Replicated:
        grad_X = einops.repeat(grad_X, "n d -> (r n) d", r=num_ipus)
    if simulator.out_sharding != Sharding.Replicated:
        grad_X *= num_ipus
        grad_Y *= num_ipus
    optimizer.step()
    return out, grad_X, grad_Y, simulator.X.data, simulator.Y.data


def run_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, op: Callable[[Any], torch.Tensor], num_ipus: int
) -> Tuple[Any, Any, Any, Any, Any]:
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
    optimizer = poptorch.optim.SGD(tester.parameters(), lr=1.0)
    tester = poptorch.trainingModel(tester, options, optimizer)
    replicated_params = ["X"] if tester.X_sharding == Sharding.Replicated else []
    utils._apply_replica_grouping(
        tester, CommGroupType.Orthogonal, 1, excluded_parameters=replicated_params
    )
    out, _ = tester()
    out = out.detach().cpu()
    grad_X = tester.getAnchoredTensor("grad_X")  # type: ignore
    grad_X = _sharding_transform_map[tester.X_sharding]["out_test"](grad_X, r=num_ipus)
    grad_Y = tester.getAnchoredTensor("grad_Y")  # type: ignore
    grad_Y = _sharding_transform_map[tester.Y_sharding]["out_test"](grad_Y, r=num_ipus)
    return out, grad_X, grad_Y, tester.X.data, tester.Y.data


@pytest.mark.parametrize("op", list(_op_mapping.keys()))
def test_sharded_matmul(op: Callable[[Any], torch.Tensor]) -> None:
    X, Y = generate_inputs()
    out = X @ Y
    num_ipus = 2
    actual = run_sharded_matmul(X, Y, op, num_ipus)
    expected = simulate_sharded_matmul(X, Y, op, num_ipus)
    assert_close(out, expected[0])
    assert_close(out, actual[0])
    list(map(assert_close, actual, expected))


if __name__ == "__main__":
    test_sharded_matmul(pea.sharded.rowcolrow_sharded_matmul)
