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

import utils

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


def rowcolrow_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int, num_chunks: int = 1
) -> torch.Tensor:
    """
    Matrix multiplication for sharded input and output tensors

    Gathers the right multiplicand across IPU program replicas

    Input tensors - X (row-sharded), Y (col-sharded)
    Output tensor - (row-sharded)
    """
    X_local_outer_dim, _ = X.shape
    _, Y_local_outer_dim = Y.shape
    result = torch.zeros((X_local_outer_dim, Y_local_outer_dim * replication_factor))
    X = einops.rearrange(X, "m (k c) -> m k c", k=num_chunks)
    Y = einops.rearrange(Y, "(k c) n -> c k n", k=num_chunks)
    for i in range(num_chunks):
        index = torch.tensor([i])
        Yg = pea.collectives.all_gather_cross_replica(
            torch.index_select(Y, dim=1, index=index).squeeze(), replication_factor
        )
        Yg = einops.rearrange(Yg, "r c n -> c (n r)")
        Xp = torch.index_select(X, dim=1, index=index).squeeze()
        result += Xp @ Yg
    return result


class _ShardedMatmulTester(torch.nn.Module):
    def __init__(self, X, Y, replication_factor, num_chunks):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.Y = nn.Parameter(
            einops.rearrange(Y, "d (m r) -> r d m", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor
        self.num_chunks = num_chunks

    def forward(self):
        out = rowcolrow_sharded_matmul(
            self.X, self.Y, self.replication_factor, self.num_chunks
        )
        return out, poptorch.identity_loss(out**2, reduction="mean")


class _ShardedMatmulSimulator(torch.nn.Module):
    def __init__(self, X, Y, replication_factor, num_chunks):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.Y = nn.Parameter(
            einops.rearrange(Y, "d (m r) -> r d m", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor
        self.num_chunks = num_chunks

    def forward(self):
        _, X_local_outer_dim, _ = self.X.shape
        _, _, Y_local_outer_dim = self.Y.shape
        X = einops.rearrange(self.X, "r m (k c) -> r m k c", k=self.num_chunks)
        Y = einops.rearrange(self.Y, "r (k c) n -> r c k n", k=self.num_chunks)

        out = torch.zeros(
            (
                self.replication_factor,
                X_local_outer_dim,
                Y_local_outer_dim * self.replication_factor,
            )
        )
        for i in range(self.num_chunks):
            Yg = Y[:, :, i]
            Yg = einops.repeat(Yg, "s c n -> r s c n", r=self.replication_factor)
            Yg = einops.rearrange(Yg, "r s c n -> r c (n s)")
            Xp = X[:, :, i]
            out += Xp @ Yg
        loss = out.pow(2).mean(dim=list(range(out.ndim))[1:])
        return torch.vstack([*out]), loss


def test_sharded_matmul():
    torch.manual_seed(112358)
    X = torch.randn(6, 4, dtype=torch.float32)
    Y = torch.randn(4, 8, dtype=torch.float32)
    num_ipus = 2

    simulator = _ShardedMatmulSimulator(
        deepcopy(X), deepcopy(Y), replication_factor=num_ipus, num_chunks=2
    )
    out_sim, loss_sim = simulator()
    loss_sim.mean().backward()
    grad_X_sim = einops.rearrange(simulator.X.grad, "r n d -> (r n) d")
    grad_Y_sim = einops.rearrange(simulator.Y.grad, "r d m -> d (m r)")

    options = poptorch.Options()
    options.replicationFactor(num_ipus)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options.anchorTensor("grad_X", "Gradient___X")
    options.anchorTensor("grad_Y", "Gradient___Y")
    options._Popart.setPatterns({"OpToIdentity": True})

    tester = _ShardedMatmulTester(
        deepcopy(X), deepcopy(Y), replication_factor=num_ipus, num_chunks=2
    )
    optimizer = poptorch.optim.SGD(tester.parameters(), lr=0.0)
    tester = poptorch.trainingModel(tester, options, optimizer)
    utils._apply_replica_grouping(tester, CommGroupType.Orthogonal, 1)
    out_actual, loss_actual = tester()
    out_actual = out_actual.detach().cpu()
    grad_X_actual = tester.getAnchoredTensor("grad_X")  # type: ignore
    grad_Y_actual = tester.getAnchoredTensor("grad_Y")  # type: ignore
    grad_Y_actual = einops.rearrange(grad_Y_actual, "(r d) m -> d (m r)", r=num_ipus)

    assert_close(X @ Y, out_sim)
    assert_close(out_sim, out_actual)
    assert_close(grad_X_sim * num_ipus, grad_X_actual)  # why?
    assert_close(grad_Y_sim * num_ipus, grad_Y_actual)  # why?


if __name__ == "__main__":
    test_sharded_matmul()
