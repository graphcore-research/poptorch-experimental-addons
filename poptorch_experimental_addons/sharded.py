# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import Any

import einops
import torch

from .collectives import all_gather_cross_replica, all_reduce_cross_replica_sum


def rowcolrow_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int, num_chunks: int = 1
) -> Any:
    """
    Matrix multiplation for row-sharded x column-sharded -> row-sharded tensors

    Gathers the right multiplicand across IPU program replicas
    """
    X_local_outer_dim, _ = X.shape
    _, Y_local_outer_dim = Y.shape
    result = torch.zeros((X_local_outer_dim, Y_local_outer_dim * replication_factor))
    X = einops.rearrange(X, "m (k c) -> m k c", k=num_chunks)
    Y = einops.rearrange(Y, "(k c) n -> c k n", k=num_chunks)
    for i in range(num_chunks):
        index = torch.tensor([i])
        Yg = all_gather_cross_replica(
            torch.index_select(Y, dim=1, index=index).squeeze(), replication_factor
        )
        Yg = einops.rearrange(Yg, "r c n -> c (n r)")
        Xp = torch.index_select(X, dim=1, index=index).squeeze()
        result += Xp @ Yg
    return result


def repcolcol_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int
) -> Any:
    """Matrix multiplation for replicated x column-sharded -> column-sharded tensors"""
    X = all_reduce_cross_replica_sum(X, replication_factor, insert_in_grad_graph=True)
    return X @ Y


def colrowrep_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int
) -> Any:
    """Matrix multiplation for row-sharded x column-sharded -> replicated tensors"""
    out = X @ Y
    return all_reduce_cross_replica_sum(out, replication_factor)
