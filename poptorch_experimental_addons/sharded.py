# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import einops
from .collectives import (
    all_reduce_cross_replica_sum,
    all_gather_cross_replica_mean_grad,
)


def rowcolrow_sharded_matmul(
    X: torch.Tensor, Y: torch.Tensor, replication_factor: int, num_chunks: int = 1
) -> torch.Tensor:
    """
    Matrix multiplication for sharded input and output tensors

    Gathers the right multiplicand across IPU program replicas in chunks to compute
    then accumulate partial resulst.


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
        Yg = pea.collectives.all_gather_cross_replica_mean_grad(
            torch.index_select(Y, dim=1, index=index).squeeze(), replication_factor
        )
        Yg = einops.rearrange(Yg, "r c n -> c (n r)")
        Yg = pea.collectives.all_reduce_cross_replica_sum(
            Yg, replication_factor, insert_in_grad_graph=True
        )
        Xp = torch.index_select(X, dim=1, index=index).squeeze()
        result += Xp @ Yg
    return result
