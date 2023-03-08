# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Primitives for collective communication across IPU clusters.
"""

from typing import Any

import poptorch
import torch


def _no_op_reshape(x: torch.Tensor) -> torch.Tensor:
    "A no op that forces reshapes to be inserted into the gradient graph"
    return x.unsqueeze(-1).squeeze(-1)


def all_gather_cross_replica_identical_grads_in(
    x: torch.Tensor, replication_factor: int
) -> Any:
    """
    All-gather across IPU program replicas.

    Gathers and stacks tensors occupying the same memory location across all IPUs

    Gradient graph generated assumes gradient inputs are identical

    x -- shape (*)
    returns --  shape (replication_factor, *)
    """
    x = _no_op_reshape(x)  # ensures grad of ReplicatedAllGather is reshaped
    out = poptorch.custom_op(
        [x],
        name="ReplicatedAllGather",
        domain="ai.graphcore",
        domain_version=1,
        example_outputs=[
            torch.zeros(dtype=x.dtype, size=(replication_factor, *x.shape))
        ],
    )[0]
    out = out.reshape(replication_factor, *x.shape)
    return out


def all_gather_cross_replica(x: torch.Tensor, replication_factor: int):
    """
    All-gather across IPU program replicas.

    Gathers and stacks tensors occupying the same memory location across all IPUs

    x -- shape (*)
    returns --  shape (replication_factor, *)
    """
    x = all_gather_cross_replica_identical_grads_in(x, replication_factor)
    x = all_reduce_cross_replica_sum(x, replication_factor, True)
    return x


def all_reduce_cross_replica_sum(
    x: torch.Tensor, replication_factor: int, insert_in_grad_graph: bool = False
) -> Any:
    """
    All-reduce across IPU program replicas

    Sums tensors occupying the same memory location across IPUs, resulting
    in replicated tensors.

    insert_in_grad_graph is a boolean argument that inserts the all_reduce in
    the gradient graph (backward pass) rather than the forward graph.

    x -- shape (*)
    returns -- shape (*)
    """
    rg_info = [replication_factor, 1, replication_factor]
    out = poptorch.custom_op(
        [x],
        name="ReplicatedAllReduceTP",
        domain="ai.graphcore",
        domain_version=1,
        example_outputs=[x],
        attributes={
            "op": "sum",
            "__collectiveReplicaGrouping": rg_info,
            "backwards": insert_in_grad_graph,
        },
    )[0]
    return out


__all__ = [
    "all_gather_cross_replica_identical_grads_in",
    "all_gather_cross_replica",
    "all_reduce_cross_replica_sum",
]
