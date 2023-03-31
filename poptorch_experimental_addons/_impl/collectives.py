# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Primitives for collective communication across IPU clusters.
"""

from dataclasses import dataclass
from typing import Any

import poptorch
import torch


@dataclass
class ReplicaGroupingInfo:
    num_replicas: int
    stride: int
    group_size: int

    @property
    def attributes(self):
        return [self.num_replicas, self.stride, self.group_size]

    @property
    def num_groups(self):
        return self.num_replicas // self.group_size


def _no_op_reshape(x: torch.Tensor) -> torch.Tensor:
    "A no op that forces reshapes to be inserted into the gradient graph"
    return x.unsqueeze(-1).squeeze(-1)


def all_gather_cross_replica_identical_grads_in(
    x: torch.Tensor, rg_info: ReplicaGroupingInfo
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
            torch.zeros(dtype=x.dtype, size=(rg_info.group_size, *x.shape))
        ],
        attributes={
            "__collectiveReplicaGrouping": rg_info.attributes,
        },
    )[0]
    out = out.reshape(rg_info.group_size, *x.shape)
    return out


def all_gather_cross_replica(x: torch.Tensor, rg_info: ReplicaGroupingInfo) -> Any:
    """
    All-gather across IPU program replicas.

    Gathers and stacks tensors occupying the same memory location across all IPUs

    x -- shape (*)
    returns --  shape (replication_factor, *)
    """
    x = all_gather_cross_replica_identical_grads_in(x, rg_info)
    x = all_reduce_cross_replica_sum(x, rg_info, insert_in_grad_graph=True)
    return x


def all_reduce_cross_replica_sum(
    x: torch.Tensor,
    rg_info: ReplicaGroupingInfo,
    insert_in_grad_graph: bool = False,
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
    out = poptorch.custom_op(
        [x],
        name="ReplicatedAllReduceTP",
        domain="ai.graphcore",
        domain_version=1,
        example_outputs=[x],
        attributes={
            "op": "sum",
            "__collectiveReplicaGrouping": rg_info.attributes,
            "backwards": insert_in_grad_graph,
        },
    )[0]
    return out


def all_to_all_single_cross_replica(
    x: torch.Tensor, rg_info: ReplicaGroupingInfo
) -> Any:
    """
    All-to-all across IPU program replicas

    Splits input tensor over leading axis and scatters to IPU according to position.

    Leading axis must equal total number of replicas.

    Does not support uneven splits.

    Also see docs for `torch.distributed.all_to_all_single` for similar

    x -- shape (*)
    returns  -- shape (*)
    """
    if rg_info.num_groups > 1:
        raise NotImplementedError(
            "all_to_all_single_cross_replica currently only \
            supports communication for a single replica group"
        )
    out = poptorch.custom_op(
        [x],
        name="ReplicatedAllToAll",
        domain="ai.graphcore",
        domain_version=1,
        example_outputs=[x],
        attributes={
            "__collectiveReplicaGrouping": rg_info.attributes,
        },
    )[0]
    return out


__all__ = [
    "all_gather_cross_replica_identical_grads_in",
    "all_gather_cross_replica",
    "all_reduce_cross_replica_sum",
    "all_to_all_single_cross_replica",
    "ReplicaGroupingInfo",
]
