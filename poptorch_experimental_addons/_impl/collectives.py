# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import poptorch


def _no_op_reshape(x: torch.Tensor) -> torch.Tensor:
    return x.unsqueeze(-1).squeeze(-1)


def all_gather_cross_replica(x: torch.Tensor, replication_factor: int) -> torch.Tensor:
    """
    All-gather across IPU program replicas.

    Gathers and stacks tensors occupying the same memory location across IPUs,
    then replicates the result.

    x -- shape (*)
    returns --  shape (replication_factor, *)
    """
    x = _no_op_reshape(x)
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


__all__ = ["all_gather_cross_replica"]
