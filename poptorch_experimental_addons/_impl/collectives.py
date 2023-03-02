# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import poptorch


def replicated_all_gather(x: torch.Tensor, replication_factor: int) -> torch.Tensor:
    """
    All-gather across IPU replicas.

    Gathers and concatenates tensors occupying the same memory location across IPUs,
    then replicates the result.
    """
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


__all__ = ["replicated_all_gather"]
