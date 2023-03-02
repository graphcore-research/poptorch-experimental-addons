# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import torch.nn as nn
import poptorch
from poptorch.enums import CommGroupType, VariableRetrievalMode
import poptorch_experimental_addons as pea
import random
import numpy as np
import einops
from copy import deepcopy
from typing import Tuple, Any

assert_close = torch.testing.assert_close  # type:ignore[attr-defined]


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class _AllGatherCrossReplicaTester(torch.nn.Module):
    def __init__(self, X: torch.Tensor, replication_factor: int):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor

    def forward(self) -> Tuple[Any, Any]:
        out = pea.collectives.all_gather_cross_replica(self.X, self.replication_factor)
        return out, poptorch.identity_loss(out, reduction="sum")


def _apply_replica_grouping(
    model: nn.Module, comm_group_type: CommGroupType, shards: int
) -> nn.Module:
    for n, _ in model.named_parameters():
        model.per_replica_params[n] = (  # type: ignore
            comm_group_type,
            shards,
            VariableRetrievalMode.OnePerGroup,
        )  # type: ignore
    return model


def test_all_gather() -> None:
    set_seed(112358)
    num_ipus = 2

    X = einops.rearrange(torch.arange(32, dtype=torch.float32), "(d n)  -> n d", n=4)

    options = poptorch.Options()
    options.replicationFactor(num_ipus)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options.anchorTensor("grad_X", "Gradient___X")
    options._Popart.setPatterns({"OpToIdentity": True})

    model = _AllGatherCrossReplicaTester(deepcopy(X), num_ipus)
    optimizer = poptorch.optim.SGD(model.parameters(), lr=1.0)
    model = poptorch.trainingModel(model, options, optimizer)
    _apply_replica_grouping(model, CommGroupType.Orthogonal, 1)

    Y, _ = model()
    Y = einops.rearrange(Y, "(r s) n d -> r (s n) d", r=num_ipus, s=num_ipus)
    Y = Y.detach().cpu()

    assert_close(Y[0], X, rtol=0, atol=0)
    assert_close(Y[1], X, rtol=0, atol=0)

    grad_X = model.getAnchoredTensor("grad_X")  # type: ignore
    assert_close(grad_X, torch.ones_like(X), rtol=0, atol=0)


if __name__ == "__main__":
    test_all_gather()
