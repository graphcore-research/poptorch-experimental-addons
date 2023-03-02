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


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class _AllGatherCrossReplicaTester(torch.nn.Module):
    def __init__(self, X, replication_factor):
        super().__init__()
        self.X = nn.Parameter(
            einops.rearrange(X, "(r n) d -> r n d", r=replication_factor).contiguous()
        )
        self.replication_factor = replication_factor

    def forward(self):
        out = pea.collectives.all_gather_cross_replica(self.X, self.replication_factor)
        return out, poptorch.identity_loss(out, reduction="sum")


def _apply_replica_grouping(model, comm_group_type, shards):
    for n, _ in model.named_parameters():
        model.per_replica_params[n] = (
            comm_group_type,
            shards,
            VariableRetrievalMode.OnePerGroup,
        )
    return model


def test_all_gather():
    set_seed(112358)
    X = einops.rearrange(torch.arange(32, dtype=torch.float32), "(d n)  -> n d", n=4)
    options = poptorch.Options()
    num_ipus = 2
    options.replicationFactor(num_ipus)
    options.outputMode(poptorch.OutputMode.All)
    options.useIpuModel(True)
    options._Popart.setPatterns({"OpToIdentity": True})
    model = _AllGatherCrossReplicaTester(deepcopy(X), num_ipus)
    optimizer = poptorch.optim.SGD(model.parameters(), lr=1.0)
    model = poptorch.trainingModel(model, options, optimizer)
    _apply_replica_grouping(model, CommGroupType.Orthogonal, 1)
    Y, _ = model()
    Y = einops.rearrange(Y, "(r s) n d -> r (s n) d", r=num_ipus, s=num_ipus)
    Y = Y.detach().cpu()
    torch.testing.assert_close(Y[0], X, rtol=0, atol=0)
    torch.testing.assert_close(Y[1], X, rtol=0, atol=0)
