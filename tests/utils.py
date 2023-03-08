# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch.nn as nn
from poptorch.enums import CommGroupType, VariableRetrievalMode


def _apply_replica_grouping(
    model: nn.Module, comm_group_type: CommGroupType, shards: int
) -> nn.Module:
    for n, _ in model.named_parameters():
        model.per_replica_params[n] = (  # type: ignore
            comm_group_type,
            shards,
            VariableRetrievalMode.OnePerGroup,
        )
    return model
