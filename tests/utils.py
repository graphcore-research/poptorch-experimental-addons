# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from typing import List

import torch.nn as nn
from poptorch.enums import CommGroupType, VariableRetrievalMode


def _apply_replica_grouping(
    model: nn.Module,
    comm_group_type: CommGroupType,
    shards: int,
    excluded_parameters: List[str] = [],
) -> nn.Module:
    for n, _ in model.named_parameters():
        if n not in excluded_parameters:
            model.per_replica_params[n] = (  # type: ignore
                comm_group_type,
                shards,
                VariableRetrievalMode.OnePerGroup,
            )
    return model
