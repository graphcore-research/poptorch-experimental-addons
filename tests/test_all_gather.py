# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import torch
import poptorch
import poptorch_experimental_addons as pea
import random
import numpy as np
import einops


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


class _ReplicatedAllGather(torch.nn.Module):
    def __init__(self, replication_factor):
        super().__init__()
        self.replication_factor = replication_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return pea.collectives.replicated_all_gather(x, self.replication_factor)


def test_all_gather():
    set_seed(112358)
    x = torch.randn(32, 24)
    options = poptorch.Options()
    num_ipus = 16
    options.replicationFactor(num_ipus)
    model = _ReplicatedAllGather(num_ipus)
    model = poptorch.inferenceModel(model, options)
    y = model(x)
    breakpoint()
    y = einops.rearrange(y, "(k n) d -> k n d", k=num_ipus)
    torch.testing.assert_allclose(y[0], x, rtol=0, atol=0)
    torch.testing.assert_allclose(y[7], x, rtol=0, atol=0)
    torch.testing.assert_allclose(y[-1], x, rtol=0, atol=0)


if __name__ == "__main__":
    test_all_gather()
