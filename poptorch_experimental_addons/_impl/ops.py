# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Standalone utilities that don't belong in larger groups.
"""

from typing import Any, Tuple
import poptorch
import torch
from torch import Tensor


def broadcasted_pairwise_distance(tensor1: Tensor, tensor2: Tensor, p: int) -> Tensor:
    """
    p-norm broadcasted pairwise distance between two collections of vectors.

    Computes p-norm reduction along trailing dimension of tensor1[:,None,:] - tensor2[None,:,:]
    without materializing the intermediate broadcasted difference, for memory optimization.

    tensor1 -- shape (M, K)
    tensor2 -- shape (N, K)
    returns --  shape (M, N)
    """
    if tensor1.dim() != 2 or tensor2.dim() != 2:
        raise ValueError(
            "broadcasted_pairwise_distance requires 2-dimensional inputs"
            f"`tensor1` (dim = {tensor1.dim()}) and `tensor2` (dim = {tensor2.dim()})"
        )

    if tensor1.shape[-1] != tensor2.shape[-1]:
        raise ValueError(
            "broadcasted_pairwise_distance requires rightmost dimension of same size"
            f"for `tensor1` ({tensor1.shape[-1]}) and `tensor2` ({tensor2.shape[-1]})"
        )

    if poptorch.isRunningOnIpu():
        if p not in [1, 2]:
            raise NotImplementedError(
                "broadcasted_pairwise_distance implemented only for p=1,2 on IPU"
            )

        out = poptorch.custom_op(
            name=f"L{p}Distance",
            domain_version=1,
            domain="ai.graphcore.pea",
            inputs=[tensor1, tensor2],
            example_outputs=[
                torch.zeros(
                    dtype=tensor1.dtype, size=[tensor1.shape[0], tensor2.shape[0]]
                )
            ],
        )[0]
    else:
        out = torch.cdist(tensor1, tensor2, p=p)

    return out


__all__ = ["broadcasted_pairwise_distance"]
