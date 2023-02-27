# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""A basic FFN benchmarking harness for `pea.sparse.StaticSparseLinear`."""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import poptorch
import torch as T

import poptorch_experimental_addons as pea


class _Benchmark(T.nn.Module):
    """A pruned FFN, implemented with sparse or dense linear layers."""

    def __init__(
        self,
        weight_up: T.Tensor,
        weight_down: T.Tensor,
        dtype: T.dtype,
        density: float,
        block_size: int,
        sparse: bool,
        loop_iterations: int,
    ):
        super().__init__()

        def pruned_linear_layer(dense_weight: T.Tensor) -> T.nn.Module:
            weight = pea.sparse.magnitude_prune(
                dense_weight.to(dtype), block_size=block_size, density=density
            )
            if sparse:
                return pea.sparse.StaticSparseLinear(weight)
            layer = T.nn.Linear(
                dense_weight.shape[1], dense_weight.shape[0], bias=False
            )
            layer.weight.data.copy_(pea.sparse.block_coo_to_dense(weight))
            return layer

        self.density = density
        self.dtype = dtype

        self.ffn = T.nn.Sequential(
            pruned_linear_layer(weight_up),
            T.nn.ReLU(),
            pruned_linear_layer(weight_down),
        )
        self.ffn_size, self.hidden_size = weight_up.shape
        self.loop_iterations = loop_iterations

    def flop_count(self, batch_size: int) -> int:
        return (
            self.loop_iterations
            * 2
            * batch_size
            * 2
            * int(self.density * self.hidden_size * self.ffn_size)
        )

    def random_input(self, batch_size: int) -> T.Tensor:
        return T.randn(batch_size, self.hidden_size, dtype=self.dtype)

    def forward(self, input: T.Tensor) -> T.Tensor:
        output = input
        if poptorch.isRunningOnIpu():
            (output,) = poptorch.for_loop(self.loop_iterations, self.ffn, [output])
            return output
        for _ in range(self.loop_iterations):
            output = self.ffn(output)
        return output


def run(
    opt_checkpoint: Optional[str],
    opt_checkpoint_layer: int,
    hidden_size: int,
    density: float,
    block_size: int,
    batch_size: int,
    dtype: T.dtype,
    method: str,
    iterations_per_run: int,
    runs: int,
    seed: Optional[int],
) -> None:
    if seed:
        T.manual_seed(seed)
    else:
        T.seed()

    if opt_checkpoint:
        weights = T.load(opt_checkpoint)
        # Handle a naming inconsistency with opt-350m.bin
        pre_prefix = "" if "decoder.layers.0.fc1.weight" in weights else "model."
        prefix = f"{pre_prefix}decoder.layers.{opt_checkpoint_layer}"
        weight_up = weights[f"{prefix}.fc1.weight"]
        weight_down = weights[f"{prefix}.fc2.weight"]
    else:
        weight_up = T.randn(4 * hidden_size, hidden_size) / np.sqrt(hidden_size)
        weight_down = T.randn(hidden_size, 4 * hidden_size) / np.sqrt(4 * hidden_size)

    settings = dict(
        opt_checkpoint_path=opt_checkpoint and str(opt_checkpoint),
        opt_checkpoint_layer=opt_checkpoint_layer,
        hidden_size=weight_up.shape[1],
        ffn_size=weight_up.shape[0],
        density=density,
        block_size=block_size,
        batch_size=batch_size,
        dtype=str(dtype),
        method=method,
        iterations_per_run=iterations_per_run,
        seed=seed,
    )
    print(f"Running {settings}", file=sys.stderr)

    benchmark = _Benchmark(
        weight_up,
        weight_down,
        density=density,
        block_size=block_size,
        loop_iterations=iterations_per_run,
        dtype=dtype,
        sparse=dict(sparse=True, dense=False)[method],
    )
    model = poptorch.inferenceModel(benchmark)
    benchmark_input = benchmark.random_input(batch_size)
    elapsed = []
    for iteration in range(runs):
        t0 = time.time()
        model(benchmark_input)
        elapsed.append(time.time() - t0)
        print(
            f"[{iteration:>03d}] Elapsed {elapsed[-1] * 1000:.1f} ms"
            f" ({benchmark.flop_count(batch_size) / elapsed[-1] / 1e12:.1f} TFLOP/s)",
            file=sys.stderr,
        )
    print(json.dumps(dict(**settings, time=elapsed)))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opt-checkpoint", type=Path, help="path to an OPT checkpoint")
    parser.add_argument(
        "--opt-checkpoint-layer",
        type=int,
        default=7,
        help="layer to run from the OPT checkpoint",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=1024,
        help="hidden size to use, ignored if --opt-checkpoint-path is set",
    )
    parser.add_argument(
        "--density",
        type=float,
        default=0.1,
        help="approximate density of nonzero blocks",
    )
    parser.add_argument(
        "--block-size",
        type=int,
        default=16,
        help="size of nonzero blocks (block_size x block_size)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8 * 128,
        help="batch size, the total number of tokens",
    )
    parser.add_argument(
        "--dtype", type=dict(float=T.float, half=T.half).__getitem__, default=T.half
    )
    parser.add_argument("--method", choices=["sparse", "dense"], default="sparse")
    parser.add_argument("--iterations-per-run", type=int, default=5000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int)
    run(**vars(parser.parse_args()))
