# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
Demo of using `pea.autograd_proxy` to train a binary unit using a
straight-through estimator.
"""

import argparse
import sys
from typing import Tuple

import poptorch
import torch
from torch import Tensor, nn

import poptorch_experimental_addons as pea

ST_PROXIES = dict(
    none=None,
    tanh=torch.tanh,
    hardtanh=nn.functional.hardtanh,
    softsign=nn.functional.softsign,
    linear=lambda x: x,
)


def binary_quantise(x: Tensor, st_estimator: str) -> Tensor:
    q = (0 < x).to(x.dtype) * 2 - 1
    proxy = ST_PROXIES[st_estimator]
    if proxy is None:
        return q
    return pea.autograd_proxy(q, proxy(x))


class Model(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, st_estimator: str) -> None:
        super().__init__()
        self.encoder = nn.Linear(input_size, hidden_size)
        self.decoder = nn.Linear(hidden_size, input_size)
        self.st_estimator = st_estimator

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        z = self.encoder(x)
        z = binary_quantise(z, st_estimator=self.st_estimator)
        y = self.decoder(z)
        return y, nn.functional.mse_loss(x, y)


def run(
    batch_size: int,
    latent_size: int,
    latent_size_multiple: float,
    data_size: int,
    st_estimator: str,
) -> None:
    # Generate artificial data using binary latent features
    data_z = ((0.5 < torch.rand(batch_size, latent_size)) * 2 - 1).to(torch.float)
    data_x = data_z @ torch.randn(latent_size, data_size)

    # Train a model (full batch mode) using the straight-through estimator
    model = Model(
        data_x.shape[1],
        int(latent_size_multiple * latent_size),
        st_estimator=st_estimator,
    )
    opt = torch.optim.Adam(model.parameters(), 0.1)
    options = poptorch.Options()
    options.useIpuModel(True)
    train_step = poptorch.trainingModel(model, options, opt)
    for n in range(100):
        _, loss = train_step(data_x)
        print(f"#{n:>03d}: {float(loss):.2f}", file=sys.stderr)
    train_step.destroy()

    # Run inference, where the straight-through estimator is unused
    inference_model = poptorch.inferenceModel(model, options)
    print("Inference:", float(inference_model(data_x)[1]), file=sys.stderr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", default=1000)
    parser.add_argument("--latent-size", default=16)
    parser.add_argument(
        "--latent-size-multiple",
        default=2.0,
        help="gives the model spare capacity versus the data, for easier training",
    )
    parser.add_argument("--data-size", default=128)
    parser.add_argument("--st-estimator", default="tanh", choices=ST_PROXIES.keys())
    run(**vars(parser.parse_args()))
