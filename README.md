# PopTorch Experimental Addons

A collection of addons to [PopTorch](https://github.com/graphcore/poptorch), with general utility.


## Usage

See [documentation](https://graphcore-research.github.io/poptorch-experimental-addons).

```bash
# Tested on Poplar SDK 3.1.0+1205, Ubuntu 20.04, Python 3.8
pip install git+https://github.com/graphcore-research/poptorch-experimental-addons

# Run an example
wget https://raw.githubusercontent.com/graphcore-research/poptorch-experimental-addons/main/examples/sparse_benchmark_spmm.py
python sparse_benchmark_spmm.py
```

```python
# Import & use
import poptorch_experimental_addons as pea
```

### Libraries

| API | Description | Note |
| --- | --- | --- |
| [`pea.autograd_proxy`](https://graphcore-research.github.io/poptorch-experimental-addons/index.html#poptorch_experimental_addons.autograd_proxy) | Custom gradients using a proxy tensor | |
| [`pea.distance_matrix`](https://graphcore-research.github.io/poptorch-experimental-addons/index.html#poptorch_experimental_addons.distance_matrix) | Pairwise broadcasted distance between two collections of vectors | Supports only L1/L2 distances on IPU |
| [`pea.collectives`](https://graphcore-research.github.io/poptorch-experimental-addons/collectives.html) | Collectives across IPU program replicas | All-gather |
| [`pea.sparse`](https://graphcore-research.github.io/poptorch-experimental-addons/sparse.html) | Static sparse-dense matmul (forward pass only) | Functions and modules for weight-sparse inference |


## Development

First-time setup:

```bash
python3 -m venv .venv
# Add to .venv/bin/activate
# source /PATH_TO_POPLAR_SDK/enable
source .venv/bin/activate
pip install wheel
pip install $POPLAR_SDK_ENABLED/../poptorch-*.whl
pip install -r requirements-dev.txt
```

We recommend symlinking some system libraries to `third_party`, to get the IDE helping:

```bash
mkdir third_party
ln -s $POPLAR_SDK_ENABLED third_party/poplar
ln -s $(cd $POPLAR_SDK_ENABLED/../popart-* && pwd) third_party/popart
# Add the following to .vscode/settings.json:
# "C_Cpp.default.includePath": [
#     "${workspaceFolder}/third_party/poplar/include",
#     "${workspaceFolder}/third_party/popart/include"
# ],
```

Note:

 - `./dev tests -k PATTERN` runs selected tests matching `PATTERN`
 - `./dev python ...` runs under Python, after building the native lib & setting `PYTHONPATH`
 - When adding `.cpp` files, they should also be added to `OBJECTS` in [Makefile](Makefile)
 - Custom codelets should have names of form `*_codelet.cpp`. They don't need to be added to `OBJECTS`; see
 [here](poptorch_experimental_addons/cpp/distance_matrix.cpp#L440) for an example of how to include them in custom ops.
 - When extending impl modules, note that we use `__all__` to control the public API


## License

Copyright (c) 2023 Graphcore Ltd. Licensed under the MIT License.

The included code is released under an MIT license, (see [LICENSE](LICENSE)).

Our dependencies are (see [requirements.txt](requirements.txt)):

| Component | About | License |
| --- | --- | --- |
| numpy | Array processing library | BSD 3-Clause |
| einops | Tensor processing utilities | MIT |

We also use additional Python dependencies for development/testing (see [requirements-dev.txt](requirements-dev.txt)).
