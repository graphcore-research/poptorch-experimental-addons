# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

"""
A collection of addons to [PopTorch](https://github.com/graphcore/poptorch),
with general utility.

```python
import poptorch_experimental_addons as pea
```

Addons are provided as standalone functions and à la carte via submodules
- please explore these to see if `pea` has something useful for you.
"""


def _load_native_library() -> None:
    import ctypes
    import pathlib
    import sysconfig

    root = pathlib.Path(__file__).parent.parent.absolute()
    name = "libpoptorch_experimental_addons.so"
    paths = [
        root / "build" / name,
        (root / name).with_suffix(sysconfig.get_config_vars()["SO"]),
    ]
    for path in paths:
        if path.exists():
            ctypes.cdll.LoadLibrary(str(path))
            return
    raise ImportError(  # pragma: no cover
        f"Cannot find extension library {name} - tried {[str(p) for p in paths]}"
    )


_load_native_library()

from . import collectives, sharded, sparse  # NOQA:F401,E402
from ._impl.core import *  # NOQA:F401,E402,F403
from ._impl.core import __all__  # NOQA:F401,E402
