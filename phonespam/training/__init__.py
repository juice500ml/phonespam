"""Training-time utilities for phonespam models.

Importing this submodule requires the optional ``[train]`` extra (praatio,
tqdm). ``prepare_datasets`` additionally needs ``phone-metrics``, which is
not on PyPI -- see the README for its install line. Pure inference users
need none of it.

Submodules are imported lazily so that ``python -m phonespam.training.<name>``
doesn't hit a RuntimeWarning from ``runpy`` finding the submodule already in
``sys.modules``.
"""

import importlib

_LAZY = ("extract_features", "prepare_datasets", "train")

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
