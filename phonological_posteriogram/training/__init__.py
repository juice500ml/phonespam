"""Training-time utilities for phonological-posteriogram models.

Importing this submodule requires the optional `[train]` extra (praatio,
scikit-learn, panphon, tqdm). Pure inference users don't need it.
"""

from . import extract_features, prepare_datasets

__all__ = ["extract_features", "prepare_datasets"]
