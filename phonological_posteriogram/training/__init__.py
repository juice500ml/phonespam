"""Training-time utilities for phonological-posteriogram models.

Importing this submodule requires the optional `[train]` extra (praatio,
scikit-learn, panphon, tqdm). Pure inference users don't need it.
"""

from . import evaluate, extract_features, prepare_datasets, train

__all__ = ["evaluate", "extract_features", "prepare_datasets", "train"]
