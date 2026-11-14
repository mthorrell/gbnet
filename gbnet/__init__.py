"""Gradient boosting modules integrated with PyTorch."""

import sys

if sys.platform == "darwin":
    # Load the compatible wheel's bundled OpenMP before a boosting backend can
    # resolve libomp through Homebrew. This also covers `from gbnet import
    # lgbmodule`, whose first third-party import is otherwise LightGBM.
    import torch as _torch  # noqa: F401
