"""Exercise an installed GBNet distribution, without pytest or training data.

Run with ``python -I scripts/smoke_install.py`` to exclude the source checkout
from Python's import path. Each import order gets its own process so native
crashes and deadlocks produce an actionable failure instead of killing CI.
"""

import argparse
import ctypes
import importlib
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path


def fit_models(first_module):
    # Import a GBNet entry point first, as a user would. On macOS this must
    # initialize PyTorch before LightGBM or XGBoost loads its native library.
    importlib.import_module(first_module)

    import numpy as np
    import pandas as pd
    import torch

    import gbnet
    from gbnet import lgbmodule, xgbmodule
    from gbnet.models.forecasting import Forecast

    print(f"GBNet {version('gbnet')}: {Path(gbnet.__file__).resolve()}", flush=True)
    for dependency in (
        "torch",
        "numpy",
        "pandas",
        "scipy",
        "scikit-learn",
        "xgboost",
        "lightgbm",
    ):
        print(f"  {dependency} {version(dependency)}", flush=True)

    if sys.platform == "darwin":
        # Inspect dyld directly: querying incompatible OpenMP runtimes through
        # threadpoolctl can itself crash. A clean install must not need a
        # Homebrew or MacPorts runtime outside this environment.
        dyld = ctypes.CDLL(None)
        dyld._dyld_image_count.restype = ctypes.c_uint32
        dyld._dyld_get_image_name.argtypes = [ctypes.c_uint32]
        dyld._dyld_get_image_name.restype = ctypes.c_char_p
        images = [
            Path(dyld._dyld_get_image_name(index).decode()).resolve()
            for index in range(dyld._dyld_image_count())
        ]
        openmp_paths = [path for path in images if path.name == "libomp.dylib"]
        assert openmp_paths, "No OpenMP runtime loaded"
        assert all(
            path.is_relative_to(Path(sys.prefix).resolve()) for path in openmp_paths
        ), openmp_paths
        print(f"  OpenMP runtimes: {openmp_paths}", flush=True)

    rng = np.random.default_rng(42)
    features = rng.normal(size=(96, 3)).astype(np.float32)
    target = torch.tensor(2 * features[:, :1] + features[:, 1:2])
    # Use multiple native threads: forcing OMP_NUM_THREADS=1 can hide conflicts.
    for cls, params in (
        (xgbmodule.XGBModule, {"nthread": 2, "max_depth": 2, "eta": 0.3}),
        (lgbmodule.LGBModule, {"num_threads": 2, "num_leaves": 7, "verbose": -1}),
    ):
        model = cls(96, 3, 1, params=params)
        losses = []
        for _ in range(8):
            model.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model(features), target)
            losses.append(loss.item())
            loss.backward(create_graph=True)
            model.gb_step()
        model.eval()
        predictions = model(features).detach().numpy()
        assert predictions.shape == (96, 1)
        assert np.isfinite(predictions).all()
        assert losses[-1] < losses[0], (cls.__name__, losses)
        print(f"  {cls.__name__}: fit/predict OK", flush=True)

    frame = pd.DataFrame(
        {
            "ds": pd.date_range("2024-01-01", periods=120),
            "y": np.arange(120) * 0.1 + np.sin(np.arange(120) * 2 * np.pi / 7),
        }
    )
    for backend, params in (
        ("XGBModule", {"nthread": 2}),
        ("LGBModule", {"num_threads": 2, "verbose": -1}),
    ):
        model = Forecast(
            nrounds=8,
            module_type=backend,
            params=params,
            changepoint_params={"nthread": 2},
            estimate_uncertainty=False,
        )
        model.fit(frame.iloc[:100], frame["y"].iloc[:100])
        predictions = model.predict(frame.iloc[100:])
        assert len(predictions) == 20
        assert np.isfinite(predictions["yhat"]).all()
        assert model.losses_[-1] < model.losses_[0]
        print(f"  Forecast({backend}): fit/predict OK", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-module", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.first_module:
        fit_models(args.first_module)
        return

    for first_module in (
        "gbnet.lgbmodule",
        "gbnet.xgbmodule",
        "gbnet.models.forecasting",
    ):
        print(f"Testing {first_module} first", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                "faulthandler",
                str(Path(__file__).resolve()),
                "--first-module",
                first_module,
            ],
            check=True,
            timeout=180,
        )
    print("Installation smoke tests passed.", flush=True)


if __name__ == "__main__":
    main()
