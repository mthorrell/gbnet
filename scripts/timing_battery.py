#!/usr/bin/env python3
"""Timing battery for XGBModule, LGBModule, and GBLinear.

Core API:
- run_battery(...): run benchmarks and return a dataframe.
- run_battery_and_record(...): run benchmarks and optionally persist JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch
import xgboost as xgb

from gbnet.gblinear import GBLinear
from gbnet.lgbmodule import LGBModule
from gbnet.xgbmodule import XGBModule

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_HISTORY_PATH = REPO_ROOT / "benchmarks" / "timing_history.json"
DISPLAY_COLUMNS = [
    "module",
    "fit_mean_s",
    "fit_std_s",
    "fit_ms_per_round",
    "predict_mean_ms",
    "predict_std_ms",
    "final_loss_mean",
    "final_loss_std",
]


@dataclass
class BatteryConfig:
    n_samples: int
    n_features: int
    output_dim: int
    n_rounds: int
    repeats: int
    predict_passes: int
    seed: int
    num_threads: int
    gblinear_lr: float
    gblinear_lambd: float


DEFAULT_BATTERY_CONFIG = BatteryConfig(
    n_samples=3000,
    n_features=32,
    output_dim=1,
    n_rounds=100,
    repeats=10,
    predict_passes=50,
    seed=2026,
    num_threads=1,
    gblinear_lr=0.5,
    gblinear_lambd=0.01,
)


def _parse_json_dict(raw: str, flag_name: str) -> dict[str, Any]:
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError(f"{flag_name} must decode to a JSON object/dict.")
    return parsed


def _git_commit(repo_root: Path) -> str:
    out = subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return out.strip()


def _git_is_dirty(repo_root: Path) -> bool:
    out = subprocess.check_output(
        ["git", "-C", str(repo_root), "status", "--porcelain"],
        stderr=subprocess.DEVNULL,
        text=True,
    )
    return bool(out.strip())


def _utc_now_iso() -> str:
    return (
        datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
    )


def _run_key(run_datetime_utc: str, commit: str | None) -> str:
    return run_datetime_utc if commit is None else f"{run_datetime_utc}|{commit}"


def _mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0, 0.0
    if arr.size == 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def _make_data(config: BatteryConfig) -> tuple[np.ndarray, torch.Tensor]:
    rng = np.random.default_rng(config.seed)
    x = rng.normal(size=(config.n_samples, config.n_features)).astype(np.float32)
    beta = rng.normal(size=(config.n_features, config.output_dim)).astype(np.float32)
    noise = 0.1 * rng.normal(size=(config.n_samples, config.output_dim)).astype(
        np.float32
    )
    y = x @ beta + noise
    y_tensor = torch.tensor(y, dtype=torch.float32)
    return x, y_tensor


def _fit_model(
    model: torch.nn.Module,
    x_train: Any,
    y_tensor: torch.Tensor,
    n_rounds: int,
) -> tuple[float, float]:
    loss_fn = torch.nn.MSELoss()
    model.train()
    start = perf_counter()
    last_loss = float("nan")

    for _ in range(n_rounds):
        model.zero_grad(set_to_none=True)
        preds = model(x_train)
        loss = loss_fn(preds, y_tensor)
        loss.backward(create_graph=True)
        model.gb_step()
        last_loss = float(loss.detach().cpu().item())

    return perf_counter() - start, last_loss


def _predict_time(model: torch.nn.Module, x_pred: Any, predict_passes: int) -> float:
    model.eval()
    with torch.no_grad():
        _ = model(x_pred)  # warmup
        start = perf_counter()
        for _ in range(predict_passes):
            out = model(x_pred)
            if isinstance(out, torch.Tensor):
                _ = out.detach()
        return perf_counter() - start


def _benchmark_module(
    module_name: str,
    make_model: Callable[[int], torch.nn.Module],
    x_train: Any,
    x_pred: Any,
    y_tensor: torch.Tensor,
    config: BatteryConfig,
) -> dict[str, Any]:
    fit_times: list[float] = []
    pred_times: list[float] = []
    final_losses: list[float] = []

    for repeat_idx in range(config.repeats):
        model = make_model(config.seed + repeat_idx)
        fit_seconds, final_loss = _fit_model(model, x_train, y_tensor, config.n_rounds)
        pred_seconds = _predict_time(model, x_pred, config.predict_passes)

        fit_times.append(fit_seconds)
        pred_times.append(pred_seconds)
        final_losses.append(final_loss)

    fit_mean_s, fit_std_s = _mean_std(fit_times)
    pred_mean_s, pred_std_s = _mean_std(pred_times)
    loss_mean, loss_std = _mean_std(final_losses)

    return {
        "module": module_name,
        "fit_mean_s": fit_mean_s,
        "fit_std_s": fit_std_s,
        "fit_ms_per_round": (fit_mean_s / config.n_rounds) * 1000.0,
        "predict_mean_ms": (pred_mean_s / config.predict_passes) * 1000.0,
        "predict_std_ms": (pred_std_s / config.predict_passes) * 1000.0,
        "final_loss_mean": loss_mean,
        "final_loss_std": loss_std,
        "raw_fit_seconds": fit_times,
        "raw_predict_seconds": pred_times,
        "raw_final_losses": final_losses,
    }


def run_battery(
    config: BatteryConfig,
    xgb_params: dict[str, Any] | None = None,
    lgb_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run benchmark battery and return one row per module."""
    xgb_params = {} if xgb_params is None else xgb_params
    lgb_params = {} if lgb_params is None else lgb_params

    torch.set_num_threads(config.num_threads)
    x_np, y_tensor = _make_data(config)

    xgb_defaults = {
        "tree_method": "hist",
        "learning_rate": 0.1,
        "max_depth": 6,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "verbosity": 0,
        "nthread": config.num_threads,
    }
    lgb_defaults = {
        "learning_rate": 0.1,
        "num_leaves": 31,
        "min_data_in_leaf": 1,
        "verbose": -1,
        "verbosity": -1,
        "num_threads": config.num_threads,
    }

    xgb_train = xgb.DMatrix(x_np)
    xgb_pred = xgb.DMatrix(x_np)

    def make_xgb_model(rep_seed: int) -> XGBModule:
        params = {**xgb_defaults, "seed": rep_seed, **xgb_params}
        return XGBModule(
            batch_size=config.n_samples,
            input_dim=config.n_features,
            output_dim=config.output_dim,
            params=params,
        )

    def make_lgb_model(rep_seed: int) -> LGBModule:
        params = {**lgb_defaults, "seed": rep_seed, **lgb_params}
        return LGBModule(
            batch_size=config.n_samples,
            input_dim=config.n_features,
            output_dim=config.output_dim,
            params=params,
        )

    def make_gblinear_model(rep_seed: int) -> GBLinear:
        torch.manual_seed(rep_seed)
        return GBLinear(
            input_dim=config.n_features,
            output_dim=config.output_dim,
            lr=config.gblinear_lr,
            lambd=config.gblinear_lambd,
        )

    rows = [
        _benchmark_module(
            module_name="XGBModule",
            make_model=make_xgb_model,
            x_train=xgb_train,
            x_pred=xgb_pred,
            y_tensor=y_tensor,
            config=config,
        ),
        _benchmark_module(
            module_name="LGBModule",
            make_model=make_lgb_model,
            x_train=x_np,
            x_pred=x_np,
            y_tensor=y_tensor,
            config=config,
        ),
        _benchmark_module(
            module_name="GBLinear",
            make_model=make_gblinear_model,
            x_train=x_np,
            x_pred=x_np,
            y_tensor=y_tensor,
            config=config,
        ),
    ]

    return pd.DataFrame(rows)


def build_run_payload(
    *,
    config: BatteryConfig,
    xgb_params: dict[str, Any],
    lgb_params: dict[str, Any],
    results_df: pd.DataFrame,
    include_git_metadata: bool = True,
    repo_root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """Build a single run payload for JSON persistence."""
    run_datetime_utc = _utc_now_iso()
    commit = _git_commit(repo_root) if include_git_metadata else None
    git_dirty = _git_is_dirty(repo_root) if include_git_metadata else None

    return {
        "run_key": _run_key(run_datetime_utc, commit),
        "run_datetime_utc": run_datetime_utc,
        "commit": commit,
        "git_dirty": git_dirty,
        "config": {
            "n_samples": config.n_samples,
            "n_features": config.n_features,
            "output_dim": config.output_dim,
            "n_rounds": config.n_rounds,
            "repeats": config.repeats,
            "predict_passes": config.predict_passes,
            "seed": config.seed,
            "num_threads": config.num_threads,
            "gblinear_lr": config.gblinear_lr,
            "gblinear_lambd": config.gblinear_lambd,
            "xgb_params": xgb_params,
            "lgb_params": lgb_params,
        },
        "results": results_df.to_dict(orient="records"),
    }


def run_battery_and_record(
    *,
    config: BatteryConfig,
    xgb_params: dict[str, Any] | None = None,
    lgb_params: dict[str, Any] | None = None,
    save_outputs: bool = True,
    history_path: Path = DEFAULT_HISTORY_PATH,
    run_json_path: Path | None = None,
    include_git_metadata: bool = True,
    repo_root: Path = REPO_ROOT,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run battery and optionally save history/run JSON outputs."""
    xgb_params = {} if xgb_params is None else xgb_params
    lgb_params = {} if lgb_params is None else lgb_params

    results_df = run_battery(
        config=config, xgb_params=xgb_params, lgb_params=lgb_params
    )
    payload = build_run_payload(
        config=config,
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        results_df=results_df,
        include_git_metadata=include_git_metadata,
        repo_root=repo_root,
    )

    if save_outputs:
        payload["run_key"] = _upsert_history(history_path, payload)
        if run_json_path is not None:
            _write_json(run_json_path, payload)

    return results_df, payload


def _upsert_history(history_path: Path, payload: dict[str, Any]) -> str:
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path.exists():
        with history_path.open("r", encoding="utf-8") as f:
            history = json.load(f)
    else:
        history = {"schema_version": 1, "runs": {}}

    if "runs" not in history or not isinstance(history["runs"], dict):
        history["runs"] = {}
    if "schema_version" not in history:
        history["schema_version"] = 1

    run_key = payload["run_key"]
    if run_key in history["runs"]:
        suffix = 2
        candidate = f"{run_key}#{suffix}"
        while candidate in history["runs"]:
            suffix += 1
            candidate = f"{run_key}#{suffix}"
        run_key = candidate
        payload["run_key"] = run_key

    history["runs"][run_key] = payload
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history, f, indent=2, sort_keys=True)
        f.write("\n")
    return run_key


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _cli() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Timing battery for XGBModule, LGBModule, and GBLinear."
    )
    parser.add_argument(
        "--n-samples", type=int, default=DEFAULT_BATTERY_CONFIG.n_samples
    )
    parser.add_argument(
        "--n-features", type=int, default=DEFAULT_BATTERY_CONFIG.n_features
    )
    parser.add_argument(
        "--output-dim", type=int, default=DEFAULT_BATTERY_CONFIG.output_dim
    )
    parser.add_argument("--n-rounds", type=int, default=DEFAULT_BATTERY_CONFIG.n_rounds)
    parser.add_argument("--repeats", type=int, default=DEFAULT_BATTERY_CONFIG.repeats)
    parser.add_argument(
        "--predict-passes",
        type=int,
        default=DEFAULT_BATTERY_CONFIG.predict_passes,
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_BATTERY_CONFIG.seed)
    parser.add_argument(
        "--num-threads",
        type=int,
        default=DEFAULT_BATTERY_CONFIG.num_threads,
    )
    parser.add_argument(
        "--gblinear-lr", type=float, default=DEFAULT_BATTERY_CONFIG.gblinear_lr
    )
    parser.add_argument(
        "--gblinear-lambd",
        type=float,
        default=DEFAULT_BATTERY_CONFIG.gblinear_lambd,
    )
    parser.add_argument(
        "--xgb-params-json",
        type=str,
        default="{}",
        help="JSON dict merged into XGB params, e.g. '{\"max_depth\":4}'",
    )
    parser.add_argument(
        "--lgb-params-json",
        type=str,
        default="{}",
        help="JSON dict merged into LGB params, e.g. '{\"num_leaves\":63}'",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=DEFAULT_HISTORY_PATH,
        help="Path to history JSON file where all runs are appended.",
    )
    parser.add_argument(
        "--run-json-path",
        type=Path,
        default=None,
        help="Optional path to write just this run payload.",
    )
    parser.add_argument(
        "--no-save-outputs",
        action="store_true",
        help="Run benchmark but do not write history or run JSON files.",
    )
    parser.add_argument(
        "--no-git-metadata",
        action="store_true",
        help="Do not include git commit/dirty metadata in payload.",
    )
    return parser.parse_args()


def main() -> None:
    args = _cli()
    xgb_params = _parse_json_dict(args.xgb_params_json, "--xgb-params-json")
    lgb_params = _parse_json_dict(args.lgb_params_json, "--lgb-params-json")

    config = BatteryConfig(
        n_samples=args.n_samples,
        n_features=args.n_features,
        output_dim=args.output_dim,
        n_rounds=args.n_rounds,
        repeats=args.repeats,
        predict_passes=args.predict_passes,
        seed=args.seed,
        num_threads=args.num_threads,
        gblinear_lr=args.gblinear_lr,
        gblinear_lambd=args.gblinear_lambd,
    )

    df, payload = run_battery_and_record(
        config=config,
        xgb_params=xgb_params,
        lgb_params=lgb_params,
        save_outputs=not args.no_save_outputs,
        history_path=args.history_path,
        run_json_path=args.run_json_path,
        include_git_metadata=not args.no_git_metadata,
        repo_root=REPO_ROOT,
    )

    print(f"Run key: {payload['run_key']}")
    if args.no_save_outputs:
        print("Outputs not saved (--no-save-outputs).")
    else:
        print(f"History file: {args.history_path}")

    print(df[DISPLAY_COLUMNS].to_string(index=False, float_format=lambda v: f"{v:.6f}"))


if __name__ == "__main__":
    main()
