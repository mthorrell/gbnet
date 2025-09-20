from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import torch


def loadModule(module):
    """Load the appropriate gradient boosting module."""
    assert module in {"XGBModule", "LGBModule"}
    if module == "XGBModule":
        from gbnet import xgbmodule

        return xgbmodule.XGBModule
    if module == "LGBModule":
        from gbnet import lgbmodule

        return lgbmodule.LGBModule


class HazardIntegrator(torch.nn.Module):
    def __init__(
        self,
        covariate_cols: List[str] = [],
        params: Dict = {},
        min_hess: float = 0.0,
        module_type: str = "XGBModule",
    ):
        """
        Parameters
        ----------
        covariate_cols
            Columns to feed into the model. "time" is always included.
        params
            Parameters for the gradient boosting model.
        min_hess
            Minimum hessian for the gradient boosting model.
        module_type
            Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
            Defaults to "XGBModule".
        """
        super().__init__()
        self.params = params.copy()
        self.min_hess = min_hess
        self.module_type = module_type
        self.covariate_cols = ["time"] + covariate_cols
        self.gb_module: Optional[object] = None
        self.Module = loadModule(module_type)

        # Buffer to store pre-processed static data during training
        self.static_data: Dict[str, torch.Tensor] = {}

    def _prepare_data(self, df: pd.DataFrame):
        """
        Pre-processes and caches data that is static during training.
        This method performs sorting, tensor conversion, and computes
        time differences and group boundaries once.
        """
        if {"unit_id", "time"} - set(df.columns):
            raise ValueError("DataFrame must contain 'unit_id' and 'time'.")

        # 1. Sort by (unit_id, time) to enable group-wise operations
        df_sorted = df.sort_values(["unit_id", "time"], kind="mergesort").reset_index()

        # 2. Create tensors for indices, unit IDs, and times
        # These are fundamental for all subsequent calculations.
        orig_idx = torch.as_tensor(df_sorted["index"].values, dtype=torch.long)
        unit_codes, _ = pd.factorize(df_sorted["unit_id"])
        unit_ids = torch.as_tensor(unit_codes, dtype=torch.long)
        times = torch.as_tensor(df_sorted["time"].values, dtype=torch.float32)

        # 3. Pre-calculate time differences (dt) within each unit
        dt = torch.diff(times, prepend=times.new_zeros(1))
        # `same_unit` mask ensures we only calculate dt for records from the same unit
        same_unit = unit_ids == torch.roll(unit_ids, 1, 0)
        same_unit[0] = False  # The first element is never preceded by the same unit
        dt.mul_(same_unit)  # In-place multiplication for efficiency

        # 4. Create the input matrix for the gradient boosting model
        # Using .to_numpy() is generally faster than .values
        X = df_sorted[self.covariate_cols]

        # Create appropriate data matrix based on module type
        if self.module_type == "XGBModule":
            import xgboost as xgb

            dmatrix = xgb.DMatrix(X, enable_categorical=True)
        elif self.module_type == "LGBModule":
            import lightgbm as lgb

            dmatrix = lgb.Dataset(X)
        else:
            raise ValueError(f"Unsupported module type: {self.module_type}")

        # 5. Store all static tensors in the cache
        self.static_data = {
            "dmatrix": dmatrix,
            "unit_ids": unit_ids,
            "dt": dt,
            "same_unit": same_unit,
            "unsort_idx": torch.argsort(orig_idx),
            "interleave_amts": torch.as_tensor(
                df_sorted.groupby("unit_id").size().to_numpy(), dtype=torch.int64
            ),
        }

    def forward(self, df: pd.DataFrame) -> Dict[str, Any]:
        # During training, use cached data. In eval mode, re-process every time.
        if not self.static_data or not self.training:
            self._prepare_data(df)

        # Unpack cached tensors for cleaner access
        dmatrix = self.static_data["dmatrix"]
        unit_ids = self.static_data["unit_ids"]
        dt = self.static_data["dt"]
        same_unit = self.static_data["same_unit"]
        unsort_idx = self.static_data["unsort_idx"]
        interleave_amts = self.static_data["interleave_amts"]

        # Lazily initialize the gradient boosting module on the first forward pass
        if self.gb_module is None:
            num_rows, num_features = dmatrix.num_row(), dmatrix.num_col()
            self.gb_module = self.Module(
                num_rows, num_features, 1, params=self.params, min_hess=self.min_hess
            )

        # 1. Model inference: This is the dynamic part of the forward pass
        log_hazard = self.gb_module(dmatrix).flatten()  # [N]
        hazard = torch.exp(log_hazard)  # λ(t)

        # 2. Per-row trapezoidal slice for hazard integration: λ(t) * Δt
        # This is more efficient than the original `roll` and masking
        trapz_slice = torch.zeros_like(hazard)
        # We only compute slices where it's the same unit
        trapz_slice[same_unit] = (
            0.5 * (hazard[same_unit] + hazard.roll(1, 0)[same_unit]) * dt[same_unit]
        )

        # 3. Cumulative hazard per unit: Λ(T) = Σ λ(t)Δt
        # `scatter_reduce` is highly efficient for grouped sums.
        # We get both the total integrated hazard per unit...
        unit_Lambda = torch.zeros(
            interleave_amts.size(0), device=hazard.device
        ).scatter_reduce_(0, unit_ids, trapz_slice, reduce="sum", include_self=False)

        # ...and the cumulative hazard Λ(t) for each time step.
        # Your original logic is solid; we just use the cached tensors.
        Lambda_global = torch.cumsum(trapz_slice, dim=0)
        # Create a tensor of cumulative sums at the end of each *previous* group
        cusum_prev_groups = torch.cat(
            [
                torch.tensor([0.0], device=hazard.device),
                torch.cumsum(unit_Lambda, 0)[:-1],
            ]
        )
        # Broadcast this value to subtract it from each member of the next group
        Lambda = Lambda_global - torch.repeat_interleave(
            cusum_prev_groups, interleave_amts
        )

        # 4. Survival function S(t) = exp(‑Λ(t))
        survival = torch.exp(-Lambda)

        # 5. Expected value 𝔼[T] ≈ Σ S(t)Δt (trapezoidal integration)
        surv_slice = torch.zeros_like(survival)
        surv_slice[same_unit] = (
            0.5 * (survival[same_unit] + survival.roll(1, 0)[same_unit]) * dt[same_unit]
        )

        unit_E = torch.zeros_like(unit_Lambda).scatter_reduce_(
            0, unit_ids, surv_slice, reduce="sum", include_self=False
        )

        # 6. Extract the final hazard value for each unit
        # last_event_mask = torch.cat([same_unit[1:], torch.tensor([True])]) & ~same_unit
        # last_hazard = hazard[last_event_mask]

        is_last_in_group = torch.cat(
            (
                unit_ids[:-1] != unit_ids[1:],
                torch.tensor([True], device=unit_ids.device),
            )
        )
        last_hazard = hazard[is_last_in_group]

        # 7. Unsort results to match original DataFrame order
        return {
            "hazard": hazard[unsort_idx],
            "unit_last_hazard": last_hazard,
            "unit_integrated_hazard": unit_Lambda,
            "survival": survival[unsort_idx],
            "unit_expected_time": unit_E,
        }

    def gb_step(self):
        """Triggers the gradient boosting model to take a step."""
        if self.gb_module:
            self.gb_module.gb_step()


def expand_overlapping_units_locf(
    df: pd.DataFrame,
    unit_col: str = "unit_id",
    time_col: str = "time",
    fill_value=np.nan,
):
    # Unique times observed anywhere in the data, sorted
    all_times = np.sort(df[time_col].unique())

    # Min & max time for each unit
    t_min = df.groupby(unit_col)[time_col].min()
    t_max = df.groupby(unit_col)[time_col].max()

    # Skeleton of unit–time combinations
    pieces = []
    for unit in t_min.index:
        mask = (all_times >= t_min[unit]) & (all_times <= t_max[unit])
        pieces.append(pd.DataFrame({unit_col: unit, time_col: all_times[mask]}))
    skeleton = pd.concat(pieces, ignore_index=True)

    # Merge and sort
    out = (
        skeleton.merge(df, on=[unit_col, time_col], how="left")
        .sort_values([unit_col, time_col], kind="mergesort")
        .reset_index(drop=True)
    )

    # Identify covariate columns (excluding unit and time)
    covariate_cols = [col for col in df.columns if col not in {unit_col, time_col}]

    # LOCF: forward fill per unit
    out[covariate_cols] = out.groupby(unit_col)[covariate_cols].ffill()

    # Optional: still fill any remaining NaNs (e.g., if a unit starts mid-way)
    # out[covariate_cols] = out[covariate_cols].fillna(fill_value)

    return out


def to_integration_df(df, cols):
    df = df.copy()
    df["unit_id"] = range(df.shape[0])
    unit_metadata = df[["unit_id", "event"]].drop_duplicates()
    edf, mdf = (
        pd.concat(
            [df, pd.DataFrame([{"unit_id": i, "time": 0} for i in range(df.shape[0])])]
        )[["unit_id", "time"]]
        .merge(unit_metadata, on="unit_id", how="inner", validate="many_to_one")
        .sort_values(["unit_id", "time"])
        .reset_index(drop=True),
        df.copy(),
    )
    return edf.merge(
        mdf[["unit_id"] + cols], on="unit_id", how="left", validate="many_to_one"
    ).copy()
