from typing import List, Dict, Any, Optional

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
        integration_method: str = "trapezoid",
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
        integration_method
            Method for integrating hazards and survival estimates. One of
            "trapezoid", "stepwise_left", or "stepwise_right".
        """
        super().__init__()
        assert integration_method in {"trapezoid", "stepwise_left", "stepwise_right"}
        self.params = params.copy()
        self.min_hess = min_hess
        self.module_type = module_type
        self.integration_method = integration_method
        self.covariate_cols = ["time"] + covariate_cols
        self.gb_module: Optional[object] = None
        self.Module = loadModule(module_type)

        # Buffer to store pre-processed static data during training
        self.static_data: Dict[str, torch.Tensor] = {}

    def _integrate_slice(self, values, dt, same_unit):
        slice_values = torch.zeros_like(values)
        if self.integration_method == "trapezoid":
            slice_values[same_unit] = (
                0.5 * (values[same_unit] + values.roll(1, 0)[same_unit]) * dt[same_unit]
            )
        elif self.integration_method == "stepwise_left":
            slice_values[same_unit] = values.roll(1, 0)[same_unit] * dt[same_unit]
        elif self.integration_method == "stepwise_right":
            slice_values[same_unit] = values[same_unit] * dt[same_unit]
        return slice_values

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
            "num_rows": X.shape[0],
            "num_cols": X.shape[1],
            "unit_ids": unit_ids,
            "dt": dt,
            "same_unit": same_unit,
            "unsort_idx": torch.argsort(orig_idx),
            "interleave_amts": torch.as_tensor(
                df_sorted.groupby("unit_id").size().to_numpy(), dtype=torch.int64
            ),
        }

    def forward(
        self, df: pd.DataFrame, return_survival_estimates: bool = True
    ) -> Dict[str, Any]:
        # During training, use cached data. In eval mode, re-process every time.
        if not self.static_data or not self.training:
            self._prepare_data(df)

        # Unpack cached tensors for cleaner access
        dmatrix = self.static_data["dmatrix"]
        num_rows = self.static_data["num_rows"]
        num_features = self.static_data["num_cols"]
        unit_ids = self.static_data["unit_ids"]
        dt = self.static_data["dt"]
        same_unit = self.static_data["same_unit"]
        unsort_idx = self.static_data["unsort_idx"]
        interleave_amts = self.static_data["interleave_amts"]

        # Lazily initialize the gradient boosting module on the first forward pass
        if self.gb_module is None:
            self.gb_module = self.Module(
                num_rows, num_features, 1, params=self.params, min_hess=self.min_hess
            )

        # 1. Model inference: This is the dynamic part of the forward pass
        log_hazard = self.gb_module(dmatrix).flatten()  # [N]
        hazard = torch.exp(log_hazard)  # λ(t)

        # 2. Per-row slice for hazard integration: λ(t) * Δt
        hazard_slice = self._integrate_slice(hazard, dt, same_unit)

        # 3. Cumulative hazard per unit: Λ(T) = Σ λ(t)Δt
        # `scatter_reduce` is highly efficient for grouped sums.
        # We get both the total integrated hazard per unit...
        unit_Lambda = torch.zeros(
            interleave_amts.size(0), device=hazard.device
        ).scatter_reduce_(0, unit_ids, hazard_slice, reduce="sum", include_self=False)

        is_last_in_group = torch.cat(
            (
                unit_ids[:-1] != unit_ids[1:],
                torch.tensor([True], device=unit_ids.device),
            )
        )
        last_hazard = hazard[is_last_in_group]

        if return_survival_estimates:
            # ...and the cumulative hazard Λ(t) for each time step.
            # Your original logic is solid; we just use the cached tensors.
            Lambda_global = torch.cumsum(hazard_slice, dim=0)
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

            # 5. Expected value 𝔼[T] ≈ Σ S(t)Δt
            surv_slice = self._integrate_slice(survival, dt, same_unit)

            unit_E = torch.zeros_like(unit_Lambda).scatter_reduce_(
                0, unit_ids, surv_slice, reduce="sum", include_self=False
            )

        # 7. Unsort results to match original DataFrame order
        return {
            "hazard": hazard[unsort_idx],
            "unit_last_hazard": last_hazard,
            "unit_integrated_hazard": unit_Lambda,
            "survival": None if not return_survival_estimates else survival[unsort_idx],
            "unit_expected_time": None if not return_survival_estimates else unit_E,
        }

    def gb_step(self):
        """Triggers the gradient boosting model to take a step."""
        if self.gb_module:
            self.gb_module.gb_step()
