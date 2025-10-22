from typing import Optional

import torch
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from .hazard_integrator import HazardIntegrator


class HazardSurvivalModel(BaseEstimator, RegressorMixin):
    """Gradient Boosting Hazard Integration Survival Model.

    This model combines gradient boosting with hazard integration for continuous
    survival analysis. It uses either XGBoost or LightGBM as the underlying
    boosting engine wrapped in a PyTorch module.

    The model supports both static and time-varying datasets:
    - Static datasets: Each unit has one observation with static features
    - Time-varying datasets: Each unit has multiple observations over time

    Parameters
    ----------
    nrounds : int, optional
        Number of boosting rounds. Defaults to 100.
    params : dict, optional
        Additional parameters passed to the gradient boosting model.
    module_type : str, optional
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
        Defaults to "XGBModule".
    min_hess : float, optional
        Minimum hessian value for numerical stability. Defaults to 0.0.

    Attributes
    ----------
    integrator_ : HazardIntegrator
        Trained hazard integrator module. Set after fitting.
    losses_ : list
        List of loss values recorded at each training iteration.
    data_format_ : str
        Detected data format: 'static' or 'time_varying'.

    Methods
    -------
    fit(X, y)
        Trains the model using input features X and survival data y.
    predict_survival(X, times)
        Predicts survival probabilities for given times.
    predict_hazard(X, times)
        Predicts hazard values for given times.
    predict(X)
        Predicts the expected survival time.
    score(X, y)
        Returns the negative log likelihood score.

    Notes
    -----
    The model uses hazard integration to model continuous survival times.
    The gradient boosting model learns hazard rates for each time point,
    which are then integrated to compute survival probabilities.

    Supported data formats:
    - Static: X is DataFrame with static features, y is DataFrame with 'time', 'event', 'unit_id'
    - Time-varying: X is DataFrame with time-varying features, y has 'time', 'event', 'unit_id'

    For survival data, y must be a DataFrame containing:
    - 'time': observed time (continuous)
    - 'event': event indicator (0=censored, 1=event)
    - 'unit_id': unique identifier for each unit/subject
    """

    def __init__(
        self,
        nrounds=None,
        params=None,
        module_type="LGBModule",
        min_hess=0.0,
    ):
        if params is None:
            params = {"max_delta_step": 1 if module_type == "XGBModule" else 5}

        self.params = params
        self.module_type = module_type
        # Default boosting rounds depend on module type
        if nrounds is None:
            nrounds = 50 if module_type == "XGBModule" else 100
        self.nrounds = nrounds
        self.min_hess = min_hess
        # self.integrator_ = None
        self.losses_ = []
        self.data_format_ = None

    def _static_to_minimal_time_varying_dataset(self, X):
        assert "time" in X and "unit_id" in X and X["unit_id"].is_unique
        X0 = X.copy()
        X0["time"] = 0
        return (
            pd.concat([X0, X], axis=0)
            .sort_values(["unit_id", "time"])
            .reset_index(drop=True)
            .copy()
        )

    def _validate_and_convert_input_data(self, X, y):
        """Validate input data according to the new requirements.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.DataFrame
            Survival data with 'time', 'event', 'unit_id' columns

        Returns
        -------
        tuple
            (data_format, modified_X) where data_format is 'static' or 'time_varying'
            and modified_X is the potentially modified X DataFrame
        """
        X = X.copy()
        assert "unit_id" in X
        if isinstance(y, pd.DataFrame):
            assert "unit_id" in y
            assert "time" in y
            assert "event" in y
            assert y["unit_id"].is_unique

        is_static = X["unit_id"].is_unique
        is_time_varying = not is_static

        if is_time_varying:
            unit_counts = X["unit_id"].value_counts()
            assert not (unit_counts < 2).any()
            assert "time" in X

        modified_X = X.copy()
        if is_static:
            if "time" not in X.columns:
                # Copy time from y to X
                if isinstance(y, np.ndarray):
                    modified_X = X.copy()
                    modified_X["time"] = max(y)
                else:
                    modified_X = X.merge(
                        y[["unit_id", "time"]], on="unit_id", how="left"
                    ).copy()
            else:
                # Validate that time columns match when joining on unit_id
                merged = X[["unit_id", "time"]].merge(
                    y[["unit_id", "time"]], on="unit_id", suffixes=("_x", "_y")
                )
                if not (merged["time_x"] == merged["time_y"]).all():
                    raise ValueError(
                        "For static datasets, 'time' column in X must match 'time' column in y when joining on 'unit_id'"
                    )
            modified_X = self._static_to_minimal_time_varying_dataset(modified_X)

        modified_X = (
            expand_overlapping_units_locf(modified_X)
            if not isinstance(y, np.ndarray)
            else expand_overlapping_units_locf(modified_X, y)
        )

        return ("static" if is_static else "time_varying", modified_X, y)

    def fit(self, X, y):
        """Fit the hazard integration survival model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features. Can be static or time-varying.
        y : pd.DataFrame
            Survival data with 'time', 'event', 'unit_id' columns.

        Returns
        -------
        self : object
            Returns self.
        """
        # Ensure X and y are DataFrames
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a pandas DataFrame")

        self.max_time = y["time"].max()
        self.data_format_, self.exp_df, self.y = self._validate_and_convert_input_data(
            X, y
        )

        # Pre-compute event indicators for efficiency
        self.event_indicators_ = self.y.groupby("unit_id")["event"].last().values
        self.n_samples_ = len(self.event_indicators_)

        # Initialize hazard integrator with appropriate covariate columns
        covariate_cols = [
            col for col in X.columns if col not in ["unit_id", "time", "event"]
        ]

        self.integrator_ = HazardIntegrator(
            covariate_cols=covariate_cols,
            params=self.params,
            min_hess=self.min_hess,
            module_type=self.module_type,
        )

        # Training loop
        self.losses_ = []

        for i in range(self.nrounds):
            self.integrator_.train()
            self.integrator_.zero_grad()

            out = self.integrator_(self.exp_df, return_survival_estimates=False)

            # Negative log-likelihood loss using pre-computed event indicators
            loss = (
                out["unit_integrated_hazard"].sum()
                - (
                    torch.log(out["unit_last_hazard"])
                    * torch.tensor(self.event_indicators_ == 1, dtype=torch.float32)
                ).sum()
            ) / self.n_samples_

            loss.backward(create_graph=True)
            self.losses_.append(loss.item())

            self.integrator_.gb_step()

        self.integrator_.eval()
        return self

    def predict_base(self, X, y):
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.DataFrame):
            raise ValueError("y must be a pandas DataFrame")

        data_format_, exp_df, y = self._validate_and_convert_input_data(X, y)
        return self.integrator_(exp_df)

    def predict_times(self, X, times=None):
        check_is_fitted(self, "integrator_")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")

        if times is None:
            times = np.linspace(0, self.max_time, 100)

        X = X.copy()

        data_format_, exp_df, y = self._validate_and_convert_input_data(X, times)
        exp_df = exp_df.reset_index(drop=True).copy()
        output = self.integrator_(exp_df)

        exp_df["hazard"] = output["hazard"]
        exp_df["survival"] = output["survival"]

        udf = exp_df[["unit_id"]].drop_duplicates().reset_index(drop=True).copy()
        udf["last_hazard"] = output["unit_last_hazard"]
        udf["integrated_hazard"] = output["unit_integrated_hazard"]
        udf["expected_time"] = output["unit_expected_time"]

        return exp_df, udf

    def predict_survival(self, X, times=None):
        check_is_fitted(self, "integrator_")
        exp_df, udf = self.predict_times(X, times)
        return exp_df[["unit_id", "time", "survival", "hazard"]]

    def predict(self, X, times=None):
        check_is_fitted(self, "integrator_")
        exp_df, udf = self.predict_times(X, times)

        median = (
            exp_df[exp_df["survival"] > 0.5]
            .groupby("unit_id")["time"]
            .max()
            .rename("predicted_median_time")
            .reset_index()
        )

        output = udf[["unit_id", "expected_time"]].merge(
            median, on="unit_id", how="left", validate="one_to_one"
        )
        output["predicted_median_time"] = output["predicted_median_time"].fillna(0)
        return output


def expand_overlapping_units_locf(
    df: pd.DataFrame,
    y: Optional[np.ndarray] = None,
    unit_col: str = "unit_id",
    time_col: str = "time",
):
    # Unique times observed anywhere in the data, sorted
    if y is None:
        all_times = np.sort(df[time_col].unique())
    else:
        all_times = np.sort(
            np.unique(np.concatenate([df[time_col].values, np.asarray(y)]))
        )

    # Min & max time for each unit
    t_min = df.groupby(unit_col)[time_col].min()
    t_max = df.groupby(unit_col)[time_col].max()

    # Skeleton of unit–time combinations
    if y is None:
        pieces = []
        for unit in t_min.index:
            mask = (all_times >= t_min[unit]) & (all_times <= t_max[unit])
            pieces.append(pd.DataFrame({unit_col: unit, time_col: all_times[mask]}))
        skeleton = pd.concat(pieces, ignore_index=True)
    else:
        skeleton = (
            df[[unit_col]]
            .drop_duplicates()
            .merge(pd.DataFrame({"time": all_times}), how="cross")
        )

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
