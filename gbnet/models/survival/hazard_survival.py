import torch
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

from .hazard_integrator import (
    HazardIntegrator,
    to_integration_df,
    expand_overlapping_units_locf,
)


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
    covariate_cols : list of str, optional
        List of covariate column names to use as features. "time" is always included.
        Defaults to empty list.
    time_varying_cols : list of str, optional
        List of column names that vary over time (for static data expansion).
        Defaults to empty list.
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
    n_features_in_ : int
        Number of features seen during fit.
    data_format_ : str
        Detected data format: 'static' or 'time_varying'.

    Methods
    -------
    fit(X, y, auto_expand_time=True)
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
        covariate_cols=None,
        time_varying_cols=None,
        nrounds=100,
        params=None,
        module_type="XGBModule",
        min_hess=0.0,
    ):
        if covariate_cols is None:
            covariate_cols = []
        if time_varying_cols is None:
            time_varying_cols = []
        if params is None:
            params = {}

        self.covariate_cols = covariate_cols
        self.time_varying_cols = time_varying_cols
        self.nrounds = nrounds
        self.params = params
        self.module_type = module_type
        self.min_hess = min_hess
        self.integrator_ = None
        self.losses_ = []
        self.n_features_in_ = None
        self.data_format_ = None

    def _detect_data_format(self, X, y):
        """Detect whether the input data is in static or time-varying format.

        Parameters
        ----------
        X : pd.DataFrame
            Input features
        y : pd.DataFrame
            Survival data with 'time', 'event', 'unit_id' columns

        Returns
        -------
        str
            Either 'static' or 'time_varying'
        """
        # Check if y has time-varying structure (multiple rows per unit_id)
        if "unit_id" in y.columns:
            units_with_multiple_obs = y.groupby("unit_id").size()
            has_time_varying = (units_with_multiple_obs > 1).any()
        else:
            has_time_varying = False

        # Check if X has time-varying structure (DataFrame with time column)
        if "time" in X.columns:
            has_time_varying = True

        return "time_varying" if has_time_varying else "static"

    def _expand_static_data(self, X, y):
        """Convert static data to time-varying format using existing transformation functions.

        Parameters
        ----------
        X : pd.DataFrame
            Static input features (with unit_id added if needed)
        y : pd.DataFrame
            Static survival data with time, event, and unit_id columns

        Returns
        -------
        pd.DataFrame
            Time-varying DataFrame ready for HazardIntegrator
        """
        # Combine X and y, excluding duplicate unit_id column
        combined_df = pd.concat([X, y], axis=1)

        # Use to_integration_df to create basic time-varying structure
        integration_df = to_integration_df(combined_df, X.columns.to_list())
        expanded_df = expand_overlapping_units_locf(
            integration_df, unit_col="unit_id", time_col="time"
        )

        return expanded_df

    def _prepare_time_varying_data(self, X, y):
        """Prepare time-varying data for the HazardIntegrator.

        Parameters
        ----------
        X : pd.DataFrame
            Time-varying input features (with unit_id added if needed)
        y : pd.DataFrame
            Time-varying survival data with 'time', 'event', 'unit_id' columns

        Returns
        -------
        pd.DataFrame
            Prepared DataFrame for HazardIntegrator
        """
        # Ensure both X and y have unit_id (should be guaranteed by validation)
        if "unit_id" not in X.columns or "unit_id" not in y.columns:
            raise ValueError("Both X and y must have 'unit_id' column after validation")

        # Combine X and y data, excluding duplicate columns
        y_without_unit_id = y.drop("unit_id", axis=1) if "unit_id" in y.columns else y
        combined_df = pd.concat([X, y_without_unit_id], axis=1)

        # Use expand_overlapping_units_locf to ensure proper time-varying structure
        expanded_df = expand_overlapping_units_locf(
            combined_df, unit_col="unit_id", time_col="time"
        )

        return expanded_df

    def _validate_time_varying_data(self, df):
        """Validate that the DataFrame has the required structure for time-varying data.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame to validate

        Raises
        ------
        ValueError
            If required columns are missing or data is invalid
        """
        required_cols = ["unit_id", "time", "event"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for valid event values
        if not df["event"].isin([0, 1]).all():
            raise ValueError(
                "Event column must contain only 0 (censored) and 1 (event) values"
            )

        # Check for valid time values
        if (df["time"] < 0).any():
            raise ValueError("Time values must be non-negative")

        # Check for valid unit_id values
        if df["unit_id"].isna().any():
            raise ValueError("Unit IDs cannot be missing")

    def fit(self, X, y, auto_expand_time=True):
        """Fit the hazard integration survival model.

        Parameters
        ----------
        X : pd.DataFrame
            Training features. Can be static or time-varying.
        y : pd.DataFrame
            Survival data with 'time', 'event', 'unit_id' columns.
        auto_expand_time : bool, default True
            Whether to automatically expand static data to time-varying format.

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

        # Validate required columns in y
        required_cols = ["time", "event"]
        missing_cols = [col for col in required_cols if col not in y.columns]
        if missing_cols:
            raise ValueError(f"y must contain columns: {missing_cols}")

        self.data_format_ = self._detect_data_format(X, y)

        # Prepare data based on format
        if self.data_format_ == "static" and auto_expand_time:
            exp_df = self._expand_static_data(X, y)
        else:
            exp_df = self._prepare_time_varying_data(X, y)

        # Validate the prepared data
        self._validate_time_varying_data(exp_df)
        self.x_time_df = exp_df.copy()
        # Pre-compute event indicators for efficiency
        self.event_indicators_ = exp_df.groupby("unit_id")["event"].last().values
        self.n_samples_ = len(self.event_indicators_)

        # Set feature count
        self.n_features_in_ = len(
            [col for col in exp_df.columns if col not in ["unit_id", "time", "event"]]
        )

        # Initialize hazard integrator with appropriate covariate columns
        covariate_cols = [
            col for col in exp_df.columns if col not in ["unit_id", "time", "event"]
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

            out = self.integrator_(exp_df)

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
