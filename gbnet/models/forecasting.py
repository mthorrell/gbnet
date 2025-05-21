import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import lstsq
from scipy.optimize import root_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from gbnet.gblinear import GBLinear


def pd_datetime_to_seconds(x: pd.Series):
    return pd.to_numeric(x) / 1000000000.0  # convert to seconds


def loadModule(module):
    assert module in {"XGBModule", "LGBModule"}
    if module == "XGBModule":
        from gbnet import xgbmodule

        return xgbmodule.XGBModule
    if module == "LGBModule":
        from gbnet import lgbmodule

        return lgbmodule.LGBModule


class Forecast(BaseEstimator, RegressorMixin):
    """
    A forecasting model class that implements a trend + seasonality + changepoints
    model using either XGBModule or LGBModule (but defaulting to XGBModule).

    Parameters
    ----------
    nrounds : int, default=50
        Number of training iterations (epochs) for the model.
    params : dict, optional
        Dictionary of additional parameters to be passed to the underlying forecast model.
        Defaults to {"eta": 0.17, "max_depth": 3, "lambda": 1, "alpha": 8}
    module_type : str, default="XGBModule"
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
    linear_params : dict, default={}
        Parameters to pass to GBLinear.
        Defaults to {"min_hess": 0.0, "lambd": 0.1, "lr": 0.9}
    changepoint_params : dict, default={}
        Parameters for changepoint detection and modeling. Defaults to:
        {
            "n_changepoints": 32,  # how many changepoints to consider
            "eta": 0.9,
            "max_depth": 9,
            "lambda": 6.5,
            "alpha": 3.8,
            "cp_gap": 0.5,  # portion of time series allowing changepoints
            "cp_train_gap": 4  # how many training rounds to NOT update the periodic component
        }

    Attributes
    ----------
    nrounds : int
        Number of training rounds for the model.
    params : dict
        Additional parameters passed to the forecast model.
    model_ : ForecastModule or None
        Trained forecast model instance. Set after fitting.
    losses_ : list
        List of loss values recorded at each training iteration.

    Methods
    -------
    fit(X, y)
        Trains the forecast model using the input features X and target variable y.
        X must contain the datetime column 'ds'.
    predict(X, components=False)
        Predicts target values based on the input features X.
        If components=True, returns tuple of (trend, periodic, changepoint) components.

    Notes
    -----
    The model uses a linear trend + periodic function + changepoints via XGBModule or LGBModule.
    The loss function used is Mean Squared Error (MSE). The trend component uses GBLinear.
    Changepoints allow the model to capture changes in the time series trend.
    """

    def __init__(
        self,
        nrounds=50,
        params={},
        module_type="XGBModule",
        linear_params={},
        changepoint_params={},
        estimate_uncertainty=True,
    ):
        self.nrounds = nrounds
        self.model_ = None
        self.losses_ = []
        self.module_type = module_type
        self.trend_type = "GBLinear"

        self.params = {"eta": 0.17, "max_depth": 3, "lambda": 1, "alpha": 8}
        self.params.update(params)

        self.linear_params = {"min_hess": 0.0, "lambd": 0.1, "lr": 0.9}
        self.linear_params.update(linear_params)

        self.changepoint_params = {
            "n_changepoints": 32,
            "eta": 0.9,
            "max_depth": 9,
            "lambda": 6.5,
            "alpha": 3.8,
            "cp_gap": 0.5,
            "cp_train_gap": 4,
        }
        self.changepoint_params.update(changepoint_params)
        self.estimate_uncertainty = estimate_uncertainty

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y

        self.model_ = ForecastModule(
            df.shape[0],
            params=self.params,
            module_type=self.module_type,
            trend_type=self.trend_type,
            linear_params=self.linear_params,
            changepoint_params=self.changepoint_params,
        )
        self.model_.train()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01)
        mse = torch.nn.MSELoss()

        for _ in range(self.nrounds):
            optimizer.zero_grad()
            preds = self.model_(df)
            loss = mse(preds.flatten(), torch.Tensor(df["y"].values).flatten())
            loss.backward(create_graph=True)
            self.losses_.append(loss.detach().item())
            self.model_.gb_step()

        if self.estimate_uncertainty:
            # Main idea: uncertainty = basic sigma2 + G * (test datetime - max train datetime)
            # G is fit using a train/validation on half the training data
            train_residuals = df["y"] - self.model_(df).detach().numpy().flatten()
            sigma2, G = _get_uncertainty_params(self, train_residuals, df)
            self.sigma2 = sigma2
            self.G = G
            # For better scaling, self.G was estimated in years
            self.max_train_years = (
                pd_datetime_to_seconds(df["ds"]).max() / 60 / 60 / 24 / 365
            )
            self.min_train_years = (
                pd_datetime_to_seconds(df["ds"]).min() / 60 / 60 / 24 / 365
            )

        self.model_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        df = X.copy()

        t, p, c = self.model_(df, components=True)
        forecast = t + p + c
        trend = t + c

        df["yhat"] = forecast.detach().numpy()
        df["trend"] = trend.detach().numpy()
        df["season"] = p.detach().numpy()

        if self.estimate_uncertainty:
            df["years"] = pd_datetime_to_seconds(df["ds"]) / 60 / 60 / 24 / 365
            df["high_h"] = df["years"] - self.max_train_years
            df["low_h"] = self.min_train_years - df["years"]
            df["h"] = df[["high_h", "low_h"]].max(axis=1)
            df.loc[df["h"] < 0, "h"] = 0
            df["var_est"] = self.sigma2 + self.G * df["h"]

        return_cols = [
            c for c in ["ds", "y", "yhat", "trend", "season", "var_est"] if c in df
        ]
        return df[return_cols].copy()


class ForecastModule(torch.nn.Module):
    """PyTorch module for time series forecasting.

    This module combines a linear trend component with a periodic function and changepoints
    learned through gradient boosting to model time series data. The trend is modeled using
    GBLinear layer, while the periodic patterns and changepoints are captured by either
    XGBoost or LightGBM.

    Parameters
    ----------
    n : int
        Number of samples in training data
    params : dict, optional
        Parameters passed to the gradient boosting model. Defaults to None.
    module_type : str, optional
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
        Defaults to "XGBModule".
    trend_type : str, optional
        Type of trend model to use, either "PyTorch" or "GBLinear". Defaults to "GBLinear".
    linear_params : dict, optional
        Parameters passed to GBLinear trend model if trend_type="GBLinear". Defaults to {}.
    changepoint_params : dict, optional
        Parameters for changepoint detection and modeling. Defaults to:
        {
            "n_changepoints": 100,
            "gbmodule": "XGBModule",
            "cp_gap": 0.9,
            "cp_train_gap": 10
        }

    Attributes
    ----------
    trend : Union[torch.nn.Linear, GBLinear]
        Linear layer for modeling trend component, either PyTorch Linear or GBLinear
    bn : torch.nn.BatchNorm1d
        Batch normalization layer (only used with PyTorch trend)
    periodic_fn : XGBModule or LGBModule
        Gradient boosting module for modeling periodic patterns
    trend_fn : XGBModule or LGBModule
        Gradient boosting module for modeling changepoints
    initialized : bool
        Whether the model has been initialized with initial trend estimates
    trend_type : str
        Type of trend model being used

    Methods
    -------
    initialize(df)
        Initializes trend parameters using least squares regression
    forward(df, components=False)
        Forward pass combining trend, periodic, and changepoint components
    gb_step()
        Performs one gradient boosting step for all components
    """

    def __init__(
        self,
        n,
        params=None,
        module_type="XGBModule",
        trend_type="GBLinear",
        linear_params={},
        changepoint_params={},
    ):
        super(ForecastModule, self).__init__()
        assert trend_type in {"PyTorch", "GBLinear"}

        self.initialized = False
        self.trend_type = trend_type

        if trend_type == "PyTorch":
            self.trend = torch.nn.Linear(1, 1)
            self.bn = nn.LazyBatchNorm1d()
        if trend_type == "GBLinear":
            self.trend = GBLinear(1, 1, **linear_params)

        GBModule = loadModule(module_type)
        self.periodic_fn = GBModule(
            batch_size=n,
            input_dim=6,  # year, month, day, hour, minute, weekday
            output_dim=1,
            params=params,
        )

        # initialize changepoints components
        cp_params = {
            "n_changepoints": 100,
            "gbmodule": "XGBModule",
            "cp_gap": 0.9,
            "cp_train_gap": 10,
        }
        cp_params.update(changepoint_params)

        self.n_changepoints = cp_params.pop("n_changepoints")
        self.cp_gap = cp_params.pop("cp_gap")
        self.cp_module = cp_params.pop("gbmodule")
        self.cp_train_gap = cp_params.pop("cp_train_gap")

        self.trend_fn = loadModule(self.cp_module)(
            batch_size=max(1, self.n_changepoints),
            input_dim=1,
            output_dim=1,
            params=cp_params,
        )
        self.cp_train_count = 0
        self.use_cp = True if self.n_changepoints > 0 else False

    def _changepoint_initialize(self, df):
        ncp = self.n_changepoints if self.use_cp else 1
        self.cp_input = np.linspace(
            df["numeric_dt"].min(),
            df["numeric_dt"].min()
            + self.cp_gap * (df["numeric_dt"].max() - df["numeric_dt"].min()),
            ncp + 2,
        )[1:-1].reshape([-1, 1])

    def _gblinear_initialize(self, df):
        X = df.copy()
        X = (
            X[X["numeric_dt"].notnull() & X["y"].notnull()]
            .reset_index(drop=True)
            .copy()
        )

        X["intercept"] = 1
        ests = lstsq(X[["intercept", "numeric_dt"]], X[["y"]])[0]

        with torch.no_grad():
            self.trend.linear.weight.copy_(torch.Tensor(ests[1:, :]))
            self.trend.linear.bias.copy_(torch.Tensor(ests[0]))

        self.initialized = True

    def initialize(self, df):
        self._changepoint_initialize(df)
        if self.trend_type == "GBLinear":
            self._gblinear_initialize(df)
            return

        X = df.copy()
        X = (
            X[X["numeric_dt"].notnull() & X["y"].notnull()]
            .reset_index(drop=True)
            .copy()
        )

        X["intercept"] = 1
        X["numeric_dt"] = (X["numeric_dt"] - X["numeric_dt"].mean()) / X[
            "numeric_dt"
        ].std()
        ests = lstsq(X[["intercept", "numeric_dt"]], X[["y"]])[0]

        with torch.no_grad():
            self.trend.weight.copy_(torch.Tensor(ests[1:, :]))
            self.trend.bias.copy_(torch.Tensor(ests[0]))

        self.initialized = True

    def forward(self, df, components=False):
        df = df.copy()
        # Assume df formatted like prophet (columns 'ds' and 'y')
        df["year"] = df["ds"].dt.year
        df["month"] = df["ds"].dt.month
        df["day"] = df["ds"].dt.day
        df["hour"] = df["ds"].dt.hour
        df["minute"] = df["ds"].dt.minute
        df["weekday"] = df["ds"].dt.weekday
        df["numeric_dt"] = pd_datetime_to_seconds(df["ds"])
        if not self.initialized:
            self.initialize(df)

        X = np.array(
            df[["year", "month", "day", "hour", "minute", "weekday"]]
        )  # TODO just need to do this once

        if self.trend_type == "PyTorch":
            trend_component = self.trend(
                self.bn(
                    torch.Tensor(np.array(pd_datetime_to_seconds(df["ds"]))).reshape(
                        [-1, 1]
                    )
                )
            )
        if self.trend_type == "GBLinear":
            trend_component = self.trend(
                torch.Tensor(np.array(pd_datetime_to_seconds(df["ds"]))).reshape(
                    [-1, 1]
                )
            )

        gb_trend = self.forward_changepoints(df)

        forecast = trend_component + self.periodic_fn(X) + gb_trend
        if components:
            return trend_component, self.periodic_fn(X), gb_trend

        return forecast

    def forward_changepoints(self, df):
        slope_adjustments = self.trend_fn(self.cp_input)
        if not self.use_cp:
            return torch.zeros([df.shape[0], 1])
        changepoints = torch.concatenate(
            [torch.Tensor(self.cp_input), slope_adjustments], axis=1
        )
        cuml_adj = piecewise_linear_function(
            changepoints, torch.Tensor(np.array(df["numeric_dt"]))
        )
        return cuml_adj.reshape([-1, 1])

    def gb_step(self):
        if self.trend_type == "GBLinear":
            self.trend.gb_step()

        if not self.use_cp:
            self.periodic_fn.gb_step()
            return

        self.trend_fn.gb_step()
        self.cp_train_count += 1
        if self.cp_train_count > self.cp_train_gap:
            self.periodic_fn.gb_step()


def piecewise_linear_function(changepoints, timepoints):
    """
    Calculate a piecewise linear function based on changepoints evaluated at timepoints.

    For every tp in timepoints, compute:
    Sum_{(cp1, cp2) such that cp1 < tp} cp2 * (tp - cp1)

    Args:
        changepoints: torch.Tensor of shape [N, 2] where:
            - First column (cp1) contains the positions of changepoints
            - Second column (cp2) contains the slopes at those changepoints
        timepoints: torch.Tensor of shape [M] containing points to evaluate the function

    Returns:
        torch.Tensor of shape [M] containing the evaluated function at each timepoint
    """
    # Extract cp1 and cp2 from changepoints
    cp1 = changepoints[:, 0]  # Shape: [N]
    cp2 = changepoints[:, 1]  # Shape: [N]

    # Reshape for broadcasting
    # cp1: [N, 1], timepoints: [1, M] for broadcasting to [N, M]
    cp1_expanded = cp1.unsqueeze(1)  # Shape: [N, 1]
    timepoints_expanded = timepoints.unsqueeze(0)  # Shape: [1, M]

    # Create mask where cp1 < tp (shape: [N, M])
    mask = (cp1_expanded < timepoints_expanded).float()

    # Calculate (tp - cp1) for all pairs (shape: [N, M])
    time_diffs = (timepoints_expanded - cp1_expanded) * mask

    # Multiply by cp2 (shape: [N, M])
    cp2_expanded = cp2.unsqueeze(1)  # Shape: [N, 1]
    contributions = cp2_expanded * time_diffs

    # Sum over all changepoints (axis 0) to get final result for each timepoint
    result = torch.sum(contributions, dim=0)  # Shape: [M]

    return result


def _get_uncertainty_params(m, train_residuals, train):
    utrain_cutoff = train["ds"].quantile(0.5)
    utrain = (
        train[train["ds"] <= utrain_cutoff]
        .sort_values("ds")
        .reset_index(drop=True)
        .copy()
    )
    utest = (
        train[train["ds"] > utrain_cutoff]
        .sort_values("ds")
        .reset_index(drop=True)
        .copy()
    )

    um = Forecast(
        nrounds=m.nrounds,
        params=m.params,
        module_type=m.module_type,
        linear_params=m.linear_params,
        changepoint_params=m.changepoint_params,
        estimate_uncertainty=False,
    )
    um.fit(utrain, utrain["y"])
    utrain["gbnet_pred"] = um.predict(utrain)["yhat"].copy()
    utest["gbnet_pred"] = um.predict(utest)["yhat"].copy()

    utrain["residual"] = utrain["y"] - utrain["gbnet_pred"]
    utest["residual"] = utest["y"] - utest["gbnet_pred"]

    try:
        G, info = _mle_G(
            utest["residual"],
            pd_datetime_to_seconds(utest["ds"]) / 60 / 60 / 24 / 365
            - pd_datetime_to_seconds(utrain["ds"]).max() / 60 / 60 / 24 / 365,
            (utrain["residual"] ** 2).sum() / (utrain.shape[0] - 1),
        )
    except (
        ValueError
    ):  # TODO maybe we estimate sigma2 and G simultaneously? to avoid this issue?
        G = 0.0

    sigma2 = (train_residuals**2).sum() / (train.shape[0] - 1)

    return sigma2, G


def _mle_G(r, h, S2, bracket_pad=1.0):
    """
    MLE for G in Var(r_i)=S2 + G*h_i with known S2.
    Returns (G_hat, diagnostics)
    """
    r = np.asarray(r, dtype=float)
    h = np.asarray(h, dtype=float)
    assert (h >= 0).all(), "h_i must be non-negative"

    def score(G):
        denom = S2 + G * h
        output = np.sum(h * (S2 + G * h - r**2) / denom**2)
        return output

    # Find a bracket [a,b] where score(a)>0, score(b)<0
    a = 0.0
    b = max(np.max((r**2 - S2) / (h + 1e-12)), 1.0) + bracket_pad
    sol = root_scalar(score, bracket=(a, b), method="brentq")
    return sol.root, sol
