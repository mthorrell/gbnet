import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from gbnet.gblinear import GBLinear


def pd_datetime_to_seconds(x: pd.Series):
    return pd.to_numeric(x) / 1000000000.0


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
    A forecasting model class that implements a trend + seasonality
    model using either XGBModule or LGBModule (but defaulting to XGBModule).

    Parameters
    ----------
    nrounds : int, default=500
        Number of training iterations (epochs) for the model. Overfitting seems to
        be fine usually.
    params : dict, optional
        Dictionary of additional parameters to be passed to the underlying forecast model.
    module_type : str, default="XGBModule"
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
    trend_type : str, default="PyTorch"
        Type of trend component to use. Can be either "PyTorch" for a PyTorch linear layer
        or "GBLinear" for a gradient boosting linear layer.
    gblinear_params : dict, default={}
        Parameters to pass to GBLinear if trend_type="GBLinear". Ignored if trend_type="PyTorch".
        See gbnet.gblinear.GBLinear for available parameters.

    Attributes
    ----------
    nrounds : int
        Number of training rounds for the model.
    params : dict
        Additional parameters passed to the forecast model.
    model_ : NSForecastModule or None
        Trained forecast model instance. Set after fitting.
    losses_ : list
        List of loss values recorded at each training iteration.

    Methods
    -------
    fit(X, y)
        Trains the forecast model using the input features X and target variable y.
        X must contain the datetime column 'ds'.
    predict(X)
        Predicts target values based on the input features X.

    Notes
    -----
    The model uses a linear trend + a periodic function via XGBModule or LGBModule. The
    loss function used is Mean Squared Error (MSE). The trend component can use either
    standard PyTorch optimization or gradient boosting updates via GBLinear.
    """

    def __init__(
        self,
        nrounds=500,
        params=None,
        module_type="XGBModule",
        trend_type="PyTorch",
        gblinear_params={},
        gbchangepoint_params={},
    ):
        if params is None:
            params = {}
        self.nrounds = nrounds
        self.params = params
        self.model_ = None
        self.losses_ = []
        self.module_type = module_type
        self.trend_type = trend_type
        self.gblinear_params = gblinear_params
        self.gbchangepoint_params = gbchangepoint_params

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y

        self.model_ = ForecastModule(
            df.shape[0],
            params=self.params,
            module_type=self.module_type,
            trend_type=self.trend_type,
            gblinear_params=self.gblinear_params,
            gbchangepoint_params=self.gbchangepoint_params,
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
            if self.trend_type == "PyTorch":
                optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X, components=False):
        check_is_fitted(self, "model_")
        df = X.copy()
        if components:
            t, p, c = self.model_(df, components=True)
            return t, p, c

        preds = self.model_(df).detach().numpy()
        return preds.flatten()


class ForecastModule(torch.nn.Module):
    """PyTorch module for time series forecasting.

    This module combines a linear trend component with a periodic function learned through
    gradient boosting to model time series data. The trend is modeled using either a PyTorch
    linear layer or GBLinear layer, while the periodic patterns are captured by either
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
        Type of trend model to use, either "PyTorch" or "GBLinear". Defaults to "PyTorch".
    gblinear_params : dict, optional
        Parameters passed to GBLinear trend model if trend_type="GBLinear". Defaults to {}.

    Attributes
    ----------
    trend : Union[torch.nn.Linear, GBLinear]
        Linear layer for modeling trend component, either PyTorch Linear or GBLinear
    bn : torch.nn.BatchNorm1d
        Batch normalization layer (only used with PyTorch trend)
    periodic_fn : XGBModule or LGBModule
        Gradient boosting module for modeling periodic patterns
    initialized : bool
        Whether the model has been initialized with initial trend estimates
    trend_type : str
        Type of trend model being used

    Methods
    -------
    initialize(df)
        Initializes trend parameters using least squares regression
    forward(df)
        Forward pass combining trend and periodic components
    """

    def __init__(
        self,
        n,
        params=None,
        module_type="XGBModule",
        trend_type="PyTorch",
        gblinear_params={},
        gbchangepoint_params={},
    ):
        super(ForecastModule, self).__init__()
        assert trend_type in {"PyTorch", "GBLinear"}

        self.initialized = False
        self.trend_type = trend_type

        if trend_type == "PyTorch":
            self.trend = torch.nn.Linear(1, 1)
            self.bn = nn.LazyBatchNorm1d()
        if trend_type == "GBLinear":
            self.trend = GBLinear(1, 1, **gblinear_params)

        GBModule = loadModule(module_type)
        self.periodic_fn = GBModule(
            batch_size=n,
            input_dim=6,  # year, month, day, hour, minute, weekday
            output_dim=1,
            params=params,
        )

        # initialize changepoints components
        cp_params = {"n_changepoints": 100, "gbmodule": "XGBModule"}
        cp_params.update(gbchangepoint_params)

        self.n_changepoints = cp_params.pop("n_changepoints")
        self.cp_module = cp_params.pop("gbmodule")
        self.trend_fn = loadModule(self.cp_module)(
            batch_size=self.n_changepoints, input_dim=1, output_dim=1, params=cp_params
        )

    def _changepoint_initialize(self, df):
        self.cp_input = np.linspace(
            df["numeric_dt"].min(), df["numeric_dt"].max(), self.n_changepoints + 2
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

        X = np.array(df[["year", "month", "day", "hour", "minute", "weekday"]])
        # X = np.array(df[["month", "day", "hour", "minute", "weekday"]])

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

        gb_trend = self.forward_changepoints(df) * torch.Tensor(
            np.array(df["numeric_dt"]).reshape([-1, 1])
        )

        forecast = trend_component + self.periodic_fn(X) + gb_trend
        if components:
            return trend_component, self.periodic_fn(X), gb_trend

        return forecast

    def forward_changepoints(self, df):
        slope_adjustments = self.trend_fn(self.cp_input)
        changepoints = torch.concatenate(
            [torch.Tensor(self.cp_input), slope_adjustments], axis=1
        )
        cuml_sum = efficient_cumulative_sum_at_timepoints(
            changepoints, torch.Tensor(np.array(df["numeric_dt"]).reshape([-1, 1]))
        )
        return cuml_sum

    def gb_step(self):
        if self.trend_type == "GBLinear":
            self.trend.gb_step()

        self.trend_fn.gb_step()
        self.periodic_fn.gb_step()


def cumulative_sum_at_timepoints(changepoints, timepoints):
    """
    Calculate the cumulative sum of the 2nd column in changepoints for rows where
    the 1st column is less than each value in timepoints.

    Args:
        changepoints: Tensor of shape [N, 2] where the 1st column contains time values
                      and the 2nd column contains change values
        timepoints: Tensor of shape [M] containing time points to evaluate

    Returns:
        Tensor of shape [M] containing cumulative sums at each timepoint
    """
    # Sort changepoints by their timepoint (1st column) for correctness
    sorted_indices = torch.argsort(changepoints[:, 0])
    sorted_changepoints = changepoints[sorted_indices]

    # Extract time and change values
    times = sorted_changepoints[:, 0]
    changes = sorted_changepoints[:, 1]

    # Calculate cumulative sum of all changes
    cumsum_changes = torch.cumsum(changes, dim=0)

    # For each timepoint, find the index of the last changepoint that's less than it
    # This uses broadcasting to compare each timepoint against all times
    mask = times.unsqueeze(0) < timepoints.unsqueeze(1)  # Shape: [M, N]

    # Get the indices of the last True value in each row
    # We first check if any value is True, for timepoints that are before all changepoints
    any_true = torch.any(mask, dim=1)

    # Find the position of the last True in each row
    last_true_indices = torch.sum(mask.int(), dim=1) - 1

    # Create the result tensor
    result = torch.zeros_like(timepoints, dtype=cumsum_changes.dtype)

    # Only assign values where there's at least one changepoint before the timepoint
    valid_indices = torch.where(any_true)[0]
    if len(valid_indices) > 0:
        result[valid_indices] = cumsum_changes[last_true_indices[valid_indices]]

    return result


def efficient_cumulative_sum_at_timepoints(changepoints, timepoints):
    """
    More efficient implementation using searchsorted.
    """
    # Sort changepoints by time (1st column)
    sorted_indices = torch.argsort(changepoints[:, 0])
    sorted_changepoints = changepoints[sorted_indices]

    # Extract time and change values
    times = sorted_changepoints[:, 0]
    changes = sorted_changepoints[:, 1]

    # Calculate cumulative sum of all changes
    cumsum_changes = torch.cumsum(changes, dim=0)

    # For each timepoint, find the index where it would be inserted in the sorted times
    # This gives us the position of the first element that is >= the timepoint
    # So we subtract 1 to get the position of the last element < timepoint
    indices = torch.searchsorted(times, timepoints) - 1

    # Create the result tensor
    result = torch.zeros_like(timepoints, dtype=cumsum_changes.dtype)

    # Only get values for timepoints that are after at least one changepoint
    valid_mask = indices >= 0
    if torch.any(valid_mask):
        result[valid_mask] = cumsum_changes[indices[valid_mask]]

    return result
