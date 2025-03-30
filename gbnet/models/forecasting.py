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

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y

        self.model_ = ForecastModule(
            df.shape[0],
            params=self.params,
            module_type=self.module_type,
            trend_type=self.trend_type,
            gblinear_params=self.gblinear_params,
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

    def predict(self, X):
        check_is_fitted(self, "model_")
        df = X.copy()
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
    ):
        super(ForecastModule, self).__init__()
        assert trend_type in {"PyTorch", "GBLinear"}
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
        trend_fn_params = params.copy()
        trend_fn_params["eta"] = trend_fn_params["eta"] / 2
        self.trend_fn = GBModule(
            batch_size=n, input_dim=1, output_dim=1, params=trend_fn_params
        )
        self.initialized = False
        self.trend_type = trend_type

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

    def forward(self, df):
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

        forecast = (
            trend_component
            + self.periodic_fn(X)
            + self.trend_fn(df["numeric_dt"]) * df["numeric_dt"]
        )

        return forecast

    def gb_step(self):
        if self.trend_type == "GBLinear":
            self.trend.gb_step()

        self.periodic_fn.gb_step()
