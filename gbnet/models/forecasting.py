import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

from gbnet.gblinear import GBLinear


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
    n_changepoints : int, default=20
        Number of potential changepoints for the piecewise linear trend.
    changepoint_penalty : float, default=0.05
        L1 regularization strength for changepoints. Higher values lead to fewer changepoints.

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
    The model uses a piecewise linear trend + a periodic function via XGBModule or LGBModule. The
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
        n_changepoints=20,
        changepoint_penalty=0.0,
        pytorch_lr=0.01,
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
        self.n_changepoints = n_changepoints
        self.changepoint_penalty = changepoint_penalty
        self.pytorch_lr = pytorch_lr

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y

        # Calculate changepoint locations
        min_date = pd.to_numeric(df["ds"]).min()
        max_date = pd.to_numeric(df["ds"]).max()

        self.model_ = ForecastModule(
            df.shape[0],
            params=self.params,
            module_type=self.module_type,
            trend_type=self.trend_type,
            gblinear_params=self.gblinear_params,
            n_changepoints=self.n_changepoints,
            changepoint_range=(min_date, max_date),
            changepoint_penalty=self.changepoint_penalty,
        )
        self.model_.train()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.pytorch_lr)
        mse = torch.nn.MSELoss()

        for _ in range(self.nrounds):
            optimizer.zero_grad()
            preds = self.model_(df)
            loss = mse(preds.flatten(), torch.Tensor(df["y"].values).flatten())

            # Add L1 penalty for changepoints
            if self.trend_type == "PyTorch":
                changepoint_penalty = self.changepoint_penalty * torch.norm(
                    self.model_.trend_changepoints.weight, p=1
                )
            else:
                changepoint_penalty = self.changepoint_penalty * torch.norm(
                    self.model_.trend_changepoints.coefs.linear.weight, p=1
                )

            loss = loss + changepoint_penalty

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

    def predict_components(self, X):
        check_is_fitted(self, "model_")
        df = X.copy()
        preds = self.model_(df, components=True)
        return preds


class LassoCoefs(torch.nn.Module):
    def __init__(self, n_features, lr=0.5, lambd=1.0, min_hess=0):
        super(LassoCoefs, self).__init__()
        self.coefs = GBLinear(
            1, n_features, bias=False, lr=lr, lambd=lambd, min_hess=min_hess
        )

    def forward(self, x):
        beta = self.coefs(np.array([[1]]))
        return torch.matmul(x, beta.T), beta.T

    def gb_step(self):
        self.coefs.gb_step()


class ForecastModule(torch.nn.Module):
    """PyTorch module for time series forecasting.

    This module combines a piecewise linear trend component with a periodic function learned through
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
    n_changepoints : int, optional
        Number of potential changepoints for the piecewise linear trend. Defaults to 20.
    changepoint_range : tuple, optional
        Range of dates (as numeric values) where changepoints can occur. Defaults to None.

    Attributes
    ----------
    trend : Union[torch.nn.Linear, GBLinear]
        Linear layer for modeling base trend component
    trend_changepoints : Union[torch.nn.Linear, GBLinear]
        Linear layer for modeling changepoints in the trend
    changepoint_locs : torch.Tensor
        Locations of potential changepoints
    bn : torch.nn.BatchNorm1d
        Batch normalization layer (only used with PyTorch trend)
    bn_changepoints : torch.nn.BatchNorm1d
        Batch normalization layer for changepoint features (only used with PyTorch trend)
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
        n_changepoints=20,
        changepoint_range=None,
        changepoint_penalty=0.1,
    ):
        super(ForecastModule, self).__init__()
        assert trend_type in {"PyTorch", "GBLinear"}
        self.n_changepoints = n_changepoints

        # Create equally spaced changepoints
        if changepoint_range is not None:
            min_date, max_date = changepoint_range
            self.changepoint_locs = torch.linspace(
                min_date, max_date, n_changepoints + 2
            )[1:-1]  # Exclude endpoints
        else:
            self.changepoint_locs = None

        if trend_type == "PyTorch":
            self.trend = torch.nn.Linear(1, 1)
            self.trend_changepoints = torch.nn.Linear(n_changepoints, 1, bias=False)
            self.bn = nn.LazyBatchNorm1d()
            self.bn_changepoints = nn.LazyBatchNorm1d()
        if trend_type == "GBLinear":
            self.trend = GBLinear(1, 1, **gblinear_params)

            cp_params = gblinear_params.copy()
            cp_params["lambd"] = (
                max(changepoint_penalty, 1.0)
                if "lambd" not in gblinear_params
                else max(changepoint_penalty, gblinear_params["lr"])
            )
            cp_params["lr"] = (
                0.25 if "lr" not in gblinear_params else gblinear_params["lr"] / 2
            )
            self.trend_changepoints = LassoCoefs(n_changepoints, **cp_params)

        GBModule = loadModule(module_type)
        self.periodic_fn = GBModule(
            batch_size=n,
            input_dim=6,  # year, month, day, hour, minute, weekday
            output_dim=1,
            params=params,
        )
        self.initialized = False
        self.trend_type = trend_type

    def _get_changepoint_features(self, numeric_dt):
        """
        Compute changepoint features for the piecewise linear trend.

        Parameters
        ----------
        numeric_dt : torch.Tensor
            Numeric datetime values

        Returns
        -------
        torch.Tensor
            Changepoint features of shape [n_samples, n_changepoints]
        """
        if self.changepoint_locs is None:
            return torch.zeros((numeric_dt.shape[0], self.n_changepoints))

        # ReLU function for changepoints: max(0, t - changepoint_loc)
        features = torch.maximum(
            torch.zeros_like(numeric_dt.expand(-1, self.n_changepoints)),
            numeric_dt.expand(-1, self.n_changepoints)
            - self.changepoint_locs.unsqueeze(0),
        )
        return features

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
            # Initialize changepoint weights to zero
            self.trend_changepoints.coefs.linear.weight.zero_()

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
            # Initialize changepoint weights to zero
            self.trend_changepoints.weight.zero_()

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
        df["numeric_dt"] = pd.to_numeric(df["ds"])
        if not self.initialized:
            self.initialize(df)

        X = np.array(df[["year", "month", "day", "hour", "minute", "weekday"]])
        numeric_dt = torch.Tensor(np.array(pd.to_numeric(df["ds"]))).reshape([-1, 1])

        # Get changepoint features
        changepoint_features = self._get_changepoint_features(numeric_dt)

        if self.trend_type == "PyTorch":
            # Apply batch normalization to the datetime
            normalized_dt = self.bn(numeric_dt)
            # Base trend
            trend_component = self.trend(normalized_dt)
            # Apply batch normalization to changepoint features
            normalized_changepoints = self.bn_changepoints(changepoint_features)
            # Add changepoint effects
            trend_component = trend_component + self.trend_changepoints(
                normalized_changepoints
            )
        if self.trend_type == "GBLinear":
            # Base trend
            trend_component = self.trend(numeric_dt)
            changepoints, beta = self.trend_changepoints(changepoint_features)
            # Add changepoint effects
            trend_component = trend_component + changepoints

        if components:
            return trend_component, self.periodic_fn(X)

        forecast = trend_component + self.periodic_fn(X)

        return forecast

    def gb_step(self):
        if self.trend_type == "GBLinear":
            self.trend.gb_step()
            self.trend_changepoints.gb_step()

        self.periodic_fn.gb_step()


class NSForecast(BaseEstimator, RegressorMixin):
    """
    A forecasting model class that implements a trend + seasonality + minor non-stationarity
    model using XGBModule or LGBModule

    Parameters
    ----------
    nrounds : int, default=3000
        Number of training iterations (epochs) for the model. Overfitting seems to
        be fine usually.
    xcols : list, optional
        List of column names to be used as input features for the model.
    params : dict, optional
        Dictionary of additional parameters to be passed to the underlying forecast model.

    Attributes
    ----------
    nrounds : int
        Number of training rounds for the model.
    xcols : list
        Input feature column names.
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
    The model uses a linear trend + some basic features in XGBModule or LGBModule. The
    loss function used is Mean Squared Error (MSE).
    """

    def __init__(self, nrounds=3000, xcols=None, params=None, module_type="XGBModule"):
        if params is None:
            params = {}
        if xcols is None:
            xcols = []
        self.nrounds = nrounds
        self.xcols = xcols
        self.params = params
        self.model_ = None
        self.losses_ = []
        self.module_type = module_type

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y
        df = self._prepare_dataframe(df)
        df = recursive_split(df, "ds", 4)
        self.model_ = NSForecastModule(
            df.shape[0],
            xcols=self.xcols,
            params=self.params,
            module_type=self.module_type,
        )
        self.model_.train()
        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.1)
        mse = torch.nn.MSELoss()

        for _ in range(self.nrounds):
            optimizer.zero_grad()
            preds = self.model_(df)
            loss = mse(preds.flatten(), torch.Tensor(df["y"].values).flatten())
            loss.backward(create_graph=True)
            self.losses_.append(loss.detach().item())
            self.model_.gb_step()
            optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        df = X.copy()
        df = self._prepare_dataframe(df)
        for j in range(1, 5):
            df[f"split_{j}"] = 1.0
        preds = self.model_(df).detach().numpy()
        return preds.flatten()

    @staticmethod
    def _prepare_dataframe(df):
        df["ds"] = pd.to_datetime(df["ds"])
        df["numeric_dt"] = pd.to_numeric(df["ds"])
        df["month"] = df["ds"].dt.month
        df["year"] = df["ds"].dt.year
        df["day"] = df["ds"].dt.day
        df["weekday"] = df["ds"].dt.weekday
        df["hour"] = df["ds"].dt.hour
        df["minute"] = df["ds"].dt.minute
        return df


class NSForecastModule(torch.nn.Module):
    def __init__(self, n_features, xcols=None, params=None, module_type="XGBModule"):
        super(NSForecastModule, self).__init__()
        if params is None:
            params = {}
        if xcols is None:
            xcols = []
        GBModule = loadModule(module_type)
        self.seasonality = GBModule(n_features, 10 + len(xcols), 1, params=params)
        self.xcols = xcols
        self.trend = torch.nn.Linear(1, 1)
        self.bn = nn.LazyBatchNorm1d()
        self.initialized = False

    def initialize(self, df):
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
        if not self.initialized:
            self.initialize(df)

        datetime_features = np.array(
            df[
                [
                    "split_1",
                    "split_2",
                    "split_3",
                    "split_4",
                    "year",
                    "minute",
                    "hour",
                    "month",
                    "day",
                    "weekday",
                ]
                + self.xcols
            ]
        )
        trend = torch.Tensor(df["year"].values).reshape([-1, 1])

        output = self.trend(self.bn(trend)) + self.seasonality(datetime_features)
        return output

    def gb_step(self):
        self.seasonality.gb_step()


def recursive_split(df, column, depth):
    """Adds columns of zeros and ones in attempt to address changepoints"""
    df = df.sort_values(by=column).reset_index(drop=True)
    binary_cols = pd.DataFrame(index=df.index)

    def split_group(indices, level):
        if level > depth:
            return
        mid = len(indices) // 2
        binary_cols.loc[indices[:mid], f"split_{level}"] = 0
        binary_cols.loc[indices[mid:], f"split_{level}"] = 1
        split_group(indices[:mid], level + 1)
        split_group(indices[mid:], level + 1)

    split_group(df.index, 1)
    return pd.concat([df, binary_cols], axis=1)
