import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gbnet import xgbmodule
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class Forecast(BaseEstimator, RegressorMixin):
    """
    A forecasting model class that implements a trend + seasonality
    model using gbnet.xgbmodule.XGBModule

    Parameters
    ----------
    nrounds : int, default=500
        Number of training iterations (epochs) for the model. Overfitting seems to
        be fine usually.
    params : dict, optional
        Dictionary of additional parameters to be passed to the underlying forecast model.

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
    The model uses a linear trend + a periodic function via XGBModule. The
    loss function used is Mean Squared Error (MSE).
    """

    def __init__(self, nrounds=500, params=None):
        if params is None:
            params = {}
        self.nrounds = nrounds
        self.params = params
        self.model_ = None
        self.losses_ = []

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y

        self.model_ = ForecastModule(df.shape[0], params=self.params)
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
            optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        df = X.copy()
        preds = self.model_(df).detach().numpy()
        return preds.flatten()


class ForecastModule(torch.nn.Module):
    def __init__(self, n, params=None):
        super(ForecastModule, self).__init__()
        self.trend = torch.nn.Linear(1, 1)
        self.bn = nn.LazyBatchNorm1d()

        self.periodic_fn = xgbmodule.XGBModule(
            batch_size=n,
            input_dim=6,  # year, month, day, hour, minute, weekday
            output_dim=1,
            params=params,
        )
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
        self.X = X

        forecast = (
            ###### linear trend defined via torch.nn.Linear
            self.trend(
                self.bn(
                    torch.Tensor(np.array(pd.to_numeric(df["ds"]))).reshape([-1, 1])
                )
            )
            ###### datetime components plugged into gbnet.xgbmodule.XGBModule
            + self.periodic_fn(np.array(X))
        )
        return forecast

    def gb_step(self):
        self.periodic_fn.gb_step(self.X)


class NSForecastModule(torch.nn.Module):
    def __init__(self, n_features, xcols=None, params=None):
        super(NSForecastModule, self).__init__()
        if params is None:
            params = {}
        if xcols is None:
            xcols = []
        self.seasonality = xgbmodule.XGBModule(
            n_features, 10 + len(xcols), 1, params=params
        )
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

        self.minput = datetime_features
        output = self.trend(self.bn(trend)) + self.seasonality(datetime_features)
        return output

    def gb_step(self):
        self.seasonality.gb_step(self.minput)


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


class NSForecast(BaseEstimator, RegressorMixin):
    """
    A forecasting model class that implements a trend + seasonality + minor non-stationarity
    model using gbnet.xgbmodule.XGBModule

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
    The model uses a linear trend + some basic features in XGBModule. The
    loss function used is Mean Squared Error (MSE).
    """

    def __init__(self, nrounds=3000, xcols=None, params=None):
        if params is None:
            params = {}
        if xcols is None:
            xcols = []
        self.nrounds = nrounds
        self.xcols = xcols
        self.params = params
        self.model_ = None
        self.losses_ = []

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y
        df = self._prepare_dataframe(df)
        df = recursive_split(df, "ds", 4)
        self.model_ = NSForecastModule(
            df.shape[0], xcols=self.xcols, params=self.params
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
