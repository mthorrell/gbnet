import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from gbnet import xgbmodule
from scipy.linalg import lstsq
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class ForecastModule(torch.nn.Module):
    def __init__(self, n_features, xcols=None, params=None):
        super(ForecastModule, self).__init__()
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
        X = df[["year"]].copy()
        y = df[["y"]].copy()

        if X["year"].std() == 0:
            with torch.no_grad():
                self.trend.weight.copy_(torch.Tensor([[0.0]]))
                self.trend.bias.copy_(torch.Tensor(y.mean().values))
            self.initialized = True
            return

        X["intercept"] = 1
        X["year"] = (X["year"] - X["year"].mean()) / X["year"].std()
        ests = lstsq(X[["intercept", "year"]], y)[0]

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


class Forecast(BaseEstimator, RegressorMixin):
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
        self.model_ = ForecastModule(df.shape[0], xcols=self.xcols, params=self.params)
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
        df["month"] = df["ds"].dt.month
        df["year"] = df["ds"].dt.year
        df["day"] = df["ds"].dt.day
        df["weekday"] = df["ds"].dt.weekday
        df["hour"] = df["ds"].dt.hour
        df["minute"] = df["ds"].dt.minute
        return df
