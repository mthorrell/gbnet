import numpy as np
import pandas as pd
import xgboost as xgb
from scipy.linalg import cho_factor, cho_solve, lstsq
from scipy.optimize import root_scalar
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def pd_datetime_to_seconds(x: pd.Series):
    return pd.to_numeric(x) / 1000000000.0


def _make_time_features(df: pd.DataFrame):
    out = df.copy()
    out["year"] = out["ds"].dt.year
    out["month"] = out["ds"].dt.month
    out["day"] = out["ds"].dt.day
    out["hour"] = out["ds"].dt.hour
    out["minute"] = out["ds"].dt.minute
    out["weekday"] = out["ds"].dt.weekday
    out["numeric_dt"] = pd_datetime_to_seconds(out["ds"])
    return out


class NumpyGBLinear:
    """A torch-free linear component updated with Newton-style steps."""

    def __init__(
        self,
        input_dim: int = 1,
        output_dim: int = 1,
        bias: bool = True,
        lr: float = 0.5,
        min_hess: float = 0.0,
        lambd: float = 0.01,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bias = bias
        self.lr = lr
        self.min_hess = min_hess
        self.lambd = lambd

        self.weight = np.zeros((self.input_dim, self.output_dim))
        self.bias_term = np.zeros((1, self.output_dim)) if self.bias else None

    def initialize(self, x: np.ndarray, y: np.ndarray):
        x = np.asarray(x).reshape([-1, self.input_dim])
        y = np.asarray(y).reshape([-1, self.output_dim])

        if self.bias:
            X_design = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        else:
            X_design = x

        ests = lstsq(X_design, y)[0]
        if self.bias:
            self.bias_term = ests[0:1, :]
            self.weight = ests[1:, :]
        else:
            self.weight = ests

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x).reshape([-1, self.input_dim])
        preds = x @ self.weight
        if self.bias:
            preds = preds + self.bias_term
        return preds

    def gb_step(self, x: np.ndarray, grad: np.ndarray, hess: np.ndarray):
        x = np.asarray(x).reshape([-1, self.input_dim])
        grad = np.asarray(grad).reshape([-1, self.output_dim])
        hess = np.asarray(hess).reshape([-1, self.output_dim])

        hess = np.maximum(hess, self.min_hess)
        hess = np.nan_to_num(hess, nan=1.0)

        if self.bias:
            X_design = np.concatenate([np.ones((x.shape[0], 1)), x], axis=1)
        else:
            X_design = x

        direction = ridge_regression(X_design, grad / hess, self.lambd)
        if self.bias:
            self.bias_term = self.bias_term - self.lr * direction[0:1, :]
            self.weight = self.weight - self.lr * direction[1:, :]
        else:
            self.weight = self.weight - self.lr * direction


class ForecastXGBOnly(BaseEstimator, RegressorMixin):
    """
    XGBoost-only forecaster with a GBLinear trend and no changepoints.

    Mirrors the torch-based Forecast (when using XGB and zero changepoints) but removes
    torch/LightGBM and relies on raw XGBoost plus a torch-free linear trend. The booster
    and trend share the same gradients every iteration so they are updated simultaneously.
    """

    def __init__(
        self,
        nrounds=50,
        params=None,
        linear_params=None,
        estimate_uncertainty=False,
    ):
        self.nrounds = nrounds
        self.params = {
            "eta": 0.17,
            "max_depth": 3,
            "lambda": 1,
            "alpha": 8,
        }
        params = params or {}
        if "objective" in params:
            raise ValueError("objective should not be specified in params")
        if "base_score" in params:
            raise ValueError("base_score should not be specified in params")
        self.params.update(params)
        self.params["objective"] = "reg:squarederror"
        self.params["base_score"] = 0

        self.linear_params = {"min_hess": 0.0, "lambd": 0.1, "lr": 0.9}
        if linear_params:
            self.linear_params.update(linear_params)

        self.estimate_uncertainty = estimate_uncertainty
        self.losses_ = []

        self.trend = None
        self.booster = None
        self.n_completed_boost_rounds = 0
        print(self.estimate_uncertainty)

    def fit(self, X, y=None):
        df = X.copy()
        df["y"] = y
        features = _make_time_features(df)

        y_values = np.asarray(df["y"], dtype=float)
        time_values = features["numeric_dt"].to_numpy(dtype=float).reshape([-1, 1])
        season_feats = np.array(
            features[["year", "month", "day", "hour", "minute", "weekday"]]
        )

        self.trend = NumpyGBLinear(1, 1, **self.linear_params)
        self.trend.initialize(time_values, y_values)

        dtrain = xgb.DMatrix(season_feats, label=np.zeros_like(y_values))
        booster = xgb.train(self.params, dtrain, num_boost_round=0)

        self.losses_ = []
        n_completed_boost_rounds = 0
        for _ in range(self.nrounds):
            trend_pred = self.trend.predict(time_values).flatten()
            season_pred = booster.predict(dtrain)
            yhat = trend_pred + season_pred

            # Shared gradients mirror the torch-based training loop
            residual = yhat - y_values
            grad = 2.0 * residual
            hess = np.full_like(grad, 2.0)

            self.losses_.append(float(np.mean(residual**2)))

            self.trend.gb_step(
                time_values,
                grad.reshape([-1, 1]),
                hess.reshape([-1, 1]),
            )

            g_boost, h_boost = _format_grad_hess(grad, hess)
            _boost_one_round(
                booster, dtrain, g_boost, h_boost, n_completed_boost_rounds
            )
            n_completed_boost_rounds += 1

        self.booster = booster
        self.n_completed_boost_rounds = n_completed_boost_rounds
        self._n_features_in_ = season_feats.shape[1]
        self.n_features_in_ = season_feats.shape[1]

        if self.estimate_uncertainty:
            train_preds = self.predict(X)
            train_residuals = df["y"].values - train_preds["yhat"].values
            sigma2, G = _get_uncertainty_params(self, train_residuals, df)
            self.sigma2 = sigma2
            self.G = G
            self.max_train_years = (
                pd_datetime_to_seconds(df["ds"]).max() / 60 / 60 / 24 / 365
            )
            self.min_train_years = (
                pd_datetime_to_seconds(df["ds"]).min() / 60 / 60 / 24 / 365
            )

        return self

    def predict(self, X):
        check_is_fitted(self, "booster")
        df = X.copy()
        features = _make_time_features(df)

        trend_pred = self.trend.predict(
            features["numeric_dt"].values.reshape([-1, 1])
        ).flatten()
        season_feat = np.array(
            features[["year", "month", "day", "hour", "minute", "weekday"]]
        )
        season_pred = self.booster.predict(xgb.DMatrix(season_feat))

        yhat = trend_pred + season_pred
        df["yhat"] = yhat
        df["trend"] = trend_pred
        df["season"] = season_pred

        if self.estimate_uncertainty and hasattr(self, "sigma2"):
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


def _format_grad_hess(grad: np.ndarray, hess: np.ndarray):
    grad = np.asarray(grad, dtype=float)
    hess = np.asarray(hess, dtype=float)

    if grad.ndim == 1:
        grad = grad.reshape([-1, 1])
    if hess.ndim == 1:
        hess = hess.reshape([-1, 1])

    if xgb.__version__ >= "2.1.0":
        return (
            grad.reshape([grad.shape[0], -1]),
            hess.reshape([hess.shape[0], -1]),
        )

    return grad.reshape([-1, 1]), hess.reshape([-1, 1])


def _boost_one_round(booster, dtrain, grad, hess, iteration):
    if xgb.__version__ <= "2.0.3":
        booster.boost(dtrain, grad, hess)
    else:
        booster.boost(dtrain, iteration + 1, grad, hess)


def ridge_regression(X, y, lambd):
    X = np.asarray(X)
    y = np.asarray(y)
    n, d = X.shape
    A = X.T @ X + lambd * np.eye(d)
    c = X.T @ y
    L = cho_factor(A)
    beta = cho_solve(L, c)
    return beta


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

    um = ForecastXGBOnly(
        nrounds=m.nrounds,
        params=m.params,
        linear_params=m.linear_params,
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
    except ValueError:
        G = 0.0

    sigma2 = (train_residuals**2).sum() / (train.shape[0] - 1)

    return sigma2, G


def _mle_G(r, h, S2, bracket_pad=1.0):
    r = np.asarray(r, dtype=float)
    h = np.asarray(h, dtype=float)
    assert (h >= 0).all(), "h_i must be non-negative"

    def score(G):
        denom = S2 + G * h
        output = np.sum(h * (S2 + G * h - r**2) / denom**2)
        return output

    a = 0.0
    b = max(np.max((r**2 - S2) / (h + 1e-12)), 1.0) + bracket_pad
    sol = root_scalar(score, bracket=(a, b), method="brentq")
    return sol.root, sol
