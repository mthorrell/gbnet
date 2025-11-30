import numpy as np
import pandas as pd

from gbnet.models.forecasting import Forecast
from gbnet.models.forecasting_xgb_only import ForecastXGBOnly


DATA_URL = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_air_passengers.csv"


def load_data():
    df = pd.read_csv(DATA_URL)
    df["ds"] = pd.to_datetime(df["ds"])
    return df.sort_values("ds").reset_index(drop=True)


def train_test_split_time(df, test_ratio=0.2):
    split_idx = int((1 - test_ratio) * len(df))
    train = df.iloc[:split_idx].copy()
    test = df.iloc[split_idx:].copy()
    return train, test


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def main():
    df = load_data()
    train, test = train_test_split_time(df, test_ratio=0.2)

    print("Training sizes:", len(train), "train,", len(test), "test")

    torch_forecaster = Forecast(
        nrounds=50,
        estimate_uncertainty=False,
    )
    torch_forecaster.fit(train, train["y"])
    torch_preds = torch_forecaster.predict(test)

    xgb_only = ForecastXGBOnly(
        nrounds=50,
        estimate_uncertainty=False,
    )
    xgb_only.fit(train, train["y"])
    xgb_preds = xgb_only.predict(test)

    torch_mse = mse(test["y"], torch_preds["yhat"])
    xgb_mse = mse(test["y"], xgb_preds["yhat"])

    print("\nComparison on Air Passengers:")
    print(f"Torch-based Forecast MSE:     {torch_mse:.4f}")
    print(f"XGB-only Forecast MSE:        {xgb_mse:.4f}")
    print("\nSample predictions (last 5 rows):")
    comparison = pd.DataFrame(
        {
            "ds": test["ds"].iloc[-5:].dt.date,
            "actual": test["y"].iloc[-5:].values,
            "torch_yhat": torch_preds["yhat"].iloc[-5:].values,
            "xgb_yhat": xgb_preds["yhat"].iloc[-5:].values,
        }
    )
    print(comparison.to_string(index=False))


if __name__ == "__main__":
    main()
