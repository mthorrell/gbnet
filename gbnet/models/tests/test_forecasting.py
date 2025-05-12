import unittest
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from gbnet.models import forecasting
import torch


class TestForecast(unittest.TestCase):
    def setUp(self):
        # Generate synthetic data
        num_samples = 200
        self.dates = [datetime.now() + timedelta(days=i) for i in range(num_samples)]
        self.y = np.sin(np.linspace(0, 20, num_samples))  # Synthetic target variable
        self.df = pd.DataFrame({"ds": self.dates, "y": self.y})

    def test_fit_predict(self):
        """
        Test that the estimator can fit and predict without errors using GBLinear trend.
        """
        # Split data into training and testing sets
        train_df = self.df.iloc[:150]
        test_df = self.df.iloc[150:]

        # Initialize and fit the estimator
        estimator = forecasting.Forecast(nrounds=10)
        estimator.fit(train_df[["ds"]], train_df["y"])

        # Make predictions
        predictions = estimator.predict(test_df[["ds"]])

        # Check that predictions have the correct length
        self.assertEqual(
            predictions.shape[0], len(test_df), "Prediction length mismatch."
        )

        # Check that predictions are not NaN
        self.assertFalse(
            np.isnan(predictions).any().any(), "Predictions contain NaN values."
        )

    def test_fit_predict_lgbm(self):
        """
        Test that the estimator can fit and predict without errors using LGBModule and GBLinear trend.
        """
        # Split data into training and testing sets
        train_df = self.df.iloc[:150]
        test_df = self.df.iloc[150:]

        # Initialize and fit the estimator
        estimator = forecasting.Forecast(nrounds=10, module_type="LGBModule")
        estimator.fit(train_df[["ds"]], train_df["y"])

        # Make predictions
        predictions = estimator.predict(test_df[["ds"]])

        # Check that predictions have the correct length
        self.assertEqual(
            predictions.shape[0], len(test_df), "Prediction length mismatch."
        )

        # Check that predictions are not NaN
        self.assertFalse(
            np.isnan(predictions).any().any(), "Predictions contain NaN values."
        )

    def test_loss_decrease(self):
        """
        Test that the training loss decreases over epochs with GBLinear trend.
        """
        # Use only the training data
        train_df = self.df.iloc[:150]

        # Initialize and fit the estimator
        estimator = forecasting.Forecast(nrounds=10)
        estimator.fit(train_df[["ds"]], train_df["y"])

        # Get the recorded losses
        losses = estimator.losses_

        # Check that losses have been recorded
        self.assertTrue(len(losses) > 0, "No losses recorded during training.")

        # Check that the loss decreases over time
        initial_loss = losses[0]
        final_loss = losses[-1]
        self.assertLess(
            final_loss, initial_loss, "Final loss is not less than initial loss."
        )

        # Check that the losses are decreasing monotonically (not required but useful)
        self.assertTrue(
            all(x >= y for x, y in zip(losses[:10], losses[1:11])),
            "Losses are not monotonically decreasing.",
        )

    def test_loss_decrease_lgbm(self):
        """
        Test that the training loss decreases over epochs with LGBModule and GBLinear trend.
        """
        # Use only the training data
        train_df = self.df.iloc[:150]

        # Initialize and fit the estimator
        estimator = forecasting.Forecast(nrounds=10, module_type="LGBModule")
        estimator.fit(train_df[["ds"]], train_df["y"])

        # Get the recorded losses
        losses = estimator.losses_

        # Check that losses have been recorded
        self.assertTrue(len(losses) > 0, "No losses recorded during training.")

        # Check that the loss decreases over time
        initial_loss = losses[0]
        final_loss = losses[-1]
        self.assertLess(
            final_loss, initial_loss, "Final loss is not less than initial loss."
        )

        # Check that the losses are decreasing monotonically (not required but useful)
        self.assertTrue(
            all(x >= y for x, y in zip(losses[:10], losses[1:11])),
            "Losses are not monotonically decreasing.",
        )

    def test_changepoint_initialization(self):
        """
        Test that changepoints are properly initialized with the correct number and spacing.
        """
        # Initialize estimator with changepoint parameters
        estimator = forecasting.Forecast(
            nrounds=10,
            changepoint_params={
                "n_changepoints": 5,
                "cp_gap": 0.8,
                "cp_train_gap": 2,
            },
        )

        # Fit the model
        estimator.fit(self.df[["ds"]], self.df["y"])

        # Check that changepoints were initialized
        self.assertTrue(
            hasattr(estimator.model_, "cp_input"), "Changepoints not initialized"
        )
        self.assertEqual(
            len(estimator.model_.cp_input), 5, "Incorrect number of changepoints"
        )

        # Check changepoint spacing
        cp_times = estimator.model_.cp_input.flatten()
        expected_min = self.df["ds"].min().timestamp()
        expected_max = self.df["ds"].min().timestamp() + 0.8 * (
            self.df["ds"].max().timestamp() - self.df["ds"].min().timestamp()
        )
        self.assertGreaterEqual(
            cp_times[0], expected_min, "First changepoint too early"
        )
        self.assertLessEqual(cp_times[-1], expected_max, "Last changepoint too late")

    def test_changepoint_prediction(self):
        """
        Test that predictions with changepoints work correctly and return expected components.
        """
        # Initialize estimator with changepoints
        estimator = forecasting.Forecast(
            nrounds=10, changepoint_params={"n_changepoints": 3}
        )

        # Fit the model
        estimator.fit(self.df[["ds"]], self.df["y"])

        # Get predictions with components
        pred_df = estimator.predict(self.df[["ds"]])

        # Check component shapes and nans
        self.assertEqual(self.df.shape[0], pred_df.shape[0])
        self.assertEqual(pd.isnull(self.df).sum().sum(), 0)

    def test_changepoint_impact(self):
        """
        Test that changepoints have a meaningful impact on predictions.
        """
        # Create a dataset with a clear changepoint
        dates = pd.date_range(start="2020-01-01", periods=200, freq="D")
        y = np.concatenate(
            [
                np.linspace(0, 1, 100),  # First trend
                np.linspace(1, 2, 100),  # Second trend
            ]
        )
        df = pd.DataFrame({"ds": dates, "y": y})

        # Fit model with and without changepoints
        model_with_cp = forecasting.Forecast(
            nrounds=10, changepoint_params={"n_changepoints": 3}
        )
        model_without_cp = forecasting.Forecast(
            nrounds=10, changepoint_params={"n_changepoints": 0}
        )

        model_with_cp.fit(df[["ds"]], df["y"])
        model_without_cp.fit(df[["ds"]], df["y"])

        # Get predictions
        preds_with_cp = model_with_cp.predict(df[["ds"]])
        preds_without_cp = model_without_cp.predict(df[["ds"]])

        # Calculate MSE for both models
        mse_with_cp = np.mean((preds_with_cp["yhat"] - df["y"]) ** 2)
        mse_without_cp = np.mean((preds_without_cp["yhat"] - df["y"]) ** 2)

        # Check that changepoints improve the fit
        self.assertLess(
            mse_with_cp, mse_without_cp, "Changepoints did not improve model fit"
        )

    def test_changepoint_parameters(self):
        """
        Test that different changepoint parameters affect the model behavior.
        """
        # Test different numbers of changepoints
        for n_cp in [0, 1, 5, 10]:
            estimator = forecasting.Forecast(
                nrounds=10, changepoint_params={"n_changepoints": n_cp}
            )
            estimator.fit(self.df[["ds"]], self.df["y"])

            # Check that changepoints were initialized correctly
            if n_cp > 0:
                self.assertEqual(
                    len(estimator.model_.cp_input),
                    n_cp,
                    f"Incorrect number of changepoints for n_cp={n_cp}",
                )
            else:
                self.assertFalse(
                    estimator.model_.use_cp, "Changepoints should be disabled"
                )

            # Check that predictions work
            preds = estimator.predict(self.df[["ds"]])
            self.assertEqual(len(preds), len(self.df), "Prediction length mismatch")
            self.assertFalse(
                np.isnan(preds["yhat"]).any(), "Predictions contain NaN values"
            )

    def test_piecewise_linear_function(self):
        """
        Test the piecewise_linear_function implementation with various scenarios.
        """
        # Test case 1: Single changepoint
        changepoints = torch.tensor(
            [[1.0, 2.0]], dtype=torch.float32
        )  # [position, slope]
        timepoints = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0], dtype=torch.float32)
        result = forecasting.piecewise_linear_function(changepoints, timepoints)

        # For t < 1.0, result should be 0
        # For t >= 1.0, result should be 2.0 * (t - 1.0)
        expected = torch.tensor([0.0, 0.0, 0.0, 1.0, 2.0], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(result, expected), "Single changepoint test failed"
        )

        # Test case 2: Multiple changepoints
        changepoints = torch.tensor(
            [
                [1.0, 1.0],  # First changepoint at t=1 with slope 1
                [2.0, 2.0],  # Second changepoint at t=2 with slope 2
            ],
            dtype=torch.float32,
        )
        timepoints = torch.tensor([0.0, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
        result = forecasting.piecewise_linear_function(changepoints, timepoints)

        # Expected results:
        # t < 1.0: 0 (no changepoints before)
        # 1.0 <= t < 2.0: 1.0 * (t - 1.0) (only first changepoint contributes)
        # t >= 2.0: 1.0 * (t - 1.0) + 2.0 * (t - 2.0) (both changepoints contribute)
        expected = torch.tensor([0.0, 0.0, 0.5, 1.0, 2.5, 4.0], dtype=torch.float32)
        self.assertTrue(
            torch.allclose(result, expected), "Multiple changepoints test failed"
        )

        # Test case 3: No changepoints
        changepoints = torch.tensor([], dtype=torch.float32).reshape(0, 2)
        timepoints = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float32)
        result = forecasting.piecewise_linear_function(changepoints, timepoints)
        expected = torch.zeros_like(timepoints)
        self.assertTrue(torch.allclose(result, expected), "No changepoints test failed")

        # Test case 4: Changepoints with negative slopes
        changepoints = torch.tensor(
            [
                [1.0, -1.0],  # First changepoint at t=1 with slope -1
                [2.0, -2.0],  # Second changepoint at t=2 with slope -2
            ],
            dtype=torch.float32,
        )
        timepoints = torch.tensor([0.0, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=torch.float32)
        result = forecasting.piecewise_linear_function(changepoints, timepoints)

        # Expected results:
        # t < 1.0: 0
        # 1.0 <= t < 2.0: -1.0 * (t - 1.0)
        # t >= 2.0: -1.0 * (t - 1.0) - 2.0 * (t - 2.0)
        expected = torch.tensor([0.0, 0.0, -0.5, -1.0, -2.5, -4.0], dtype=torch.float32)
        self.assertTrue(torch.allclose(result, expected), "Negative slopes test failed")
