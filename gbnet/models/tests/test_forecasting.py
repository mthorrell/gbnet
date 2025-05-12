import unittest
from unittest.mock import MagicMock, patch

from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal, assert_frame_equal
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


class TestMleG(unittest.TestCase):
    def test_mle_G_when_true_G_is_zero_perfect_data(self):
        """Test _mle_G when true G is zero (r_i^2 = S2)."""
        S2 = 1.0
        h = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])  # Needs some non-zero h
        r_sq = np.full_like(h, S2)
        r = np.sqrt(r_sq)

        G_estimated, _ = forecasting._mle_G(
            r, h, S2, bracket_pad=0.01
        )  # Small pad for precision
        self.assertAlmostEqual(G_estimated, 0.0, places=5)

    def test_mle_G_when_true_G_is_zero_h_all_zeros(self):
        """Test _mle_G when h is all zeros, G should be 0."""
        S2 = 1.0
        h = np.zeros(10)
        r = np.full(10, np.sqrt(S2))  # r^2 = S2 exactly

        # If all h are zero, score function is identically zero.
        # root_scalar might return 'a' (0.0) from bracket (a,b)
        G_estimated, _ = forecasting._mle_G(r, h, S2, bracket_pad=0.1)
        self.assertAlmostEqual(G_estimated, 0.0, places=5)

    def test_mle_G_positive_G_perfect_fit(self):
        """Test _mle_G with a known positive G and no noise."""
        S2 = 1.0
        G_true = 0.5
        h = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0])
        # r_i^2 = S2 + G_true * h_i
        r = np.sqrt(S2 + G_true * h)

        G_estimated, _ = forecasting._mle_G(r, h, S2, bracket_pad=0.01)
        self.assertAlmostEqual(G_estimated, G_true, places=5)

    def test_mle_G_positive_G_with_noise(self):
        """Test _mle_G with positive G and noisy r."""
        S2 = 1.0
        G_true = 0.5
        N = 200  # Larger N for more stable estimate with noise
        h = np.random.rand(N) * 5
        h[0] = 0.0  # Ensure some h can be zero

        # r_i ~ N(0, variance = S2 + G_true * h_i)
        variances = S2 + G_true * h
        # Ensure variances are positive
        self.assertTrue(np.all(variances > 0.00001))  # Avoid zero variance

        r = np.array([np.random.normal(0, np.sqrt(v)) for v in variances])

        G_estimated, _ = forecasting._mle_G(r, h, S2, bracket_pad=1.0)
        # With noise, G_estimated won't be exact. Check if it's reasonably close.
        # This is a statistical estimate, so large deviation is possible for small N.
        self.assertAlmostEqual(G_estimated, G_true, delta=0.35)  # Allow some deviation

    def test_mle_G_h_must_be_non_negative_assertion(self):
        """Test that _mle_G asserts h_i >= 0."""
        S2 = 1.0
        r = np.array([1.0, 1.0])
        h_negative = np.array([1.0, -1.0])
        with self.assertRaisesRegex(AssertionError, "h_i must be non-negative"):
            forecasting._mle_G(r, h_negative, S2)


class TestGetUncertaintyParams(unittest.TestCase):
    @patch("gbnet.models.forecasting.Forecast")  # Patch Forecast where it's looked up
    @patch("gbnet.models.forecasting._mle_G")  # Patch _mle_G where it's looked up
    def test_get_uncertainty_params_main_logic(self, mock_mle_G, MockForecastClass):
        """Test the main workflow of _get_uncertainty_params."""
        # 1. Mock for 'm', the main model instance passed to the function
        mock_m_outer = MagicMock()
        mock_m_outer.nrounds = 10
        mock_m_outer.params = {"key_param": "val_param"}
        mock_m_outer.module_type = "XGBModule"
        mock_m_outer.linear_params = {"lin_key": "lin_val"}
        mock_m_outer.changepoint_params = {"cp_key": "cp_val"}

        # 2. train_residuals (from outer model) and train_df
        N = 40
        self.assertTrue(N >= 4, "Need N>=4 for split and variance calcs with N-1 denom")
        original_train_residuals = np.random.normal(0, 1.5, N)
        expected_final_sigma2 = np.sum(original_train_residuals**2) / (N - 1)

        base_time = pd.Timestamp("2023-01-01")
        train_ds = pd.Series([base_time + pd.Timedelta(days=i) for i in range(N)])
        train_y = np.arange(N, dtype=float) * 0.5
        train_df = pd.DataFrame({"ds": train_ds, "y": train_y})

        # 3. Configure MockForecastClass for the internal 'um' model
        mock_um_instance = MockForecastClass.return_value

        # Determine utrain and utest splits as done in the actual function
        utrain_cutoff_ds = train_df["ds"].quantile(0.5)  # Inclusive for '<='
        utrain_df_internal = (
            train_df[train_df["ds"] <= utrain_cutoff_ds]
            .sort_values("ds")
            .reset_index(drop=True)
        )
        utest_df_internal = (
            train_df[train_df["ds"] > utrain_cutoff_ds]
            .sort_values("ds")
            .reset_index(drop=True)
        )

        N_utrain = utrain_df_internal.shape[0]
        N_utest = utest_df_internal.shape[0]
        self.assertTrue(
            N_utrain >= 2 and N_utest >= 1, "Splits resulted in too small DFs"
        )

        target_S2_for_mle_G = 2.0
        # utrain_residuals = y_utrain - yhat_utrain. We want var(utrain_residuals) = target_S2_for_mle_G
        # (sum of squares) / (N_utrain - 1) = target_S2_for_mle_G
        # If residuals are const c_res: c_res^2 * N_utrain / (N_utrain-1) approx target_S2_for_mle_G
        # More directly: sum(c_res^2) = target_S2_for_mle_G * (N_utrain - 1)
        # c_res = sqrt(target_S2_for_mle_G * (N_utrain - 1) / N_utrain)
        utrain_residual_val = np.sqrt(target_S2_for_mle_G * (N_utrain - 1) / N_utrain)

        yhat_utrain = utrain_df_internal["y"].values - utrain_residual_val
        utest_residuals_for_mle = np.random.normal(0, 1, N_utest)
        yhat_utest = utest_df_internal["y"].values - utest_residuals_for_mle

        # Configure predict method of mock_um_instance
        # It's called twice: first with utrain, then with utest.
        # Using a list of return values for side_effect is often cleaner.
        mock_um_instance.predict.side_effect = [
            pd.DataFrame({"yhat": yhat_utrain}),  # For utrain call
            pd.DataFrame({"yhat": yhat_utest}),  # For utest call
        ]
        mock_um_instance.fit.return_value = None

        # 4. Configure mock_mle_G
        expected_G_from_mle = 0.8
        mock_mle_G.return_value = (expected_G_from_mle, "mock_mle_diagnostics")

        # 5. Call the function under test
        returned_sigma2, returned_G = forecasting._get_uncertainty_params(
            mock_m_outer, original_train_residuals, train_df
        )

        # 6. Assertions
        self.assertAlmostEqual(returned_sigma2, expected_final_sigma2, places=5)
        self.assertEqual(returned_G, expected_G_from_mle)

        MockForecastClass.assert_called_once_with(
            nrounds=mock_m_outer.nrounds,
            params=mock_m_outer.params,
            module_type=mock_m_outer.module_type,
            linear_params=mock_m_outer.linear_params,
            changepoint_params=mock_m_outer.changepoint_params,
            estimate_uncertainty=False,
        )

        # Verify um.fit call arguments
        fit_call = mock_um_instance.fit.call_args
        self.assertIsNotNone(fit_call, "um.fit was not called")
        called_fit_df, called_fit_y = fit_call[0]
        assert_frame_equal(
            called_fit_df[["ds", "y"]], utrain_df_internal, check_dtype=False
        )
        assert_series_equal(called_fit_y, utrain_df_internal["y"], check_dtype=False)

        # Verify mock_mle_G was called correctly
        secs_in_year = 60 * 60 * 24 * 365.0
        expected_h_utrain_max_time_years = (
            forecasting.pd_datetime_to_seconds(utrain_df_internal["ds"]).max()
            / secs_in_year
        )
        expected_h_utest_ds_years = (
            forecasting.pd_datetime_to_seconds(utest_df_internal["ds"]) / secs_in_year
        )
        expected_h_values = (
            expected_h_utest_ds_years.values - expected_h_utrain_max_time_years
        )

        mle_G_call_args_list = mock_mle_G.call_args_list
        self.assertEqual(len(mle_G_call_args_list), 1, "_mle_G should be called once")
        called_r_for_mle, called_h_for_mle, called_S2_for_mle = mle_G_call_args_list[0][
            0
        ]

        np.testing.assert_array_almost_equal(
            called_r_for_mle, utest_residuals_for_mle, decimal=5
        )
        np.testing.assert_array_almost_equal(
            called_h_for_mle, expected_h_values, decimal=5
        )
        self.assertAlmostEqual(called_S2_for_mle, target_S2_for_mle_G, places=5)

    @patch("gbnet.models.forecasting.Forecast")
    @patch("gbnet.models.forecasting._mle_G")
    def test_get_uncertainty_params_mle_G_raises_ValueError(
        self, mock_mle_G, MockForecastClass
    ):
        """Test _get_uncertainty_params when _mle_G raises ValueError."""
        mock_m_outer = MagicMock()
        mock_m_outer.nrounds = 5
        mock_m_outer.params = {}
        mock_m_outer.module_type = "XGBModule"
        mock_m_outer.linear_params = {}
        mock_m_outer.changepoint_params = {}

        N = 20
        original_train_residuals = np.random.randn(N)
        expected_final_sigma2 = np.sum(original_train_residuals**2) / (N - 1)

        base_time = pd.Timestamp("2024-01-01")
        train_ds = pd.Series([base_time + pd.Timedelta(hours=i * 12) for i in range(N)])
        train_y = np.sin(np.linspace(0, 4 * np.pi, N))
        train_df = pd.DataFrame({"ds": train_ds, "y": train_y})

        mock_um_instance = MockForecastClass.return_value

        # Mock predict to return yhats that allow residuals to be calculated
        def predict_side_effect_for_value_error(df_arg):
            return pd.DataFrame({"yhat": np.zeros(df_arg.shape[0])})

        mock_um_instance.predict.side_effect = predict_side_effect_for_value_error
        mock_um_instance.fit.return_value = None

        mock_mle_G.side_effect = ValueError("MLE failed for testing")

        returned_sigma2, returned_G = forecasting._get_uncertainty_params(
            mock_m_outer, original_train_residuals, train_df
        )

        self.assertAlmostEqual(returned_sigma2, expected_final_sigma2, places=5)
        self.assertEqual(
            returned_G, 0.0, "G should default to 0.0 on ValueError from _mle_G"
        )
        mock_mle_G.assert_called_once()
