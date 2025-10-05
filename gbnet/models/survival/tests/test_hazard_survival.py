import numpy as np
import pandas as pd
from unittest import TestCase

from gbnet.models.survival.hazard_survival import (
    HazardSurvivalModel,
    expand_overlapping_units_locf,
)


class TestHelperFunctions(TestCase):
    """Test helper functions for the hazard survival module."""

    def test_expand_overlapping_units_locf_basic(self):
        """Test basic functionality of expand_overlapping_units_locf."""
        # Create test data with overlapping units
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2, 3],
                "time": [0, 2, 1, 3, 2],
                "feature1": [10, 20, 30, 40, 50],
                "feature2": [100, 200, 300, 400, 500],
            }
        )

        result = expand_overlapping_units_locf(df)

        # Check that LOCF is applied correctly
        unit1_data = result[result["unit_id"] == 1].sort_values("time")
        self.assertEqual(unit1_data.iloc[0]["feature1"], 10)  # Original value
        self.assertEqual(unit1_data.iloc[1]["feature1"], 10)  # Forward filled
        self.assertEqual(unit1_data.iloc[2]["feature1"], 20)  # Original value

    def test_expand_overlapping_units_locf_with_y(self):
        """Test expand_overlapping_units_locf with y parameter."""
        df = pd.DataFrame(
            {"unit_id": [1, 1, 2], "time": [0, 2, 1], "feature1": [10, 20, 30]}
        )
        y = np.array([0, 1, 2, 3])

        result = expand_overlapping_units_locf(df, y)

        # Should include all times from both df and y
        all_times = sorted(set(df["time"].unique()) | set(y))
        result_times = sorted(result["time"].unique())
        self.assertEqual(result_times, all_times)

    def test_expand_overlapping_units_locf_custom_columns(self):
        """Test expand_overlapping_units_locf with custom column names."""
        df = pd.DataFrame(
            {"subject_id": [1, 1, 2], "timestamp": [0, 2, 1], "feature1": [10, 20, 30]}
        )

        result = expand_overlapping_units_locf(
            df, unit_col="subject_id", time_col="timestamp"
        )

        # Check that custom columns are used
        self.assertIn("subject_id", result.columns)
        self.assertIn("timestamp", result.columns)
        self.assertIn("feature1", result.columns)

    def test_expand_overlapping_units_locf_single_unit(self):
        """Test expand_overlapping_units_locf with single unit."""
        df = pd.DataFrame(
            {"unit_id": [1, 1, 1], "time": [0, 1, 2], "feature1": [10, 20, 30]}
        )

        result = expand_overlapping_units_locf(df)

        # Should return the same data since all times are already present
        self.assertEqual(len(result), len(df))
        pd.testing.assert_frame_equal(
            result.sort_values(["unit_id", "time"]).reset_index(drop=True),
            df.sort_values(["unit_id", "time"]).reset_index(drop=True),
        )


class TestHazardSurvivalModel(TestCase):
    """Test the HazardSurvivalModel class."""

    def setUp(self):
        """Set up test data and fit a shared model."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5

        # Create synthetic survival data with DataFrame format
        self.X_static = pd.DataFrame(
            {
                "unit_id": range(self.n_samples),
                "feature1": np.random.randn(self.n_samples),
                "feature2": np.random.randn(self.n_samples),
                "feature3": np.random.randn(self.n_samples),
                "feature4": np.random.randn(self.n_samples),
                "feature5": np.random.randn(self.n_samples),
            }
        )

        # Create survival data
        times = np.random.exponential(5, self.n_samples) + 1
        events = np.random.binomial(1, 0.7, self.n_samples)  # 70% event rate

        self.y = pd.DataFrame(
            {"unit_id": range(self.n_samples), "time": times, "event": events}
        )

        # Create time-varying data
        self.X_time_varying = pd.DataFrame(
            {
                "unit_id": np.repeat(range(self.n_samples), 3),
                "time": np.tile([0, 1, 2], self.n_samples),
                "feature1": np.random.randn(self.n_samples * 3),
                "feature2": np.random.randn(self.n_samples * 3),
                "feature3": np.random.randn(self.n_samples * 3),
            }
        )

        # Fit a shared model for most tests
        self.model = HazardSurvivalModel(nrounds=10)
        self.model.fit(self.X_static, self.y)

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = HazardSurvivalModel()

        self.assertEqual(model.module_type, "XGBModule")
        self.assertEqual(model.nrounds, 50)
        self.assertEqual(model.min_hess, 0.0)
        self.assertEqual(len(model.losses_), 0)
        self.assertIsNone(model.data_format_)

    def test_init_lgb_module(self):
        """Test initialization with LGBModule."""
        model = HazardSurvivalModel(module_type="LGBModule")

        self.assertEqual(model.module_type, "LGBModule")
        self.assertEqual(model.nrounds, 100)

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {"max_depth": 3, "learning_rate": 0.1}
        model = HazardSurvivalModel(
            nrounds=50,
            params=params,
            module_type="LGBModule",
            min_hess=0.1,
        )

        self.assertEqual(model.nrounds, 50)
        self.assertEqual(model.params, params)
        self.assertEqual(model.module_type, "LGBModule")
        self.assertEqual(model.min_hess, 0.1)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        # Check that the shared model was initialized correctly
        self.assertIsNotNone(self.model.integrator_)
        self.assertEqual(len(self.model.losses_), 10)
        self.assertEqual(self.model.data_format_, "static")

    def test_fit_time_varying(self):
        """Test fitting with time-varying data."""
        model = HazardSurvivalModel(nrounds=5)
        model.fit(self.X_time_varying, self.y)

        self.assertIsNotNone(model.integrator_)
        self.assertEqual(len(model.losses_), 5)
        self.assertEqual(model.data_format_, "time_varying")

    def test_fit_invalid_inputs(self):
        """Test that fit raises errors for invalid inputs."""
        model = HazardSurvivalModel()

        # Test with non-DataFrame X
        with self.assertRaises(ValueError):
            model.fit(np.random.randn(10, 5), self.y)

        # Test with non-DataFrame y
        with self.assertRaises(ValueError):
            model.fit(self.X_static, np.random.randn(10))

    def test_predictions_combined(self):
        """Test all prediction methods together."""
        times = np.array([1.0, 2.5, 5.0, 10.0])

        # Test survival probability prediction
        survival_probs = self.model.predict_survival(self.X_static[:5], times)
        self.assertEqual(survival_probs.shape, (5 * (len(times) + 1), 4))
        self.assertTrue(np.all(survival_probs >= 0))
        self.assertTrue(np.all(survival_probs["survival"] <= 1.0001))

        # Test expected survival time prediction
        expected_times = self.model.predict(self.X_static[:5])
        self.assertEqual(expected_times.shape, (5, 3))
        self.assertTrue(np.all(expected_times["expected_time"] > 0))
        self.assertTrue(np.all(expected_times["predicted_median_time"] >= 0))

        # Test scoring
        score = self.model.score(self.X_static[:10], self.y[:10])
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))

    def test_predict_survival_default_times(self):
        """Test predict_survival with default times."""
        survival_probs = self.model.predict_survival(self.X_static[:3])

        # Should use default times (100 points from 0 to max_time)
        expected_times = 100
        self.assertEqual(survival_probs.shape, (3 * expected_times, 4))

    def test_predict_hazard(self):
        """Test hazard prediction functionality."""
        # This should work through predict_base method
        result = self.model.predict_base(self.X_static[:3], self.y[:3])

        # Check that result contains expected keys
        expected_keys = [
            "hazard",
            "survival",
            "unit_last_hazard",
            "unit_integrated_hazard",
            "unit_expected_time",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

    def test_fit_not_fitted_error(self):
        """Test that methods raise error when model is not fitted."""
        model = HazardSurvivalModel()

        with self.assertRaises(AttributeError):
            model.predict_survival(self.X_static, np.array([1, 2, 3]))

        with self.assertRaises(AttributeError):
            model.predict(self.X_static)

        with self.assertRaises(AttributeError):
            model.score(self.X_static, self.y)

    def test_different_module_types_and_edge_cases(self):
        """Test different module types and edge cases."""
        # Test different module types
        for module_type in ["XGBModule", "LGBModule"]:
            with self.subTest(module_type=module_type):
                model = HazardSurvivalModel(module_type=module_type, nrounds=5)

                # Fit the model
                model.fit(self.X_static, self.y)

                # Test prediction
                survival_probs = model.predict_survival(
                    self.X_static[:3], np.array([1.0, 2.0, 3.0])
                )
                self.assertEqual(survival_probs.shape, (12, 4))

        # Test edge cases with small dataset
        X_small = self.X_static[:2]
        y_small = self.y[:2]
        small_model = HazardSurvivalModel(nrounds=5)
        small_model.fit(X_small, y_small)

        # Test prediction with single time point
        survival_probs = small_model.predict_survival(X_small, np.array([1.0]))
        self.assertEqual(survival_probs.shape, (4, 4))

        # Test prediction with single sample
        survival_probs = small_model.predict_survival(
            X_small[:1], np.array([1.0, 2.0, 3.0])
        )
        self.assertEqual(survival_probs.shape, (4, 4))

    def test_loss_decreases(self):
        """Test that loss generally decreases during training."""
        # Check that we have losses recorded
        self.assertEqual(len(self.model.losses_), 10)

        # Check that all losses are finite and that the loss decreases
        self.assertTrue(all(np.isfinite(loss) for loss in self.model.losses_))
        self.assertTrue(self.model.losses_[0] > self.model.losses_[-1])

    def test_static_to_minimal_time_varying_dataset(self):
        """Test the _static_to_minimal_time_varying_dataset method."""
        # Create test data
        X = pd.DataFrame(
            {"unit_id": [1, 2, 3], "time": [5.0, 3.0, 7.0], "feature1": [10, 20, 30]}
        )

        result = self.model._static_to_minimal_time_varying_dataset(X)

        # Check that each unit has two observations (time 0 and original time)
        for unit in X["unit_id"]:
            unit_data = result[result["unit_id"] == unit]
            self.assertEqual(len(unit_data), 2)
            self.assertTrue(0 in unit_data["time"].values)
            self.assertTrue(
                X[X["unit_id"] == unit]["time"].iloc[0] in unit_data["time"].values
            )

    def test_validate_and_convert_input_data_static(self):
        """Test data validation for static datasets."""
        X = pd.DataFrame({"unit_id": [1, 2, 3], "feature1": [10, 20, 30]})
        y = pd.DataFrame(
            {"unit_id": [1, 2, 3], "time": [5.0, 3.0, 7.0], "event": [1, 0, 1]}
        )

        data_format, exp_df, y_out = self.model._validate_and_convert_input_data(X, y)

        self.assertEqual(data_format, "static")
        self.assertIsInstance(exp_df, pd.DataFrame)
        self.assertIsInstance(y_out, pd.DataFrame)

    def test_validate_and_convert_input_data_time_varying(self):
        """Test data validation for time-varying datasets."""
        X = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2],
                "time": [0, 1, 0, 1],
                "feature1": [10, 20, 30, 40],
            }
        )
        y = pd.DataFrame({"unit_id": [1, 2], "time": [1.0, 1.0], "event": [1, 0]})

        data_format, exp_df, y_out = self.model._validate_and_convert_input_data(X, y)

        self.assertEqual(data_format, "time_varying")
        self.assertIsInstance(exp_df, pd.DataFrame)
        self.assertIsInstance(y_out, pd.DataFrame)


class TestHazardSurvivalModelIntegration(TestCase):
    """Integration tests for HazardSurvivalModel."""

    def setUp(self):
        """Set up test data and fit models for integration tests."""
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 3

        # Create more realistic survival data
        self.X_static = pd.DataFrame(
            {
                "unit_id": range(self.n_samples),
                "feature1": np.random.randn(self.n_samples),
                "feature2": np.random.randn(self.n_samples),
                "feature3": np.random.randn(self.n_samples),
            }
        )

        # Create survival times with some structure
        # Higher values of first feature should lead to longer survival
        times = np.random.exponential(5, self.n_samples) + 1

        # Create events (higher probability for shorter times)
        event_probs = 1 / (times + 1)
        events = np.random.binomial(1, event_probs)

        self.y = pd.DataFrame(
            {"unit_id": range(self.n_samples), "time": times, "event": events}
        )

        # Create time-varying data
        self.X_time_varying = pd.DataFrame(
            {
                "unit_id": np.repeat(range(self.n_samples), 3),
                "time": np.tile([0, 1, 2], self.n_samples),
                "feature1": np.random.randn(self.n_samples * 3),
                "feature2": np.random.randn(self.n_samples * 3),
                "feature3": np.random.randn(self.n_samples * 3),
            }
        )

        # Fit models for different module types
        self.xgb_model = HazardSurvivalModel(nrounds=20, module_type="XGBModule")
        self.xgb_model.fit(self.X_static, self.y)

        self.lgb_model = HazardSurvivalModel(nrounds=20, module_type="LGBModule")
        self.lgb_model.fit(self.X_static, self.y)

    def test_end_to_end_training_and_curve_properties(self):
        """Test complete end-to-end training, prediction, and curve properties."""
        # Test XGBoost model predictions
        times = np.array([1.0, 2.0, 5.0, 10.0, 15.0])
        survival_probs = self.xgb_model.predict_survival(self.X_static, times)
        expected_times = self.xgb_model.predict(self.X_static)
        score = self.xgb_model.score(self.X_static, self.y)

        # Check shapes
        self.assertEqual(survival_probs.shape, ((len(times) + 1) * self.n_samples, 4))
        self.assertEqual(expected_times.shape, (self.n_samples, 3))

        # Check value ranges
        print(survival_probs["survival"].max())
        self.assertTrue(np.all(survival_probs["hazard"] >= 0))
        self.assertTrue(np.all(survival_probs["survival"] >= 0))
        self.assertTrue(np.all(survival_probs["survival"] <= 1.0001))
        self.assertTrue(np.all(expected_times["expected_time"] > 0))
        self.assertTrue(np.all(expected_times["predicted_median_time"] > 0))
        self.assertTrue(np.isfinite(score))

        # Check that losses were recorded
        self.assertEqual(len(self.xgb_model.losses_), 20)
        self.assertTrue(all(np.isfinite(loss) for loss in self.xgb_model.losses_))

        # Test survival curve properties with LightGBM model
        sample_X = self.X_static[:1]
        times_detailed = np.linspace(0, self.xgb_model.max_time, 50)

        survival_probs_detailed = self.lgb_model.predict_survival(
            sample_X, times_detailed
        )
        survival_curve = survival_probs_detailed["survival"]

        # Survival probability should generally decrease over time
        # (allowing for some numerical noise)
        for i in range(1, len(survival_curve)):
            self.assertLessEqual(survival_curve[i], survival_curve[i - 1] + 1e-5)

    def test_time_varying_data_integration(self):
        """Test integration with time-varying data."""
        model = HazardSurvivalModel(nrounds=10, module_type="XGBModule")
        model.fit(self.X_time_varying, self.y)

        # Test predictions
        times = np.array([1.0, 2.0, 3.0])
        survival_probs = model.predict_survival(self.X_time_varying[:9], times)
        expected_times = model.predict(self.X_time_varying[:9])

        # Check shapes and values
        self.assertEqual(survival_probs.shape, (3 * (len(times) + 1), 4))
        self.assertEqual(expected_times.shape, (3, 3))
        self.assertTrue(np.all(survival_probs >= 0))
        self.assertTrue(np.all(survival_probs["survival"] <= 1))
        self.assertTrue(np.all(expected_times >= 0))

    def test_data_format_detection(self):
        """Test that data format is correctly detected."""
        # Static data
        static_model = HazardSurvivalModel(nrounds=5)
        static_model.fit(self.X_static, self.y)
        self.assertEqual(static_model.data_format_, "static")

        # Time-varying data
        tv_model = HazardSurvivalModel(nrounds=5)
        tv_model.fit(self.X_time_varying, self.y)
        self.assertEqual(tv_model.data_format_, "time_varying")

    def test_predict_times_method(self):
        """Test the predict_times method directly."""
        times = np.array([1.0, 2.0, 3.0])
        exp_df, udf = self.xgb_model.predict_times(self.X_static[:5], times)

        # Check that both DataFrames are returned
        self.assertIsInstance(exp_df, pd.DataFrame)
        self.assertIsInstance(udf, pd.DataFrame)

        # Check that exp_df has the expected columns
        expected_cols = ["unit_id", "time", "hazard", "survival"]
        for col in expected_cols:
            self.assertIn(col, exp_df.columns)

        # Check that udf has the expected columns
        expected_udf_cols = [
            "unit_id",
            "last_hazard",
            "integrated_hazard",
            "expected_time",
        ]
        for col in expected_udf_cols:
            self.assertIn(col, udf.columns)

    def test_sklearn_compatibility(self):
        """Test sklearn compatibility methods."""
        # Test get_params and set_params
        model = HazardSurvivalModel(nrounds=50, module_type="LGBModule")
        params = model.get_params()

        self.assertIn("nrounds", params)
        self.assertIn("module_type", params)
        self.assertEqual(params["nrounds"], 50)
        self.assertEqual(params["module_type"], "LGBModule")

        # Test set_params
        model.set_params(nrounds=25, module_type="XGBModule")
        self.assertEqual(model.nrounds, 25)
        self.assertEqual(model.module_type, "XGBModule")


if __name__ == "__main__":
    # Run tests
    import unittest

    unittest.main()
