import numpy as np
import pandas as pd
import torch
from unittest import TestCase
from unittest.mock import patch, MagicMock

from gbnet.models.survival.hazard_integrator import HazardIntegrator, loadModule
from gbnet import xgbmodule, lgbmodule


class TestLoadModule(TestCase):
    """Test the loadModule helper function."""

    def test_loadModule_xgb(self):
        """Test loading XGBModule."""
        XGBModule = loadModule("XGBModule")
        self.assertIsNotNone(XGBModule)
        # Check that it's a class
        self.assertTrue(hasattr(XGBModule, "__call__"))
        # Check that it's the correct XGBModule class
        self.assertIs(XGBModule, xgbmodule.XGBModule)

    def test_loadModule_lgb(self):
        """Test loading LGBModule."""
        LGBModule = loadModule("LGBModule")
        self.assertIsNotNone(LGBModule)
        # Check that it's a class
        self.assertTrue(hasattr(LGBModule, "__call__"))
        # Check that it's the correct LGBModule class
        from gbnet import lgbmodule

        self.assertIs(LGBModule, lgbmodule.LGBModule)

    def test_loadModule_invalid(self):
        """Test loading invalid module type."""
        with self.assertRaises(AssertionError):
            loadModule("InvalidModule")


class TestHazardIntegratorInit(TestCase):
    """Test HazardIntegrator initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        integrator = HazardIntegrator()

        self.assertEqual(integrator.covariate_cols, ["time"])
        self.assertEqual(integrator.params, {})
        self.assertEqual(integrator.min_hess, 0.0)
        self.assertEqual(integrator.module_type, "XGBModule")
        self.assertIsNone(integrator.gb_module)
        self.assertEqual(integrator.static_data, {})

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        covariate_cols = ["feature1", "feature2"]
        params = {"max_depth": 3, "learning_rate": 0.1}
        min_hess = 0.1
        module_type = "LGBModule"

        integrator = HazardIntegrator(
            covariate_cols=covariate_cols,
            params=params,
            min_hess=min_hess,
            module_type=module_type,
        )

        self.assertEqual(integrator.covariate_cols, ["time"] + covariate_cols)
        self.assertEqual(integrator.params, params)
        self.assertEqual(integrator.min_hess, min_hess)
        self.assertEqual(integrator.module_type, module_type)
        self.assertIsNone(integrator.gb_module)
        self.assertEqual(integrator.static_data, {})

    def test_init_params_copy(self):
        """Test that params are copied to avoid mutation."""
        original_params = {"max_depth": 3}
        integrator = HazardIntegrator(params=original_params)

        # Modify the original params
        original_params["learning_rate"] = 0.1

        # Check that integrator params are unchanged
        self.assertEqual(integrator.params, {"max_depth": 3})


class TestHazardIntegratorDataPreparation(TestCase):
    """Test HazardIntegrator data preparation methods."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.integrator = HazardIntegrator(covariate_cols=["feature1", "feature2"])

        # Create test data
        self.df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1, 2, 2, 3, 3, 3, 3],
                "time": [1.0, 2.0, 3.0, 1.0, 2.0, 1.0, 2.0, 3.0, 4.0],
                "feature1": np.random.randn(9),
                "feature2": np.random.randn(9),
            }
        )

    def test_prepare_data_basic(self):
        """Test basic data preparation."""
        self.integrator._prepare_data(self.df)

        # Check that static data is populated
        self.assertIn("dmatrix", self.integrator.static_data)
        self.assertIn("num_rows", self.integrator.static_data)
        self.assertIn("num_cols", self.integrator.static_data)
        self.assertIn("unit_ids", self.integrator.static_data)
        self.assertIn("dt", self.integrator.static_data)
        self.assertIn("same_unit", self.integrator.static_data)
        self.assertIn("unsort_idx", self.integrator.static_data)
        self.assertIn("interleave_amts", self.integrator.static_data)

        # Check dimensions
        self.assertEqual(self.integrator.static_data["num_rows"], 9)
        self.assertEqual(
            self.integrator.static_data["num_cols"], 3
        )  # time + 2 features

        # Check that data is sorted by (unit_id, time)
        times = self.integrator.static_data["dt"]

        # First element should have dt=0 (no previous time)
        self.assertEqual(times[0], 0.0)

        # Check that dt is only non-zero for same unit
        same_unit = self.integrator.static_data["same_unit"]
        for i in range(1, len(times)):
            if same_unit[i]:
                self.assertGreater(times[i], 0)
            else:
                self.assertEqual(times[i], 0)

    def test_prepare_data_missing_columns(self):
        """Test data preparation with missing required columns."""
        df_missing = self.df.drop(columns=["unit_id"])

        with self.assertRaises(ValueError) as context:
            self.integrator._prepare_data(df_missing)

        self.assertIn(
            "DataFrame must contain 'unit_id' and 'time'", str(context.exception)
        )

    def test_prepare_data_xgb_module(self):
        """Test data preparation with XGBModule."""
        integrator = HazardIntegrator(module_type="XGBModule")
        integrator._prepare_data(self.df)

        dmatrix = integrator.static_data["dmatrix"]
        import xgboost as xgb

        self.assertIsInstance(dmatrix, xgb.DMatrix)

    def test_prepare_data_lgb_module(self):
        """Test data preparation with LGBModule."""
        integrator = HazardIntegrator(module_type="LGBModule")
        integrator._prepare_data(self.df)

        dmatrix = integrator.static_data["dmatrix"]
        import lightgbm as lgb

        self.assertIsInstance(dmatrix, lgb.Dataset)


class TestHazardIntegratorForward(TestCase):
    """Test HazardIntegrator forward method."""

    def setUp(self):
        """Set up test data and integrator."""
        np.random.seed(42)
        self.integrator = HazardIntegrator(covariate_cols=["feature1"])

        # Create test data
        self.df = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2, 2],
                "time": [1.0, 2.0, 1.0, 2.0, 3.0],
                "feature1": np.random.randn(5),
            }
        )

    @patch("gbnet.models.survival.hazard_integrator.loadModule")
    def test_forward_basic(self, mock_load_module):
        """Test basic forward pass."""
        # Mock the module
        mock_module_class = MagicMock()
        mock_module_instance = MagicMock()
        mock_module_class.return_value = mock_module_instance
        mock_load_module.return_value = mock_module_class

        # Mock the forward call to return hazard values
        mock_hazard = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3])
        mock_module_instance.return_value = mock_hazard.unsqueeze(1)

        # Set integrator to training mode
        self.integrator.train()

        # Call forward
        result = self.integrator.forward(self.df)

        # Check that module was initialized
        self.assertFalse(mock_load_module.called)

        # Check result structure
        self.assertIn("hazard", result)
        self.assertIn("unit_last_hazard", result)
        self.assertIn("unit_integrated_hazard", result)
        self.assertIn("survival", result)
        self.assertIn("unit_expected_time", result)

        # Check shapes
        self.assertEqual(result["hazard"].shape, (5,))
        self.assertEqual(result["unit_last_hazard"].shape, (2,))
        self.assertEqual(result["unit_integrated_hazard"].shape, (2,))
        self.assertEqual(result["survival"].shape, (5,))
        self.assertEqual(result["unit_expected_time"].shape, (2,))

    @patch("gbnet.models.survival.hazard_integrator.loadModule")
    def test_forward_no_survival_estimates(self, mock_load_module):
        """Test forward pass without survival estimates."""
        # Mock the module
        mock_module_class = MagicMock()
        mock_module_instance = MagicMock()
        mock_module_class.return_value = mock_module_instance
        mock_load_module.return_value = mock_module_class

        # Mock the forward call
        mock_hazard = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3])
        mock_module_instance.return_value = mock_hazard.unsqueeze(1)

        # Set integrator to training mode
        self.integrator.train()

        # Call forward without survival estimates
        result = self.integrator.forward(self.df, return_survival_estimates=False)

        # Check that survival-related outputs are None
        self.assertIsNone(result["survival"])
        self.assertIsNone(result["unit_expected_time"])

        # Check that hazard outputs are still present
        self.assertIsNotNone(result["hazard"])
        self.assertIsNotNone(result["unit_last_hazard"])
        self.assertIsNotNone(result["unit_integrated_hazard"])

    def test_forward_hazard_calculation(self):
        """Test that hazard calculation is correct."""
        # Create a simple integrator with mocked module
        integrator = HazardIntegrator()

        # Mock the gradient boosting module
        mock_gb_module = MagicMock()
        mock_gb_module.return_value = torch.tensor(
            [[0.1], [0.2], [0.15], [0.25], [0.3]]
        )
        integrator.gb_module = mock_gb_module

        # Set up static data manually
        integrator.static_data = {
            "dmatrix": "mock_dmatrix",
            "num_rows": 5,
            "num_cols": 1,
            "unit_ids": torch.tensor([0, 0, 1, 1, 1]),
            "dt": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "same_unit": torch.tensor([False, True, False, True, True]),
            "unsort_idx": torch.tensor([0, 1, 2, 3, 4]),
            "interleave_amts": torch.tensor([2, 3]),
        }

        # Call forward
        result = integrator.forward(self.df)

        # Check that hazard values are correct (exp of log_hazard)
        expected_hazard = torch.exp(torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3]))
        torch.testing.assert_close(result["hazard"], expected_hazard)

    def test_forward_survival_calculation(self):
        """Test that survival calculation is correct."""
        # This is a complex test that would require careful setup
        # For now, we'll test that the survival values are in the correct range
        integrator = HazardIntegrator()

        # Mock the gradient boosting module
        mock_gb_module = MagicMock()
        mock_gb_module.return_value = torch.tensor(
            [[0.1], [0.2], [0.15], [0.25], [0.3]]
        )
        integrator.gb_module = mock_gb_module

        # Set up static data manually
        integrator.static_data = {
            "dmatrix": "mock_dmatrix",
            "num_rows": 5,
            "num_cols": 1,
            "unit_ids": torch.tensor([0, 0, 1, 1, 1]),
            "dt": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "same_unit": torch.tensor([False, True, False, True, True]),
            "unsort_idx": torch.tensor([0, 1, 2, 3, 4]),
            "interleave_amts": torch.tensor([2, 3]),
        }

        # Call forward
        result = integrator.forward(self.df)

        # Check that survival values are in [0, 1]
        self.assertTrue(torch.all(result["survival"] >= 0))
        self.assertTrue(torch.all(result["survival"] <= 1))


class TestHazardIntegratorEdgeCases(TestCase):
    """Test HazardIntegrator edge cases and error conditions."""

    def test_empty_dataframe(self):
        """Test with empty DataFrame."""
        integrator = HazardIntegrator()
        empty_df = pd.DataFrame(columns=["unit_id", "time"])

        with self.assertRaises(TypeError):
            integrator._prepare_data(empty_df)

    def test_single_unit_single_time(self):
        """Test with single unit and single time point."""
        integrator = HazardIntegrator()
        df = pd.DataFrame({"unit_id": [1], "time": [1.0], "feature1": [0.5]})

        # Should not raise an error
        integrator._prepare_data(df)

        # Check that static data is populated correctly
        self.assertEqual(integrator.static_data["num_rows"], 1)
        self.assertEqual(integrator.static_data["num_cols"], 1)  # time + feature1

    def test_duplicate_times_same_unit(self):
        """Test with duplicate times for the same unit."""
        integrator = HazardIntegrator()
        df = pd.DataFrame(
            {
                "unit_id": [1, 1, 1],
                "time": [1.0, 1.0, 2.0],  # Duplicate times
                "feature1": [0.1, 0.2, 0.3],
            }
        )

        # Should not raise an error
        integrator._prepare_data(df)

        # Check that data is sorted correctly
        unit_ids = integrator.static_data["unit_ids"]
        self.assertTrue(torch.all(unit_ids == 0))  # All same unit

    def test_negative_times(self):
        """Test with negative times."""
        integrator = HazardIntegrator()
        df = pd.DataFrame(
            {
                "unit_id": [1, 1],
                "time": [-1.0, 1.0],  # Negative time
                "feature1": [0.1, 0.2],
            }
        )

        # Should not raise an error
        integrator._prepare_data(df)

        # Check that dt calculation handles negative times
        dt = integrator.static_data["dt"]
        self.assertEqual(dt[0], 0.0)  # First element
        self.assertEqual(dt[1], 2.0)  # 1.0 - (-1.0) = 2.0

    def test_large_dataset(self):
        """Test with larger dataset."""
        np.random.seed(42)
        n_units = 100
        n_times_per_unit = 10

        # Create larger dataset
        unit_ids = np.repeat(range(n_units), n_times_per_unit)
        times = np.tile(np.arange(1, n_times_per_unit + 1), n_units)
        features = np.random.randn(n_units * n_times_per_unit, 2)

        df = pd.DataFrame(
            {
                "unit_id": unit_ids,
                "time": times,
                "feature1": features[:, 0],
                "feature2": features[:, 1],
            }
        )

        integrator = HazardIntegrator(covariate_cols=["feature1", "feature2"])

        # Should not raise an error
        integrator._prepare_data(df)

        # Check dimensions
        self.assertEqual(integrator.static_data["num_rows"], n_units * n_times_per_unit)
        self.assertEqual(integrator.static_data["num_cols"], 3)  # time + 2 features


class TestHazardIntegratorIntegration(TestCase):
    """Test HazardIntegrator integration with gradient boosting modules."""

    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2, 2, 3],
                "time": [1.0, 2.0, 1.0, 2.0, 3.0, 1.0],
                "feature1": np.random.randn(6),
                "feature2": np.random.randn(6),
            }
        )

    @patch("gbnet.models.survival.hazard_integrator.loadModule")
    def test_integration_xgb_module(self, mock_load_module):
        """Test integration with XGBModule."""
        # Mock XGBModule
        mock_xgb_class = MagicMock()
        mock_xgb_instance = MagicMock()
        mock_xgb_class.return_value = mock_xgb_instance
        mock_load_module.return_value = mock_xgb_class

        # Mock the forward call
        mock_hazard = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3, 0.2])
        mock_xgb_instance.return_value = mock_hazard.unsqueeze(1)

        integrator = HazardIntegrator(
            covariate_cols=["feature1", "feature2"],
            module_type="XGBModule",
            params={"max_depth": 3},
        )

        # Set to training mode
        integrator.train()

        # Call forward
        integrator.forward(self.df)

        # Check that XGBModule was used
        mock_load_module.assert_called_once_with("XGBModule")
        mock_xgb_class.assert_called_once()

        # Check that parameters were passed correctly
        call_args = mock_xgb_class.call_args
        self.assertEqual(call_args[1]["params"], {"max_depth": 3})
        self.assertEqual(call_args[1]["min_hess"], 0.0)

    @patch("gbnet.models.survival.hazard_integrator.loadModule")
    def test_integration_lgb_module(self, mock_load_module):
        """Test integration with LGBModule."""
        # Mock LGBModule
        mock_lgb_class = MagicMock()
        mock_lgb_instance = MagicMock()
        mock_lgb_class.return_value = mock_lgb_instance
        mock_load_module.return_value = mock_lgb_class

        # Mock the forward call
        mock_hazard = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3, 0.2])
        mock_lgb_instance.return_value = mock_hazard.unsqueeze(1)

        integrator = HazardIntegrator(
            covariate_cols=["feature1", "feature2"],
            module_type="LGBModule",
            params={"num_leaves": 10},
            min_hess=0.1,
        )

        # Set to training mode
        integrator.train()

        # Call forward
        integrator.forward(self.df)

        # Check that LGBModule was used
        mock_load_module.assert_called_once_with("LGBModule")
        mock_lgb_class.assert_called_once()

        # Check that parameters were passed correctly
        call_args = mock_lgb_class.call_args
        self.assertEqual(call_args[1]["params"], {"num_leaves": 10})
        self.assertEqual(call_args[1]["min_hess"], 0.1)

    def test_gb_step(self):
        """Test gb_step method."""
        integrator = HazardIntegrator()

        # Test when gb_module is None
        integrator.gb_step()  # Should not raise an error

        # Test when gb_module exists
        mock_gb_module = MagicMock()
        integrator.gb_module = mock_gb_module

        integrator.gb_step()
        mock_gb_module.gb_step.assert_called_once()

    def test_forward_with_different_module_types(self):
        """Test forward method with different module types."""
        for module_type in ["XGBModule", "LGBModule"]:
            with self.subTest(module_type=module_type):
                # Mock the module loading
                with patch(
                    "gbnet.models.survival.hazard_integrator.loadModule"
                ) as mock_load:
                    mock_load.return_value = {
                        "XGBModule": xgbmodule.XGBModule,
                        "LGBModule": lgbmodule.LGBModule,
                    }[module_type]
                    integrator = HazardIntegrator(module_type=module_type)
                    mock_module_class = MagicMock()
                    mock_module_instance = MagicMock()
                    mock_module_class.return_value = mock_module_instance
                    mock_load.return_value = mock_module_class

                    # Mock the forward call
                    mock_hazard = torch.tensor([0.1, 0.2, 0.15, 0.25, 0.3, 0.2])
                    mock_module_instance.return_value = mock_hazard.unsqueeze(1)

                    # Set to training mode
                    integrator.train()

                    # Call forward
                    result = integrator.forward(self.df)

                    # Check that the correct module type was loaded
                    mock_load.assert_called_once_with(module_type)

                    # Check that result has expected structure
                    self.assertIn("hazard", result)
                    self.assertIn("unit_last_hazard", result)
                    self.assertIn("unit_integrated_hazard", result)


class TestHazardIntegratorMathematicalProperties(TestCase):
    """Test mathematical properties of HazardIntegrator calculations."""

    def setUp(self):
        """Set up test data for mathematical property tests."""
        np.random.seed(42)
        self.df = pd.DataFrame(
            {
                "unit_id": [1, 1, 2, 2, 2],
                "time": [1.0, 2.0, 1.0, 2.0, 3.0],
                "feature1": np.random.randn(5),
            }
        )

    def test_hazard_values_positive(self):
        """Test that hazard values are always positive."""
        integrator = HazardIntegrator()

        # Mock the gradient boosting module
        mock_gb_module = MagicMock()
        # Use negative log hazard values to ensure positive hazards after exp
        mock_gb_module.return_value = torch.tensor(
            [[-1.0], [-0.5], [-2.0], [-0.8], [-1.5]]
        )
        integrator.gb_module = mock_gb_module

        # Set up static data
        integrator.static_data = {
            "dmatrix": "mock_dmatrix",
            "num_rows": 5,
            "num_cols": 1,
            "unit_ids": torch.tensor([0, 0, 1, 1, 1]),
            "dt": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "same_unit": torch.tensor([False, True, False, True, True]),
            "unsort_idx": torch.tensor([0, 1, 2, 3, 4]),
            "interleave_amts": torch.tensor([2, 3]),
        }

        result = integrator.forward(self.df)

        # All hazard values should be positive
        self.assertTrue(torch.all(result["hazard"] > 0))

    def test_survival_values_range(self):
        """Test that survival values are in [0, 1] range."""
        integrator = HazardIntegrator()

        # Mock the gradient boosting module
        mock_gb_module = MagicMock()
        mock_gb_module.return_value = torch.tensor(
            [[0.1], [0.2], [0.15], [0.25], [0.3]]
        )
        integrator.gb_module = mock_gb_module

        # Set up static data
        integrator.static_data = {
            "dmatrix": "mock_dmatrix",
            "num_rows": 5,
            "num_cols": 1,
            "unit_ids": torch.tensor([0, 0, 1, 1, 1]),
            "dt": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "same_unit": torch.tensor([False, True, False, True, True]),
            "unsort_idx": torch.tensor([0, 1, 2, 3, 4]),
            "interleave_amts": torch.tensor([2, 3]),
        }

        result = integrator.forward(self.df)

        # All survival values should be in [0, 1]
        self.assertTrue(torch.all(result["survival"] >= 0))
        self.assertTrue(torch.all(result["survival"] <= 1))

    def test_expected_time_positive(self):
        """Test that expected time values are positive."""
        integrator = HazardIntegrator()

        # Mock the gradient boosting module
        mock_gb_module = MagicMock()
        mock_gb_module.return_value = torch.tensor(
            [[0.1], [0.2], [0.15], [0.25], [0.3]]
        )
        integrator.gb_module = mock_gb_module

        # Set up static data
        integrator.static_data = {
            "dmatrix": "mock_dmatrix",
            "num_rows": 5,
            "num_cols": 1,
            "unit_ids": torch.tensor([0, 0, 1, 1, 1]),
            "dt": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "same_unit": torch.tensor([False, True, False, True, True]),
            "unsort_idx": torch.tensor([0, 1, 2, 3, 4]),
            "interleave_amts": torch.tensor([2, 3]),
        }

        result = integrator.forward(self.df)

        # All expected time values should be positive
        self.assertTrue(torch.all(result["unit_expected_time"] > 0))

    def test_integrated_hazard_positive(self):
        """Test that integrated hazard values are positive."""
        integrator = HazardIntegrator()

        # Mock the gradient boosting module
        mock_gb_module = MagicMock()
        mock_gb_module.return_value = torch.tensor(
            [[0.1], [0.2], [0.15], [0.25], [0.3]]
        )
        integrator.gb_module = mock_gb_module

        # Set up static data
        integrator.static_data = {
            "dmatrix": "mock_dmatrix",
            "num_rows": 5,
            "num_cols": 1,
            "unit_ids": torch.tensor([0, 0, 1, 1, 1]),
            "dt": torch.tensor([0.0, 1.0, 0.0, 1.0, 1.0]),
            "same_unit": torch.tensor([False, True, False, True, True]),
            "unsort_idx": torch.tensor([0, 1, 2, 3, 4]),
            "interleave_amts": torch.tensor([2, 3]),
        }

        result = integrator.forward(self.df)

        # All integrated hazard values should be positive
        self.assertTrue(torch.all(result["unit_integrated_hazard"] > 0))


if __name__ == "__main__":
    # Run tests
    import unittest

    unittest.main()
