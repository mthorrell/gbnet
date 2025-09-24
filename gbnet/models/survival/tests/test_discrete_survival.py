import numpy as np
import torch
from unittest import TestCase

from gbnet.models.survival.discrete_survival import (
    BetaSurvivalModel,
    ThetaSurvivalModel,
    loadModule,
    create_data_matrix,
    log_p_event,
    log_p_surv,
    log_p_event_geometric,
    log_p_surv_geometric,
)


class TestHelperFunctions(TestCase):
    """Test helper functions for the discrete beta survival module."""

    def test_loadModule_xgb(self):
        """Test loading XGBModule."""
        XGBModule = loadModule("XGBModule")
        self.assertIsNotNone(XGBModule)
        # Check that it's a class
        self.assertTrue(hasattr(XGBModule, "__call__"))
        # Check that it's the correct XGBModule class
        from gbnet import xgbmodule

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

    def test_create_data_matrix_xgb(self):
        """Test creating XGBoost DMatrix."""
        X = np.random.rand(10, 5)
        dmatrix = create_data_matrix(X, "XGBModule")
        self.assertIsNotNone(dmatrix)
        # Check that it's an XGBoost DMatrix
        import xgboost as xgb

        self.assertIsInstance(dmatrix, xgb.DMatrix)

    def test_create_data_matrix_lgb(self):
        """Test creating LightGBM Dataset."""
        X = np.random.rand(10, 5)
        dataset = create_data_matrix(X, "LGBModule")
        self.assertIsNotNone(dataset)
        # Check that it's a LightGBM Dataset
        import lightgbm as lgb

        self.assertIsInstance(dataset, lgb.Dataset)

    def test_create_data_matrix_invalid(self):
        """Test creating data matrix with invalid module type."""
        X = np.random.rand(10, 5)
        with self.assertRaises(ValueError):
            create_data_matrix(X, "InvalidModule")

    def test_log_p_event(self):
        """Test log probability of event function."""
        t = torch.tensor([1.0, 2.0, 3.0])
        alpha = torch.tensor([2.0, 3.0, 4.0])
        beta = torch.tensor([1.0, 2.0, 3.0])

        log_probs = log_p_event(t, alpha, beta)

        # Check shape
        self.assertEqual(log_probs.shape, t.shape)
        # Check that all values are finite
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

    def test_log_p_surv(self):
        """Test log probability of survival function."""
        t = torch.tensor([1.0, 2.0, 3.0])
        alpha = torch.tensor([2.0, 3.0, 4.0])
        beta = torch.tensor([1.0, 2.0, 3.0])

        log_probs = log_p_surv(t, alpha, beta)

        # Check shape
        self.assertEqual(log_probs.shape, t.shape)
        # Check that all values are finite
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

    def _p_event_recursive(self, t, a, b):
        """Recursive implementation of P(T=t | alpha, beta) for comparison."""
        if t == 1:
            return a / (a + b)
        return (b + t - 2) / (a + b + t - 1) * self._p_event_recursive(t - 1, a, b)

    def _p_surv_recursive(self, t, a, b):
        """Recursive implementation of P(T > t | alpha, beta) for comparison."""
        if t == 1:
            return b / (a + b)
        return (b + t - 1) / (a + b + t - 1) * self._p_surv_recursive(t - 1, a, b)

    def test_log_p_event_vs_recursive(self):
        """Test that log_p_event matches recursive implementation."""
        # Test with various parameter combinations
        test_cases = [
            (1.0, 2.0, 1.0),
            (2.0, 3.0, 2.0),
            (3.0, 1.0, 3.0),
            (5.0, 4.0, 2.0),
            (1.0, 1.0, 1.0),  # Edge case: alpha = beta = 1
        ]

        for t, alpha, beta in test_cases:
            with self.subTest(t=t, alpha=alpha, beta=beta):
                # Get log probability from the function
                t_tensor = torch.tensor(t, dtype=torch.float32)
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
                beta_tensor = torch.tensor(beta, dtype=torch.float32)

                log_prob = log_p_event(t_tensor, alpha_tensor, beta_tensor)

                # Get probability from recursive implementation
                prob_recursive = self._p_event_recursive(t, alpha, beta)
                log_prob_recursive = np.log(prob_recursive)

                # Compare (allowing for numerical precision)
                self.assertAlmostEqual(
                    log_prob.item(),
                    log_prob_recursive,
                    places=5,
                    msg=f"log_p_event({t}, {alpha}, {beta}) mismatch",
                )

    def test_log_p_surv_vs_recursive(self):
        """Test that log_p_surv matches recursive implementation."""
        # Test with various parameter combinations
        test_cases = [
            (1.0, 2.0, 1.0),
            (2.0, 3.0, 2.0),
            (3.0, 1.0, 3.0),
            (5.0, 4.0, 2.0),
            (1.0, 1.0, 1.0),  # Edge case: alpha = beta = 1
        ]

        for t, alpha, beta in test_cases:
            with self.subTest(t=t, alpha=alpha, beta=beta):
                # Get log probability from the function
                t_tensor = torch.tensor(t, dtype=torch.float32)
                alpha_tensor = torch.tensor(alpha, dtype=torch.float32)
                beta_tensor = torch.tensor(beta, dtype=torch.float32)

                log_prob = log_p_surv(t_tensor, alpha_tensor, beta_tensor)

                # Get probability from recursive implementation
                prob_recursive = self._p_surv_recursive(t, alpha, beta)
                log_prob_recursive = np.log(prob_recursive)

                # Compare (allowing for numerical precision)
                self.assertAlmostEqual(
                    log_prob.item(),
                    log_prob_recursive,
                    places=5,
                    msg=f"log_p_surv({t}, {alpha}, {beta}) mismatch",
                )

    def test_log_p_event_batch_vs_recursive(self):
        """Test that log_p_event works correctly with batch inputs."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = torch.tensor([2.0, 3.0, 1.0, 4.0, 2.0])
        beta = torch.tensor([1.0, 2.0, 3.0, 1.0, 3.0])

        # Get log probabilities from the function
        log_probs = log_p_event(t, alpha, beta)

        # Compare with recursive implementation for each element
        for i in range(len(t)):
            with self.subTest(i=i):
                t_val = t[i].item()
                alpha_val = alpha[i].item()
                beta_val = beta[i].item()

                prob_recursive = self._p_event_recursive(t_val, alpha_val, beta_val)
                log_prob_recursive = np.log(prob_recursive)

                self.assertAlmostEqual(
                    log_probs[i].item(),
                    log_prob_recursive,
                    places=5,
                    msg=f"Batch log_p_event[{i}] mismatch",
                )

    def test_log_p_surv_batch_vs_recursive(self):
        """Test that log_p_surv works correctly with batch inputs."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        alpha = torch.tensor([2.0, 3.0, 1.0, 4.0, 2.0])
        beta = torch.tensor([1.0, 2.0, 3.0, 1.0, 3.0])

        # Get log probabilities from the function
        log_probs = log_p_surv(t, alpha, beta)

        # Compare with recursive implementation for each element
        for i in range(len(t)):
            with self.subTest(i=i):
                t_val = t[i].item()
                alpha_val = alpha[i].item()
                beta_val = beta[i].item()

                prob_recursive = self._p_surv_recursive(t_val, alpha_val, beta_val)
                log_prob_recursive = np.log(prob_recursive)

                self.assertAlmostEqual(
                    log_probs[i].item(),
                    log_prob_recursive,
                    places=5,
                    msg=f"Batch log_p_surv[{i}] mismatch",
                )

    def test_probability_consistency(self):
        """Test that event and survival probabilities are consistent."""
        # For discrete time, P(T=t) + P(T>t) should equal P(T>t-1) for t > 1
        t = torch.tensor([2.0, 3.0, 4.0, 5.0])
        alpha = torch.tensor([2.0, 3.0, 1.0, 4.0])
        beta = torch.tensor([1.0, 2.0, 3.0, 1.0])

        log_p_event_vals = log_p_event(t, alpha, beta)
        log_p_surv_vals = log_p_surv(t, alpha, beta)
        log_p_surv_prev_vals = log_p_surv(t - 1, alpha, beta)

        # Convert to probabilities
        p_event = torch.exp(log_p_event_vals)
        p_surv = torch.exp(log_p_surv_vals)
        p_surv_prev = torch.exp(log_p_surv_prev_vals)

        # Check consistency: P(T=t) + P(T>t) = P(T>t-1)
        total_prob = p_event + p_surv
        self.assertTrue(
            torch.allclose(total_prob, p_surv_prev, atol=1e-5),
            msg="Event and survival probabilities are not consistent",
        )


class TestBetaSurvivalModel(TestCase):
    """Test the BetaSurvivalModel class."""

    def setUp(self):
        """Set up test data and fit a shared model."""
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5

        # Create synthetic survival data
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Create structured survival data
        times = np.random.randint(1, 20, self.n_samples)
        events = np.random.binomial(1, 0.7, self.n_samples)  # 70% event rate

        self.y = np.array(
            [(t, e) for t, e in zip(times, events)],
            dtype=[("time", "i4"), ("event", "i4")],
        )

        # Fit a shared model for most tests
        self.model = BetaSurvivalModel(nrounds=10)
        self.model.fit(self.X, self.y)

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        model = BetaSurvivalModel()

        self.assertEqual(model.module_type, "XGBModule")
        self.assertEqual(model.nrounds, 100)  # Default for XGBModule
        self.assertEqual(model.min_hess, 0.0)
        self.assertIsNone(model.model_)
        self.assertEqual(len(model.losses_), 0)

    def test_init_lgb_module(self):
        """Test initialization with LGBModule."""
        model = BetaSurvivalModel(module_type="LGBModule")

        self.assertEqual(model.module_type, "LGBModule")
        self.assertEqual(model.nrounds, 100)  # Default for LGBModule

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        params = {"max_depth": 3, "learning_rate": 0.1}
        model = BetaSurvivalModel(
            nrounds=100,
            params=params,
            module_type="LGBModule",
            min_hess=0.1,
        )

        self.assertEqual(model.nrounds, 100)
        self.assertEqual(model.params, params)
        self.assertEqual(model.module_type, "LGBModule")
        self.assertEqual(model.min_hess, 0.1)

    def test_fit_basic(self):
        """Test basic fitting functionality."""
        # Check that the shared model was initialized correctly
        self.assertIsNotNone(self.model.model_)
        self.assertEqual(self.model.n_features_in_, self.n_features)
        self.assertEqual(len(self.model.losses_), 10)

    def test_predictions_combined(self):
        """Test all prediction methods together."""
        times = [1, 5, 10, 15]

        # Test survival probability prediction
        survival_probs = self.model.predict_survival(self.X[:5], times)
        self.assertEqual(survival_probs.shape, (5, len(times)))
        self.assertTrue(np.all(survival_probs >= 0))
        self.assertTrue(np.all(survival_probs <= 1))

        # Test expected survival time prediction
        expected_times = self.model.predict(self.X[:5])
        self.assertEqual(expected_times.shape, (5,))
        self.assertTrue(np.all(expected_times > 0))

        # Test scoring
        score = self.model.score(self.X[:10], self.y[:10])
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))

    def test_fit_not_fitted_error(self):
        """Test that methods raise error when model is not fitted."""
        model = BetaSurvivalModel()

        with self.assertRaises(Exception):  # check_is_fitted should raise an exception
            model.predict_survival(self.X, [1, 2, 3])

        with self.assertRaises(Exception):
            model.predict(self.X)

        with self.assertRaises(Exception):
            model.score(self.X, self.y)

    def test_different_module_types_and_edge_cases(self):
        """Test different module types and edge cases."""
        # Test different module types
        for module_type in ["XGBModule", "LGBModule"]:
            with self.subTest(module_type=module_type):
                model = BetaSurvivalModel(module_type=module_type, nrounds=5)

                # Fit the model
                model.fit(self.X, self.y)

                # Test prediction
                survival_probs = model.predict_survival(self.X[:3], [1, 5, 10])
                self.assertEqual(survival_probs.shape, (3, 3))

        # Test edge cases with small dataset
        X_small = self.X[:2]
        y_small = self.y[:2]
        small_model = BetaSurvivalModel(nrounds=5)
        small_model.fit(X_small, y_small)

        # Test prediction with single time point
        survival_probs = small_model.predict_survival(X_small, [1])
        self.assertEqual(survival_probs.shape, (2, 1))

        # Test prediction with single sample
        survival_probs = small_model.predict_survival(X_small[:1], [1, 2, 3])
        self.assertEqual(survival_probs.shape, (1, 3))

    def test_loss_decreases(self):
        """Test that loss generally decreases during training."""
        # Check that we have losses recorded
        self.assertEqual(len(self.model.losses_), 10)

        # Check that all losses are finite and that the loss decreases
        self.assertTrue(all(np.isfinite(loss) for loss in self.model.losses_))
        self.assertTrue(self.model.losses_[0] > self.model.losses_[-1])


class TestBetaSurvivalModelIntegration(TestCase):
    """Integration tests for BetaSurvivalModel."""

    def setUp(self):
        """Set up test data and fit models for integration tests."""
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 3

        # Create more realistic survival data
        self.X = np.random.randn(self.n_samples, self.n_features)

        # Create survival times with some structure
        # Higher values of first feature should lead to longer survival
        times = np.random.exponential(5, self.n_samples) + 1
        times = np.floor(times).astype(int)
        times = np.clip(times, 1, 20)  # Clip to reasonable range

        # Create events (higher probability for shorter times)
        event_probs = 1 / (times + 1)
        events = np.random.binomial(1, event_probs)

        self.y = np.array(
            [(t, e) for t, e in zip(times, events)],
            dtype=[("time", "i4"), ("event", "i4")],
        )

        # Fit models for different module types
        self.xgb_model = BetaSurvivalModel(nrounds=20, module_type="XGBModule")
        self.xgb_model.fit(self.X, self.y)

        self.lgb_model = BetaSurvivalModel(nrounds=20, module_type="LGBModule")
        self.lgb_model.fit(self.X, self.y)

    def test_end_to_end_training_and_curve_properties(self):
        """Test complete end-to-end training, prediction, and curve properties."""
        # Test XGBoost model predictions
        times = [1, 5, 10, 15, 20]
        survival_probs = self.xgb_model.predict_survival(self.X, times)
        expected_times = self.xgb_model.predict(self.X)
        score = self.xgb_model.score(self.X, self.y)

        # Check shapes
        self.assertEqual(survival_probs.shape, (self.n_samples, len(times)))
        self.assertEqual(expected_times.shape, (self.n_samples,))

        # Check value ranges
        self.assertTrue(np.all(survival_probs >= 0))
        self.assertTrue(np.all(survival_probs <= 1))
        self.assertTrue(np.all(expected_times > 0))
        self.assertTrue(np.isfinite(score))

        # Check that losses were recorded
        self.assertEqual(len(self.xgb_model.losses_), 20)
        self.assertTrue(all(np.isfinite(loss) for loss in self.xgb_model.losses_))

        # Test survival curve properties with LightGBM model
        sample_X = self.X[:1]
        times_detailed = list(range(21))

        survival_probs_detailed = self.lgb_model.predict_survival(
            sample_X, times_detailed
        )
        survival_curve = survival_probs_detailed[0]

        # Survival probability should be 1 at time 0 (implicitly)
        # and generally decrease over time
        self.assertAlmostEqual(survival_curve[0], 1.0, places=5)

        # Survival should generally be non-increasing
        # (allowing for some numerical noise)
        for i in range(1, len(survival_curve)):
            self.assertLessEqual(survival_curve[i], survival_curve[i - 1] + 1e-6)


class TestThetaHelperFunctions(TestCase):
    """Test geometric helper functions for the theta survival module."""

    def test_log_p_event_geometric(self):
        """Test geometric log probability of event function."""
        t = torch.tensor([1.0, 2.0, 3.0])
        theta = torch.tensor([0.2, 0.5, 0.8])

        log_probs = log_p_event_geometric(t, theta)

        self.assertEqual(log_probs.shape, t.shape)
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        # Known value check: t=2, theta=0.5 -> log(0.5) + 1*log(0.5)
        expected = np.log(0.5) + (2 - 1) * np.log(1 - 0.5)
        self.assertAlmostEqual(log_probs[1].item(), expected, places=7)

    def test_log_p_surv_geometric(self):
        """Test geometric log probability of survival function."""
        t = torch.tensor([0.0, 1.0, 2.0, 3.0])
        theta = torch.tensor([0.2, 0.5, 0.8, 0.3])

        log_probs = log_p_surv_geometric(t, theta)

        self.assertEqual(log_probs.shape, t.shape)
        self.assertTrue(torch.all(torch.isfinite(log_probs)))

        # Known value checks
        # t=0, any theta -> log(1) = 0
        self.assertAlmostEqual(log_probs[0].item(), 0.0, places=7)
        # t=3, theta=0.2 -> 3*log(0.8)
        expected = 2 * np.log(1 - 0.8)
        self.assertAlmostEqual(log_probs[2].item(), expected, places=6)

    def test_probability_consistency_geometric(self):
        """For geometric, P(T=t)+P(T>t)=P(T>t-1) for t>0; P(T>0)=1."""
        t = torch.tensor([1.0, 2.0, 3.0, 4.0])
        theta = torch.tensor([0.2, 0.5, 0.8, 0.3])

        log_p_t = log_p_event_geometric(t, theta)
        log_surv_t = log_p_surv_geometric(t, theta)
        log_surv_prev = log_p_surv_geometric(t - 1, theta)

        p_t = torch.exp(log_p_t)
        p_surv_t = torch.exp(log_surv_t)
        p_surv_prev = torch.exp(log_surv_prev)

        total = p_t + p_surv_t
        self.assertTrue(torch.allclose(total, p_surv_prev, atol=1e-6))


class TestThetaSurvivalModel(TestCase):
    """Test the ThetaSurvivalModel class."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 100
        self.n_features = 5

        self.X = np.random.randn(self.n_samples, self.n_features)

        times = np.random.randint(1, 20, self.n_samples)
        events = np.random.binomial(1, 0.7, self.n_samples)

        self.y = np.array(
            [(t, e) for t, e in zip(times, events)],
            dtype=[("time", "i4"), ("event", "i4")],
        )

        self.model = ThetaSurvivalModel(nrounds=10)
        self.model.fit(self.X, self.y)

    def test_init_default_params(self):
        model = ThetaSurvivalModel()

        self.assertEqual(model.module_type, "XGBModule")
        self.assertEqual(model.nrounds, 100)
        self.assertEqual(model.min_hess, 0.0)
        self.assertIsNone(model.model_)
        self.assertEqual(len(model.losses_), 0)

    def test_init_lgb_module(self):
        model = ThetaSurvivalModel(module_type="LGBModule")

        self.assertEqual(model.module_type, "LGBModule")
        self.assertEqual(model.nrounds, 100)

    def test_init_custom_params(self):
        params = {"max_depth": 3, "learning_rate": 0.1}
        model = ThetaSurvivalModel(
            nrounds=100, params=params, module_type="LGBModule", min_hess=0.1
        )

        self.assertEqual(model.nrounds, 100)
        self.assertEqual(model.params, params)
        self.assertEqual(model.module_type, "LGBModule")
        self.assertEqual(model.min_hess, 0.1)

    def test_fit_basic(self):
        self.assertIsNotNone(self.model.model_)
        self.assertEqual(self.model.n_features_in_, self.n_features)
        self.assertEqual(len(self.model.losses_), 10)

    def test_predictions_combined(self):
        times = [0, 1, 5, 10]

        survival_probs = self.model.predict_survival(self.X[:5], times)
        self.assertEqual(survival_probs.shape, (5, len(times)))
        self.assertTrue(np.all(survival_probs >= 0))
        self.assertTrue(np.all(survival_probs <= 1))

        expected_times = self.model.predict(self.X[:5])
        self.assertEqual(expected_times.shape, (5,))
        self.assertTrue(np.all(expected_times > 0))

        score = self.model.score(self.X[:10], self.y[:10])
        self.assertIsInstance(score, float)
        self.assertTrue(np.isfinite(score))

    def test_fit_not_fitted_error(self):
        model = ThetaSurvivalModel()

        with self.assertRaises(Exception):
            model.predict_survival(self.X, [0, 1, 2])

        with self.assertRaises(Exception):
            model.predict(self.X)

        with self.assertRaises(Exception):
            model.score(self.X, self.y)

    def test_different_module_types_and_edge_cases(self):
        for module_type in ["XGBModule", "LGBModule"]:
            with self.subTest(module_type=module_type):
                model = ThetaSurvivalModel(module_type=module_type, nrounds=5)
                model.fit(self.X, self.y)

                survival_probs = model.predict_survival(self.X[:3], [0, 1, 5, 10])
                self.assertEqual(survival_probs.shape, (3, 4))

        X_small = self.X[:2]
        y_small = self.y[:2]
        small_model = ThetaSurvivalModel(nrounds=5)
        small_model.fit(X_small, y_small)

        survival_probs = small_model.predict_survival(X_small, [0])
        self.assertEqual(survival_probs.shape, (2, 1))

        survival_probs = small_model.predict_survival(X_small[:1], [0, 1, 2])
        self.assertEqual(survival_probs.shape, (1, 3))

    def test_loss_decreases(self):
        self.assertEqual(len(self.model.losses_), 10)
        self.assertTrue(all(np.isfinite(loss) for loss in self.model.losses_))
        self.assertTrue(self.model.losses_[0] > self.model.losses_[-1])


class TestThetaSurvivalModelIntegration(TestCase):
    """Integration tests for ThetaSurvivalModel."""

    def setUp(self):
        np.random.seed(42)
        self.n_samples = 50
        self.n_features = 3

        self.X = np.random.randn(self.n_samples, self.n_features)

        times = np.random.exponential(5, self.n_samples) + 1
        times = np.floor(times).astype(int)
        times = np.clip(times, 1, 20)

        event_probs = 1 / (times + 1)
        events = np.random.binomial(1, event_probs)

        self.y = np.array(
            [(t, e) for t, e in zip(times, events)],
            dtype=[("time", "i4"), ("event", "i4")],
        )

        self.xgb_model = ThetaSurvivalModel(nrounds=20, module_type="XGBModule")
        self.xgb_model.fit(self.X, self.y)

        self.lgb_model = ThetaSurvivalModel(nrounds=20, module_type="LGBModule")
        self.lgb_model.fit(self.X, self.y)

    def test_end_to_end_training_and_curve_properties(self):
        times = [0, 1, 5, 10, 15, 20]
        survival_probs = self.xgb_model.predict_survival(self.X, times)
        expected_times = self.xgb_model.predict(self.X)
        score = self.xgb_model.score(self.X, self.y)

        self.assertEqual(survival_probs.shape, (self.n_samples, len(times)))
        self.assertEqual(expected_times.shape, (self.n_samples,))
        self.assertTrue(np.all(survival_probs >= 0))
        self.assertTrue(np.all(survival_probs <= 1))
        self.assertTrue(np.all(expected_times > 0))
        self.assertTrue(np.isfinite(score))

        self.assertEqual(len(self.xgb_model.losses_), 20)
        self.assertTrue(all(np.isfinite(loss) for loss in self.xgb_model.losses_))

        sample_X = self.X[:1]
        times_detailed = list(range(21))

        survival_probs_detailed = self.lgb_model.predict_survival(
            sample_X, times_detailed
        )
        survival_curve = survival_probs_detailed[0]

        self.assertAlmostEqual(survival_curve[0], 1.0, places=5)

        for i in range(1, len(survival_curve)):
            self.assertLessEqual(survival_curve[i], survival_curve[i - 1] + 1e-6)


if __name__ == "__main__":
    # Run tests
    import unittest

    unittest.main()
