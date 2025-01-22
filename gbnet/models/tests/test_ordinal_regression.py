import unittest
import numpy as np
import pandas as pd
from gbnet.models import ordinal_regression


class TestOrdinalRegression(unittest.TestCase):
    def setUp(self):
        # Generate synthetic ordinal data
        num_samples = 200
        num_features = 3
        self.num_classes = 4

        # Generate random features
        self.X = np.random.randn(num_samples, num_features)

        # Generate ordinal targets (0 to num_classes-1)
        self.y = pd.Series(np.random.randint(0, self.num_classes, num_samples))

    def test_fit_predict(self):
        """Test that the estimator can fit and predict without errors."""
        # Split data into train/test
        train_idx = range(150)
        test_idx = range(150, 200)

        X_train = self.X[train_idx]
        y_train = self.y[train_idx]
        X_test = self.X[test_idx]

        # Initialize and fit estimator
        model = ordinal_regression.GBOrd(num_classes=self.num_classes, nrounds=10)
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Check predictions have correct length
        self.assertEqual(len(predictions), len(X_test))

        # Check predictions are not NaN
        self.assertFalse(np.isnan(predictions).any())

        # Check predictions are numerical
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))

    def test_loss_decreases(self):
        """Test that the loss decreases during training"""
        model = ordinal_regression.GBOrd(num_classes=self.num_classes, nrounds=10)
        model.fit(self.X[:50], self.y[:50])

        # Check losses are recorded
        self.assertTrue(len(model.losses_) > 0)

        # Check loss generally decreases
        self.assertLess(np.mean(model.losses_[-3:]), np.mean(model.losses_[:3]))

    def test_invalid_num_classes(self):
        """Test that model raises error if num_classes doesn't match data"""
        with self.assertRaises(AssertionError):
            model = ordinal_regression.GBOrd(
                num_classes=10
            )  # More classes than in data
            model.fit(self.X, self.y)

    def test_fit_predict_xgb(self):
        """Test that the estimator can fit and predict without errors."""
        # Split data into train/test
        train_idx = range(150)
        test_idx = range(150, 200)

        X_train = self.X[train_idx]
        y_train = self.y[train_idx]
        X_test = self.X[test_idx]

        # Initialize and fit estimator
        model = ordinal_regression.GBOrd(
            num_classes=self.num_classes, nrounds=10, module_type="XGBModule"
        )
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Check predictions have correct length
        self.assertEqual(len(predictions), len(X_test))

        # Check predictions are not NaN
        self.assertFalse(np.isnan(predictions).any())

        # Check predictions are numerical
        self.assertTrue(np.issubdtype(predictions.dtype, np.number))

    def test_loss_decreases_xgb(self):
        """Test that the loss decreases during training"""
        model = ordinal_regression.GBOrd(
            num_classes=self.num_classes, nrounds=10, module_type="XGBModule"
        )
        model.fit(self.X[:50], self.y[:50])

        # Check losses are recorded
        self.assertTrue(len(model.losses_) > 0)

        # Check loss generally decreases
        self.assertLess(np.mean(model.losses_[-3:]), np.mean(model.losses_[:3]))

    def test_invalid_num_classes_xgb(self):
        """Test that model raises error if num_classes doesn't match data"""
        with self.assertRaises(AssertionError):
            model = ordinal_regression.GBOrd(
                num_classes=10, module_type="XGBModule"
            )  # More classes than in data
            model.fit(self.X, self.y)
