from unittest import mock, TestCase

import numpy as np
import torch
import xgboost as xgb
from gbnet import xgbmodule as xgm


def test_basic_loss():
    gbm = xgm.XGBModule(5, 3, 1)
    floss = torch.nn.MSELoss()

    gbm.zero_grad()
    np.random.seed(11010)
    input_dmatrix = xgb.DMatrix(np.random.random([5, 3]))
    preds = gbm(input_dmatrix)
    loss = floss(preds.flatten(), torch.Tensor(np.array([1, 2, 3, 4, 5])).flatten())

    loss.backward(create_graph=True)

    with (
        mock.patch("gbnet.xgbmodule.XGBObj", side_effect=xgm.XGBObj) as m_obj,
        mock.patch.object(gbm.bst, "boost", side_effect=gbm.bst.boost) as m_boost,
    ):
        gbm.gb_step()

    assert np.all(
        np.isclose(
            m_obj.call_args_list[-1].args[0].detach().numpy(),
            np.array([-2, -4, -6, -8, -10]).reshape([-1, 1]),
        )
    )
    assert np.all(
        np.isclose(
            m_obj.call_args_list[-1].args[1].detach().numpy(),
            np.array([2, 2, 2, 2, 2]).reshape([-1, 1]),
        )
    )

    m_boost.assert_called_once()


def test_XGBObj():
    np.random.seed(10101)
    grad = torch.tensor(np.random.random([20, 10]))
    hess = torch.tensor(np.random.random([20, 10]))

    obj = xgm.XGBObj(grad, hess)

    with mock.patch("xgboost.__version__", new="2.0.0"):
        ograd, ohess = obj(grad, hess)
        assert np.all(np.isclose(ograd, grad.detach().numpy().reshape([-1, 1])))
        assert np.all(np.isclose(ohess, hess.detach().numpy().reshape([-1, 1])))
    with mock.patch("xgboost.__version__", new="2.1.0"):
        ograd, ohess = obj(grad, hess)
        assert np.all(np.isclose(ograd, grad.detach().numpy()))
        assert np.all(np.isclose(ohess, hess.detach().numpy()))


class TestInputChecking(TestCase):
    def test_input_is_dmatrix_training_true_dtrain_none(self):
        """Test with xgb.DMatrix input, training mode True, and dtrain is None."""
        module = xgm.XGBModule(10, 5, 3)
        data = np.random.rand(10, 5)
        dmatrix = xgb.DMatrix(data)
        result = module._input_checking_setting(dmatrix)
        self.assertIs(result, module.dtrain)
        self.assertIs(result, dmatrix)
        np.testing.assert_array_equal(
            result.get_label(), np.zeros(module.batch_size * module.output_dim)
        )
        self.assertEqual(module.training_n, dmatrix.num_row())

    def test_input_is_ndarray_training_true_dtrain_none(self):
        """Test with np.ndarray input, training mode True, and dtrain is None."""
        module = xgm.XGBModule(10, 5, 3)
        data = np.random.rand(10, 5)
        result = module._input_checking_setting(data)
        self.assertIs(result, module.dtrain)
        self.assertIsInstance(result, xgb.DMatrix)
        np.testing.assert_array_equal(
            result.get_label(), np.zeros(module.batch_size * module.output_dim)
        )
        self.assertEqual(module.training_n, data.shape[0])

    def test_input_is_dmatrix_training_true_dtrain_set_same_nrows(self):
        """Test with xgb.DMatrix input, training mode True, and dtrain already set with same number of rows."""
        module = xgm.XGBModule(10, 5, 3)
        data = np.random.rand(10, 5)
        dmatrix = xgb.DMatrix(data)
        result = module._input_checking_setting(dmatrix)
        self.assertIs(result, module.dtrain)
        self.assertIs(result, dmatrix)

    def test_input_is_dmatrix_training_true_dtrain_set_different_nrows(self):
        """Test with xgb.DMatrix input, training mode True, and dtrain already set with different number of rows."""
        module = xgm.XGBModule(10, 5, 3)
        data1 = np.random.rand(10, 5)
        module(data1)
        data2 = np.random.rand(5, 5)
        with self.assertRaises(AssertionError) as context:
            module(data2)
        self.assertIn(
            "Changing datasets while training is not currently supported",
            str(context.exception),
        )

    def test_input_is_dmatrix_training_false(self):
        """Test with xgb.DMatrix input and training mode False."""
        module = xgm.XGBModule(10, 5, 3)
        module(np.random.rand(10, 5))
        module.eval()
        data = np.random.rand(10, 5)
        dmatrix = xgb.DMatrix(data)
        result = module._input_checking_setting(dmatrix)
        self.assertIs(result, dmatrix)

    def test_input_is_ndarray_training_false(self):
        """Test with np.ndarray input and training mode False."""
        module = xgm.XGBModule(10, 5, 3)
        module(np.random.rand(10, 5))
        module.eval()
        data = np.random.rand(10, 5)
        result = module._input_checking_setting(data)
        self.assertIsInstance(result, xgb.DMatrix)
        self.assertEqual(result.num_row(), data.shape[0])

    def test_input_invalid_type(self):
        """Test with invalid input type."""
        module = xgm.XGBModule(10, 5, 3)
        data = [1, 2, 3]  # Invalid type
        with self.assertRaises(AssertionError):
            module._input_checking_setting(data)
