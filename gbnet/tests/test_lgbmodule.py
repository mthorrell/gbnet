from unittest import mock, TestCase

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from gbnet import lgbmodule as lgm


def test_basic_loss():
    gbm = lgm.LGBModule(5, 3, 1, params={"min_data_in_leaf": 0})
    floss = torch.nn.MSELoss()

    gbm.zero_grad()
    np.random.seed(11010)
    input_dataset = lgb.Dataset(np.random.random([5, 3]))
    preds = gbm(input_dataset)
    loss = floss(preds.flatten(), torch.Tensor(np.array([1, 2, 3, 4, 5])).flatten())

    loss.backward(create_graph=True)

    m_obj = mock.MagicMock(side_effect=lgm.LightGBObj)
    m_train = mock.MagicMock(side_effect=lgb.train)
    with (
        mock.patch("gbnet.lgbmodule.LightGBObj", m_obj),
        mock.patch("lightgbm.train", m_train),
    ):
        gbm.gb_step()

    assert (
        np.max(
            np.abs(
                m_obj.call_args_list[-1].args[0].detach().numpy()
                - np.array([-2, -4, -6, -8, -10]).reshape([-1, 1])
            )
        )
        < 1e-8
    )

    assert (
        np.max(
            np.abs(
                m_obj.call_args_list[-1].args[1].detach().numpy()
                - np.array([2, 2, 2, 2, 2]).reshape([-1, 1])
            )
        )
        < 1e-8
    )

    m_train.assert_called_once()


def test_LightGBObj():
    np.random.seed(10101)
    grad = torch.tensor(np.random.random([20, 10]))
    hess = torch.tensor(np.random.random([20, 10]))

    obj = lgm.LightGBObj(grad, hess)

    ograd, ohess = obj(1, 2)
    assert (
        np.max(np.abs(ograd - grad.detach().numpy())) == 0
    ), "LightGBObj grad does not match instantiation"
    assert (
        np.max(np.abs(ohess - hess.detach().numpy())) == 0
    ), "LightGBObj hess does not match instantiation"


class TestLGBModule(TestCase):
    def test_input_is_dataset_training_true_train_dat_none(self):
        """Test with lgb.Dataset input, training mode True, and train_dat is None."""
        module = lgm.LGBModule(100, 10, 1)
        data = np.random.rand(100, 10)
        dataset = lgb.Dataset(data)
        result = module._input_checking_setting(dataset)
        self.assertIs(result, module.train_dat)
        self.assertIsInstance(result, lgb.Dataset)
        self.assertEqual(module.training_n, module.train_dat.num_data())
        self.assertEqual(module.training_n, dataset.num_data())

    def test_input_is_ndarray_training_true_train_dat_none(self):
        """Test with np.ndarray input, training mode True, and train_dat is None."""
        module = lgm.LGBModule(100, 10, 1)
        data = np.random.rand(100, 10)
        result = module._input_checking_setting(data)
        self.assertIs(result, module.train_dat)
        self.assertIsInstance(result, lgb.Dataset)
        self.assertEqual(module.training_n, module.train_dat.num_data())
        self.assertEqual(module.training_n, 100)

    def test_input_is_dataframe_training_true_train_dat_none(self):
        """Test with pd.DataFrame input, training mode True, and train_dat is None."""
        module = lgm.LGBModule(100, 10, 1)
        data = pd.DataFrame(np.random.rand(100, 10))
        result = module._input_checking_setting(data)
        self.assertIs(result, module.train_dat)
        self.assertIsInstance(result, lgb.Dataset)
        self.assertEqual(module.training_n, module.train_dat.num_data())
        self.assertEqual(module.training_n, 100)

    def test_input_is_dataset_training_true_train_dat_set_same_nrows(self):
        """Test with lgb.Dataset input, training mode True, train_dat set, same number of data."""
        module = lgm.LGBModule(100, 10, 1)
        data = np.random.rand(100, 10)
        dataset = lgb.Dataset(data)
        result = module._input_checking_setting(dataset)
        self.assertIs(result, module.train_dat)
        self.assertIs(result, dataset)

    def test_input_is_dataset_training_true_train_dat_set_different_nrows(self):
        """Test with lgb.Dataset input, training mode True, train_dat set, different number of data."""
        module = lgm.LGBModule(100, 10, 1)
        data1 = np.random.rand(100, 10)
        module(data1)
        data2 = np.random.rand(50, 10)
        with self.assertRaises(AssertionError) as context:
            module(data2)
        self.assertIn(
            "Changing datasets while training is not currently supported",
            str(context.exception),
        )

    def test_input_is_dataset_training_false(self):
        """Test with lgb.Dataset input and training mode False."""
        module = lgm.LGBModule(100, 10, 1)
        module(np.random.rand(10, 10))
        module.eval()
        data = np.random.rand(100, 10)
        dataset = lgb.Dataset(data)
        result = module._input_checking_setting(dataset)
        # Should return the original data
        self.assertTrue(np.array_equal(result, data))
        # Ensure free_raw_data is set to False
        self.assertFalse(dataset.free_raw_data)

    def test_input_is_ndarray_training_false(self):
        """Test with np.ndarray input and training mode False."""
        module = lgm.LGBModule(100, 10, 1)
        module.eval()
        data = np.random.rand(100, 10)
        result = module._input_checking_setting(data)
        # Should return input data as is
        self.assertIs(result, data)

    def test_input_is_dataframe_training_false(self):
        """Test with pd.DataFrame input and training mode False."""
        module = lgm.LGBModule(100, 10, 1)
        module.eval()
        data = pd.DataFrame(np.random.rand(100, 10))
        result = module._input_checking_setting(data)
        # Should return input data as is
        self.assertIs(result, data)

    def test_input_invalid_type(self):
        """Test with invalid input type."""
        module = lgm.LGBModule(10, 5, 1)
        data = [1, 2, 3]  # Invalid type
        with self.assertRaises(AssertionError):
            module._input_checking_setting(data)
