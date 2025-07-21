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


class TestAssertions(TestCase):
    def test_assert_objective_in_params(self):
        params = {"objective": "regression"}
        with self.assertRaises(AssertionError) as context:
            lgm.LGBModule(5, 3, 1, params=params)
        self.assertIn(
            "objective should not be specified in params", str(context.exception)
        )


class TestLGBModule(TestCase):
    def test_input_is_dataset_training_true_train_dat_none(self):
        """Test with lgb.Dataset input, training mode True, and train_dat is None."""
        module = lgm.LGBModule(100, 10, 1)
        data = np.random.rand(100, 10)
        dataset = lgb.Dataset(data)
        result = module._input_checking_setting(dataset)
        self.assertIs(result, module.train_dat)
        self.assertIsInstance(result, lgb.Dataset)

    def test_input_is_ndarray_training_true_train_dat_none(self):
        """Test with np.ndarray input, training mode True, and train_dat is None."""
        module = lgm.LGBModule(100, 10, 1)
        data = np.random.rand(100, 10)
        result = module._input_checking_setting(data)
        self.assertIs(result, module.train_dat)
        self.assertIsInstance(result, lgb.Dataset)

    def test_input_is_dataframe_training_true_train_dat_none(self):
        """Test with pd.DataFrame input, training mode True, and train_dat is None."""
        module = lgm.LGBModule(100, 10, 1)
        data = pd.DataFrame(np.random.rand(100, 10))
        result = module._input_checking_setting(data)
        self.assertIs(result, module.train_dat)
        self.assertIsInstance(result, lgb.Dataset)

    def test_input_is_dataset_training_true_train_dat_set_same_nrows(self):
        """Test with lgb.Dataset input, training mode True, train_dat set, same number of data."""
        module = lgm.LGBModule(100, 10, 1)
        data = np.random.rand(100, 10)
        dataset = lgb.Dataset(data)
        result = module._input_checking_setting(dataset)
        self.assertIs(result, module.train_dat)
        self.assertIs(result, dataset)

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

    def test_if_bst_is_none(self):
        """
        Test scenario where self.training = True, self.train_dat is not None,
        and self.bst is None. The method should return self.train_dat directly.
        """
        module = lgm.LGBModule(10, 5, 1)
        arr = np.random.rand(10, 5)
        module._set_train_dat(lgb.Dataset(arr))

        # Since bst is None, method should return train_dat directly
        result = module._input_checking_setting(np.random.rand(10, 5))
        self.assertIs(
            result, module.train_dat, "Expected to return train_dat when bst is None."
        )

    def test_raises_input_changed_lgb_dataset(self):
        """
        Test scenario where self.training = True, self.train_dat not None,
        self.bst not None, and input_dataset is an lgb.Dataset with a _handle.
        Expect no assertion error and return self.train_dat.
        """
        module = lgm.LGBModule(10, 5, 1)
        arr = np.random.rand(10, 5)
        initial_dataset = lgb.Dataset(arr)
        initial_dataset.construct()
        module._set_train_dat(initial_dataset)

        # bst is non-None now
        module.bst = object()

        with self.assertRaises(AssertionError):
            module._input_checking_setting(lgb.Dataset(np.random.random([10, 5])))

    def test_raises_input_changed_ndarray(self):
        """
        Test scenario where self.training = True, self.train_dat not None,
        self.bst not None, and input_dataset is an lgb.Dataset with a _handle.
        Expect no assertion error and return self.train_dat.
        """
        module = lgm.LGBModule(10, 5, 1)
        arr = np.random.rand(10, 5)
        initial_dataset = lgb.Dataset(arr)
        initial_dataset.construct()
        module._set_train_dat(initial_dataset)

        # bst is non-None now
        module.bst = object()

        with self.assertRaises(AssertionError):
            module._input_checking_setting(np.random.random([11, 5]))


class Wrapper(torch.nn.Module):
    def __init__(self, batch_size, input_dim, output_dim):
        super().__init__()
        self.model = lgm.LGBModule(batch_size, input_dim, output_dim)

    def forward(self, x):
        return self.model(x)

    def gb_step(self):
        self.model.gb_step()


class TestSaveLoad(TestCase):
    def test_lgbmodule_save_load_predictions(self):
        batch_size = 100
        input_dim = 3
        output_dim = 1
        np.random.seed(42)
        X = np.random.randn(batch_size, input_dim)
        Y = np.random.random([batch_size, output_dim])

        model = lgm.LGBModule(batch_size, input_dim, output_dim)
        MSE = torch.nn.MSELoss()
        model.train()
        loss = MSE(model(X), torch.Tensor(Y))
        loss.backward(create_graph=True)
        model.gb_step()
        output_before = model(X).detach().numpy()
        # Save
        state = model.state_dict()
        # Load into a new model
        model2 = lgm.LGBModule(batch_size, input_dim, output_dim)
        model2.load_state_dict(state)
        model2.eval()
        output_after = model2(X).detach().numpy()
        assert np.allclose(
            output_before, output_after
        ), "Outputs differ after save/load"

    def test_lgbmodule_save_load_submodule_predictions(self):
        batch_size = 100
        input_dim = 3
        output_dim = 1
        np.random.seed(42)
        X = np.random.randn(batch_size, input_dim)
        Y = np.random.random([batch_size, output_dim])

        wrapper = Wrapper(batch_size, input_dim, output_dim)
        MSE = torch.nn.MSELoss()
        wrapper.train()
        loss = MSE(wrapper(X), torch.Tensor(Y))
        loss.backward(create_graph=True)
        wrapper.gb_step()
        output_before = wrapper(X).detach().numpy()
        # Save
        state = wrapper.state_dict()
        # Load into a new wrapper
        wrapper2 = Wrapper(batch_size, input_dim, output_dim)
        wrapper2.load_state_dict(state)
        wrapper2.eval()
        output_after = wrapper2(X).detach().numpy()
        assert np.allclose(
            output_before, output_after
        ), "Outputs differ after save/load with submodule"
