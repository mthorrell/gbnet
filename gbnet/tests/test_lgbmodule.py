from unittest import mock

import lightgbm as lgb
import numpy as np
import torch
from gbnet import lgbmodule as lgm


def test_basic_loss():
    gbm = lgm.LGBModule(5, 3, 1, params={"min_data_in_leaf": 0})
    floss = torch.nn.MSELoss()

    gbm.zero_grad()
    np.random.seed(11010)
    input_array = np.random.random([5, 3])
    preds = gbm(input_array)
    loss = floss(preds.flatten(), torch.Tensor(np.array([1, 2, 3, 4, 5])).flatten())

    loss.backward(create_graph=True)

    m_obj = mock.MagicMock(side_effect=lgm.LightGBObj)
    m_Dataset = mock.MagicMock(side_effect=lgb.Dataset)
    m_train = mock.MagicMock(side_effect=lgb.train)
    with (
        mock.patch("gbnet.lgbmodule.LightGBObj", m_obj),
        mock.patch("lightgbm.Dataset", m_Dataset),
        mock.patch("lightgbm.train", m_train),
    ):
        gbm.gb_step(input_array)

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

    m_Dataset.assert_called_once_with(input_array, params={"verbose": -1})
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
