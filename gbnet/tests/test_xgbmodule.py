from unittest import mock

import numpy as np
import torch
import xgboost as xgb
from gbnet import xgbmodule as xgm


def test_basic_loss():
    gbm = xgm.XGBModule(5, 3, 1)
    floss = torch.nn.MSELoss()

    gbm.zero_grad()
    np.random.seed(11010)
    input_array = np.random.random([5, 3])
    preds = gbm(input_array)
    loss = floss(preds.flatten(), torch.Tensor(np.array([1, 2, 3, 4, 5])).flatten())

    loss.backward(create_graph=True)

    with (
        mock.patch("gbnet.xgbmodule.XGBObj", side_effect=xgm.XGBObj) as m_obj,
        mock.patch("xgboost.DMatrix", side_effect=xgb.DMatrix) as m_DMatrix,
        mock.patch.object(gbm.bst, "boost", side_effect=gbm.bst.boost) as m_boost,
    ):
        gbm.gb_step(input_array)

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

    assert np.all(np.isclose(m_DMatrix.call_args_list[-1].args[0], input_array))
    assert np.all(np.isclose(m_DMatrix.call_args_list[-1].kwargs["label"], np.zeros(5)))
    m_boost.assert_called_once()


def test_XGBObj():
    np.random.seed(10101)
    grad = torch.tensor(np.random.random([20, 10]))
    hess = torch.tensor(np.random.random([20, 10]))

    obj = xgm.XGBObj(grad, hess)

    ograd, ohess = obj(grad, hess)
    assert np.all(np.isclose(ograd, grad.detach().numpy().reshape([-1, 1])))
    assert np.all(np.isclose(ohess, hess.detach().numpy().reshape([-1, 1])))
