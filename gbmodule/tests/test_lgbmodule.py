import numpy as np
import torch

from gbmodule import lgbmodule as lgm


def test_LightGBObj():
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
