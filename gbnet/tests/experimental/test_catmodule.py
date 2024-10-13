import numpy as np
import torch
from catboost import CatBoostRegressor
from gbnet.experimental import catmodule


def test_basic_catboost_example():
    np.random.seed(100)
    n = 10
    xs = np.random.random([n, 4])
    ys = np.random.random([n, 2])
    niters = 10

    model = CatBoostRegressor(
        iterations=niters,
        learning_rate=0.03,
        boost_from_average=False,
        loss_function="MultiRMSE",
    )
    model.fit(xs, ys)
    pred = model.predict(xs)

    cbnet = catmodule.CatModule(
        xs.shape[0],
        xs.shape[1],
        ys.shape[1],
        params={
            "learning_rate": 0.03,
        },
    )
    cbmse = torch.nn.MSELoss()
    for i in range(niters):
        cbnet.zero_grad()
        cbpred = cbnet(xs)

        loss = cbmse(cbpred, torch.Tensor(ys))
        loss.backward(create_graph=True)

        cbnet.gb_step(xs)
    cbnet.eval()

    assert np.isclose(pred, cbnet(xs).detach().numpy()).all()
