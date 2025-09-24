import time

import lightgbm as lgb
import numpy as np
import torch
import xgboost as xgb

from gbnet import lgbmodule, xgbmodule


def test_readme_examples():
    # Generate Dataset
    np.random.seed(100)
    n = 1000
    input_dim = 20
    output_dim = 1
    X = np.random.random([n, input_dim])
    B = np.random.random([input_dim, output_dim])
    Y = X.dot(B) + np.random.random([n, output_dim])

    iters = 100
    t0 = time.time()

    # XGBoost training for comparison
    xbst = xgb.train(
        params={"objective": "reg:squarederror", "base_score": 0.0},
        dtrain=xgb.DMatrix(X, label=Y),
        num_boost_round=iters,
    )
    t1 = time.time()

    # LightGBM training for comparison
    lbst = lgb.train(
        params={"verbose": -1},
        train_set=lgb.Dataset(X, label=Y.flatten(), init_score=[0 for i in range(n)]),
        num_boost_round=iters,
    )
    t2 = time.time()

    # XGBModule training
    xnet = xgbmodule.XGBModule(n, input_dim, output_dim, params={})
    xmse = torch.nn.MSELoss()

    X_dmatrix = xgb.DMatrix(X)
    for i in range(iters):
        xnet.zero_grad()
        xpred = xnet(X_dmatrix)

        loss = 1 / 2 * xmse(xpred, torch.Tensor(Y))  # xgboost uses 1/2 (Y - P)^2
        loss.backward(create_graph=True)

        xnet.gb_step()
    xnet.eval()
    t3 = time.time()

    # LGBModule training
    lnet = lgbmodule.LGBModule(n, input_dim, output_dim, params={})
    lmse = torch.nn.MSELoss()

    X_dataset = lgb.Dataset(X)
    for i in range(iters):
        lnet.zero_grad()
        lpred = lnet(X_dataset)

        loss = lmse(lpred, torch.Tensor(Y))
        loss.backward(create_graph=True)

        lnet.gb_step()
    lnet.eval()
    t4 = time.time()

    assert np.isclose(
        0.0,
        np.max(
            np.abs(
                xbst.predict(xgb.DMatrix(X))
                - xnet(X_dmatrix).detach().numpy().flatten()
            )
        ),
        atol=1e-05,
    )
    assert np.isclose(
        0.0,
        np.max(np.abs(lbst.predict(X) - lnet(X).detach().numpy().flatten())),
        atol=1e-05,
    )
    assert (t3 - t2) / (t1 - t0) < 4
    assert (t4 - t3) / (t2 - t1) < 4


def test_combine_example():
    # Create new module that jointly trains multi-output xgboost and lightgbm models
    # the outputs of these gbm models is then combined by a linear layer
    class GBPlus(torch.nn.Module):
        def __init__(self, input_dim, intermediate_dim, output_dim):
            super(GBPlus, self).__init__()

            self.xgb = xgbmodule.XGBModule(n, input_dim, intermediate_dim, {"eta": 0.1})
            self.lgb = lgbmodule.LGBModule(n, input_dim, intermediate_dim, {"eta": 0.1})
            self.linear = torch.nn.Linear(intermediate_dim, output_dim)

        def forward(self, input_array):
            xpreds = self.xgb(input_array)
            lpreds = self.lgb(input_array)
            preds = self.linear(xpreds + lpreds)
            return preds

        def gb_step(self):
            self.xgb.gb_step()
            self.lgb.gb_step()

    # Generate Dataset
    np.random.seed(100)
    n = 1000
    input_dim = 10
    output_dim = 1
    X = np.random.random([n, input_dim])
    B = np.random.random([input_dim, output_dim])
    Y = X.dot(B) + np.random.random([n, output_dim])

    intermediate_dim = 10
    gbp = GBPlus(input_dim, intermediate_dim, output_dim)
    mse = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(gbp.parameters(), lr=0.005)

    t0 = time.time()
    losses = []
    for i in range(100):
        optimizer.zero_grad()
        preds = gbp(X)

        loss = mse(preds, torch.Tensor(Y))
        loss.backward(create_graph=True)  # create_graph=True required for any gbnet
        losses.append(loss.detach().numpy().copy())

        gbp.gb_step()  # required to update the gbms
        optimizer.step()
    t1 = time.time()
    assert losses[-1] < 1e-05
    assert (t1 - t0) < 20
