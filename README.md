# gbnet

Gradient Boosting Modules for pytorch

## Introduction

Gradient Boosting Machines only require gradients and, for modern packages, hessians to train. Pytorch (and other neural network packages) calculates gradients and hessians. GBMs can therefore be fit as the first layer in neural networks using Pytorch. This package provides access to XGBoost and LightGBM as Pytorch Modules to do exactly this.

CatBoost is supported in an experimental capacity since the current gbnet integration with CatBoost is not as performant as the other GBDT packages.

## Install

`pip install gbnet`

## Troubleshooting

1. Currently, the biggest difference between training using `gbnet` vs basic `torch`, is that `gbnet`, like basic usage of `xgboost` and `lightgbm`, requires the entire dataset to be fed in. Cached predictions allow these packages to train quickly, and caching cannot happen if input batches change with each training/boosting round. Some additional info is provided in [#12](https://github.com/mthorrell/gbnet/issues/12).

## Basic training of a GBM for comparison to existing packages

```python
import time

import lightgbm as lgb
import numpy as np
import xgboost as xgb
import torch

from gbnet import lgbmodule, xgbmodule

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
    params={'objective': 'reg:squarederror', 'base_score': 0.0},
    dtrain=xgb.DMatrix(X, label=Y),
    num_boost_round=iters
)
t1 = time.time()

# LightGBM training for comparison
lbst = lgb.train(
    params={'verbose':-1},
    train_set=lgb.Dataset(X, label=Y.flatten(), init_score=[0 for i in range(n)]),
    num_boost_round=iters
)
t2 = time.time()

# XGBModule training
xnet = xgbmodule.XGBModule(n, input_dim, output_dim, params={})
xmse = torch.nn.MSELoss()

for i in range(iters):
    xnet.zero_grad()
    xpred = xnet(X)

    loss = 1/2 * xmse(xpred, torch.Tensor(Y))  # xgboost uses 1/2 (Y - P)^2
    loss.backward(create_graph=True)

    xnet.gb_step(X)
t3 = time.time()

# LGBModule training
lnet = lgbmodule.LGBModule(n, input_dim, output_dim, params={})
lmse = torch.nn.MSELoss()
for i in range(iters):
    lnet.zero_grad()
    lpred = lnet(X)

    loss = lmse(lpred, torch.Tensor(Y))
    loss.backward(create_graph=True)

    lnet.gb_step(X)
t4 = time.time()


print(np.max(np.abs(xbst.predict(xgb.DMatrix(X)) - xnet(X).detach().numpy().flatten())))  # 9.537e-07
print(np.max(np.abs(lbst.predict(X) - lnet(X).detach().numpy().flatten())))  # 2.479e-07
print(f'xgboost time: {t1 - t0}')   # 0.089
print(f'lightgbm time: {t2 - t1}')  # 0.084
print(f'xgbmodule time: {t3 - t2}') # 0.166
print(f'lgbmodule time: {t4 - t3}') # 0.123
```

## Training XGBoost and LightGBM together

```python
import time

import numpy as np
import torch

from gbnet import lgbmodule, xgbmodule


# Create new module that jointly trains multi-output xgboost and lightgbm models
# the outputs of these gbm models is then combined by a linear layer
class GBPlus(torch.nn.Module):
    def __init__(self, input_dim, intermediate_dim, output_dim):
        super(GBPlus, self).__init__()

        self.xgb = xgbmodule.XGBModule(n, input_dim, intermediate_dim, {'eta': 0.1})
        self.lgb = lgbmodule.LGBModule(n, input_dim, intermediate_dim, {'eta': 0.1})
        self.linear = torch.nn.Linear(intermediate_dim, output_dim)

    def forward(self, input_array):
        xpreds = self.xgb(input_array)
        lpreds = self.lgb(input_array)
        preds = self.linear(xpreds + lpreds)
        return preds

    def gb_step(self, input_array):
        self.xgb.gb_step(input_array)
        self.lgb.gb_step(input_array)

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

    gbp.gb_step(X)  # required to update the gbms
    optimizer.step()
t1 = time.time()
print(t1 - t0)  # 5.821
```

<img width="750" alt="image" src="https://github.com/mthorrell/gbmodule/assets/15166269/949c7000-7fc3-4600-8916-03cdf60eeeb8">
