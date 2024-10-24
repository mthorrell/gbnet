# gbnet

Gradient boosting libraries integrated with Pytorch

## Install

`pip install gbnet`

## Introduction

There are two main components of `gbnet`:
- (1) `gbnet` provides the Pytorch Modules that allow fitting of XGBoost and/or LightGBM models using Pytorch's computational network and differentiation capabilities.
    - For example, if $`F(X)`$ is the output of an XGBoost model, you can use Pytorch to define the loss function, $`L(y, F(X))`$. Pytorch handles the gradients of $`L`$ so, as a user, you only specify the loss function.
    - You can also fit two (or more) boosted models together with Pytorch-supported parametric components.  For instance, a recommendation prediction might look like this: $`\sigma(F(user) \times G(item))`$ where both $`F`$ and $`G`$ are separate boosting models producing embeddings of users and items respectively. `gbnet` makes defining and fitting such a model almost as easy as using Pytorch itself.

- (2) `gbnet` provides specific example estimators that accomplish things that were not previously possible using only XGBoost or LightGBM.
    - You can find these estimators in `gbnet/models/`. Right now there is a forecasting model that in the settings we tested, beats the performance of Meta's Prophet algorithm (see [the forecasting PR](https://github.com/mthorrell/gbnet/pull/20) for a comparison).
    - Other models with plans to be integrated are Ordinal Regression and Time-varying Survival analysis.

## Models

### Forecasting

`gbnet.models.forecasting.Forecast` outperforms Meta's popular Prophet algorithm on basic benchmarks (see [the forecasting PR](https://github.com/mthorrell/gbnet/pull/20) for a comparison).  Starter comparison code:

```python
import pandas as pd
from sklearn.metrics import root_mean_squared_error

from gbnet.models import forecasting

## Load and split data
url = "https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv"
df = pd.read_csv(url)
df['ds'] = pd.to_datetime(df['ds'])

train = df[df['ds'] < df['ds'].median()].reset_index(drop=True).copy()
test = df[df['ds'] >= df['ds'].median()].reset_index(drop=True).copy()

## train and predict comparing out-of-the-box gbnet & prophet

# gbnet
gbnet_forecast_model = forecasting.Forecast()
gbnet_forecast_model.fit(train, train['y'])
test['gbnet_pred'] = gbnet_forecast_model.predict(test)

# prophet
prophet_model = Prophet()
prophet_model.fit(train)
test['prophet_pred'] = prophet_model.predict(test)['yhat']

sel = test['y'].notnull()
print(f"gbnet rmse: {root_mean_squared_error(test[sel]['y'], test[sel]['gbnet_pred'])}")
print(f"prophet rmse: {root_mean_squared_error(test[sel]['y'], test[sel]['prophet_pred'])}")

# gbnet rmse: 11.036844941272257
# prophet rmse: 20.10509806878121
```

## Pytorch Modules

There are currently just two Pytorch Modules: `lgbmodule.LGBModule` and `xgbmodule.XGBModule`. These create the interface between Pytorch and the LightGBM and XGBoost packages respectively.

### Conceptually, how can Pytorch be used to fit XGBoost or LightGBM models?

Gradient Boosting Machines only require gradients and, for modern packages, hessians to train. Pytorch (and other neural network packages) calculates gradients and hessians. GBMs can therefore be fit as the first layer in neural networks using Pytorch.

CatBoost is also supported but in an experimental capacity since the current gbnet integration with CatBoost is not as performant as the other GBDT packages.



### Is training a `gbnet` model closer to training a neural network or to training a GBM?

It's closer to training a GBM. Currently, the biggest difference between training using `gbnet` vs basic `torch`, is that `gbnet`, like basic usage of `xgboost` and `lightgbm`, requires the entire dataset to be fed in. Cached predictions allow these packages to train quickly, and caching cannot happen if input batches change with each training/boosting round. Some additional info is provided in [#12](https://github.com/mthorrell/gbnet/issues/12).

### Basic training of a GBM for comparison to existing gradient boosting packages

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

### Training XGBoost and LightGBM together

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
