# GBNet
[![DOI](https://joss.theoj.org/papers/10.21105/joss.08047/status.svg)](https://doi.org/10.21105/joss.08047)



Pytorch Modules for XGBoost and LightGBM

## Table of Contents

1. [Introduction](#introduction)
2. [Install and Docs](#install-and-docs)
3. [Pytorch Modules](#pytorch-modules)
   - [Conceptually, how can Pytorch be used to fit XGBoost or LightGBM models?](#conceptually-how-can-pytorch-be-used-to-fit-xgboost-or-lightgbm-models)
   - [Is training a `gbnet` model closer to training a neural network or to training a GBM?](#is-training-a-gbnet-model-closer-to-training-a-neural-network-or-to-training-a-gbm)
   - [Basic training of a GBM for comparison to existing gradient boosting packages](#basic-training-of-a-gbm-for-comparison-to-existing-gradient-boosting-packages)
   - [Training XGBoost and LightGBM together](#training-xgboost-and-lightgbm-together)
4. [Models](#models)
   - [Forecasting](#forecasting)
   - [Ordinal Regression](#ordinal-regression)
   - [Discrete Beta Survival](#discrete-beta-survival)
5. [Contributing](#contributing)
6. [Cite this work](#cite-this-work)



## Introduction

XGBoost and LightGBM are industry-standard gradient boosting packages used to solve tabular data machine learning problems. Users of these packages wishing to define custom loss functions, novel architectures, or other advanced modeling scenarios, however, may face substantial difficulty due to potentially complex gradient and Hessian calculations required by both XGBoost and LightGBM. GBNet provides PyTorch Modules wrapping XGBoost and LightGBM so that users can construct and fit nearly arbitrary model architectures involving XGBoost or LightGBM without requiring users to provide gradient and Hessian calculations. PyTorch's autograd system calculates derivative information automatically; GBNet orchestrates delivery of that information back to the boosting algorithms. GBNet, by linking XGBoost and LightGBM to PyTorch, expands the set of applications for gradient boosting models.

There are two main components of `gbnet`:

- (1) `gbnet.xgbmodule`, `gbnet.lgbmodule` and `gbnet.gblinear` provide the Pytorch Modules that allow fitting of XGBoost, LightGBM and Boosted Linear models using Pytorch's computational network and differentiation capabilities.

  - For example, if $`F(X)`$ is the output of an XGBoost model, you can use Pytorch to define the loss function, $`L(y, F(X))`$. Pytorch handles the gradients of $`L`$ so, as a user, you only specify the loss function.
  - You can also fit two (or more) boosted models together with Pytorch-supported parametric components. For instance, a recommendation prediction might look like this: $`\sigma(F(user) \times G(item))`$ where both $`F`$ and $`G`$ are separate boosting models producing embeddings of users and items respectively. `gbnet` makes defining and fitting such a model almost as easy as using Pytorch itself.

- (2) `gbnet.models` provides specific example estimators that accomplish things that were not previously possible using only XGBoost or LightGBM. Current models:
  - `Forecast` is a forecasting model similar in execution to Metas' Prophet algorithm. In the settings we tested, `gbnet.models.forecasting.Forecast` beats the performance of Meta's Prophet algorithm (see [the forecasting PR](https://github.com/mthorrell/gbnet/pull/20) for a comparison).
  - `GBOrd` is Ordinal Regression using GBMs (both XGBoost and LightGBM supported). The complex loss function (with fitable parameters) is specified in PyTorch and put on top of either `XGBModule` or `LGBModule`.
  - `BetaSurvivalModel` is a discrete-time survival analysis model using Beta distributions with gradient boosting.
  - `ThetaSurvivalModel` is a discrete-time survival model that parameterizes a geometric distribution via a parameter theta that is the output of a a GBM.
  - Other models with plans to be integrated are more advanced survival analysis and NLP applications.

## Install and Docs

`pip install gbnet`

### Troubleshooting and dependencies

Use of virtual environments/conda is best practice when installing GBNet. GBNet requires XGBoost, LightGBM and PyTorch as key dependencies and may use these packages simultaneously. Each of these packages rely on OpenMP implementations for parallelization. Conflicts in the OpenMP implementations will throw warnings and may produce slow or incorrect outputs. Prior to installing these python dependencies, it is best to ensure each of these dependencies point to a single OpenMP implementation. Apple Silicon users may prefer to install `libomp` via `brew` prior to the python package dependency installations (see, for example, [build notes](https://xgboost.readthedocs.io/en/stable/build.html#running-cmake-and-build) for XGBoost for additional details).

### Docs
https://gbnet.readthedocs.io/

## Pytorch Modules

There are currently three Pytorch Modules in `gbnet`: `lgbmodule.LGBModule`, `xgbmodule.XGBModule` and `gblinear.GBLinear`. These create the interface between PyTorch and the boosting algorithms. LightGBM and XGBoost are wrapped in `LGBModule` and `XGBModule` respectively. `GBLinear` is a linear layer that is trained with boosting (rather than gradient descent) -- for some applications it trains much faster than gradient descent (see this [PR](https://github.com/mthorrell/gbnet/pull/60) for details).

### Conceptually, how can Pytorch be used to fit XGBoost or LightGBM models?

Gradient Boosting Machines only require gradients and, for modern packages, hessians to train. Pytorch (and other neural network packages) calculates gradients and hessians. GBMs can therefore be fit as the first layer in neural networks using Pytorch.

CatBoost is also supported but in an experimental capacity since the current gbnet integration with CatBoost is not as performant as the other GBDT packages.

### Is training a `gbnet` model closer to training a neural network or to training a GBM?

It's closer to training a GBM. Currently, the biggest difference between training using `gbnet` vs basic `torch`, is that `gbnet`, like basic usage of `xgboost` and `lightgbm`, requires the entire dataset to be fed in. Cached predictions allow these packages to train quickly, and caching cannot happen if input batches change with each training/boosting round. There are some ways around this but there is currently no native functionality in `gbnet` for true batch training. Additional info is provided in [#12](https://github.com/mthorrell/gbnet/issues/12).

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

X_dmatrix = xgb.DMatrix(X)
for i in range(iters):
    xnet.zero_grad()
    xpred = xnet(X_dmatrix)

    loss = 1/2 * xmse(xpred, torch.Tensor(Y))  # xgboost uses 1/2 (Y - P)^2
    loss.backward(create_graph=True)

    xnet.gb_step()
xnet.eval()  # like any torch module, use eval mode for predictions
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
lnet.eval()  # use eval mode for predictions
t4 = time.time()

print(np.max(np.abs(xbst.predict(xgb.DMatrix(X)) - xnet(X_dmatrix).detach().numpy().flatten())))  # 9.537e-07
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
print(t1 - t0)  # 5.821
```

<img width="500" alt="image" src="https://github.com/mthorrell/gbmodule/assets/15166269/949c7000-7fc3-4600-8916-03cdf60eeeb8">


## Models

### Forecasting

`gbnet.models.forecasting.Forecast` outperforms Meta's popular Prophet algorithm on basic benchmarks (see [the forecasting PR](https://github.com/mthorrell/gbnet/pull/20) for a comparison). Starter comparison code:

```python
import pandas as pd
from prophet import Prophet
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
test['gbnet_pred'] = gbnet_forecast_model.predict(test)['yhat']

# prophet
prophet_model = Prophet()
prophet_model.fit(train)
test['prophet_pred'] = prophet_model.predict(test)['yhat']

sel = test['y'].notnull()
print(f"gbnet rmse: {root_mean_squared_error(test[sel]['y'], test[sel]['gbnet_pred'])}")
print(f"prophet rmse: {root_mean_squared_error(test[sel]['y'], test[sel]['prophet_pred'])}")

# gbnet rmse: 8.757314439339462
# prophet rmse: 20.10509806878121
```

### Ordinal Regression

See [this notebook](https://github.com/mthorrell/gbnet/blob/main/examples/ordinal_regression_comparison.ipynb) for examples.

```python
from gbnet.models import ordinal_regression

sklearn_estimator = ordinal_regression.GBOrd(num_classes=10)
```

### Discrete Beta Survival

`gbnet.models.survival.discrete_survival.BetaSurvivalModel` provides discrete survival analysis using Beta distributions with gradient boosting. This model can handle censored data and supports both XGBoost and LightGBM backends. See [this notebook](https://github.com/mthorrell/gbnet/blob/main/examples/discrete_beta_survival_example.ipynb) for an example usage. This is an implementation of the model described in this [paper](https://proceedings.mlr.press/v146/hubbard21a.html).

```python
from gbnet.models.survival import discrete_survival

# Load survival data (time, event)
survival_model = discrete_survival.BetaSurvivalModel()
survival_model.fit(X, y)  # y should have 'time' and 'event' columns

# Predict survival probabilities
survival_probs = survival_model.predict_survival(X, times=[1, 5, 10])
```

### Theta Survival

`gbnet.models.survival.discrete_survival.ThetaSurvivalModel` models discrete survival with a geometric distribution (via a learned theta). It is a lightweight alternative to the Beta-based model.

## Contributing

Contributions are welcome! Here are some ways you can help:

- Report bugs and request features by opening issues
- Submit pull requests with bug fixes or new features
- Improve documentation and examples
- Add tests to increase code coverage

Before submitting a pull request:

1. Fork the repository and create a new branch
2. Add tests for any new functionality
3. Ensure all tests pass by running `pytest`
4. Update documentation as needed
5. Follow the existing code style

For major changes, please open an issue first to discuss what you would like to change.


## Cite this work

```
Horrell, M., (2025). GBNet: Gradient Boosting packages integrated into PyTorch. Journal of Open Source Software, 10(111), 8047, https://doi.org/10.21105/joss.08047
```
