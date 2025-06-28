.. GBNet documentation master file, created by
   sphinx-quickstart on Mon Mar  3 22:00:51 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

GBNet Documentation
==================

GBNet is a Python library that provides XGBoost and LightGBM PyTorch Modules.

Key Features
-----------

- XGBoost and LightGBM PyTorch Modules
- Linear Module that updates using Gradient Boosting for improved performance
- Specific model implementations using XGBoost and LightGBM Modules
   - Forecasting
   - Ordinal Regression

Installation
-----------

Install GBNet using pip:

.. code-block:: bash

   pip install gbnet

Additional troubleshooting details are available in the `Overview <overview.html>`_.

Quick Start
----------

Here's a simple example of using XGBModule to mimic standard XGBoost.

.. code-block:: python

   from gbnet import xgbmodule
   
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

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   overview
   modules/forecasting
   examples/index
   api/index
