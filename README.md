# gbmodule

Gradient Boosting Modules for pytorch

# introduction

Gradient Boosting Machines only requires gradients and, for modern packages, hessians to train. Pytorch (and other neural network packages) calculates gradients and hessians. GBMs can therefore be fit as the first layer in neural networks using Pytorch. This package provides access to XGBoost and LightGBM as Pytorch Modules to do exactly this.

# install

Clone and pip install.

# basic training of a GBM with this pacakge
