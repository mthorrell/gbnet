from typing import Union

import numpy as np
import pandas as pd

from scipy.linalg import cho_solve, cho_factor
import torch
import torch.nn as nn

from gbnet.base import BaseGBModule


class GBLinear(BaseGBModule):
    """A linear gradient boosting module that uses gradient boosting for updates.

    This module implements a linear layer that can be trained using gradient boosting.
    It maintains state between iterations and updates parameters using computed gradients
    and hessians.

    Parameters
    ----------
    input_dim : int
        Input feature dimension
    output_dim : int
        Output prediction dimension
    bias : bool, optional
        Whether to include a bias term. Defaults to True.
    lr : float, optional
        Learning rate for parameter updates. Defaults to 0.5.
    min_hess : float, optional
        Minimum hessian threshold. Defaults to 0.0.
    lambd : float, optional
        L2 regularization parameter. Defaults to 0.01.

    Attributes
    ----------
    linear : nn.Linear
        The underlying linear layer
    FX : torch.Tensor
        Current predictions tensor
    input : numpy.ndarray
        Input data cache
    g : torch.Tensor
        Gradient cache
    h : torch.Tensor
        Hessian cache
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bias: bool = True,
        lr: float = 0.5,
        min_hess: float = 0.0,
        lambd: float = 0.01,
    ) -> None:
        super(GBLinear, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_hess = min_hess
        self.bias = bias
        self.lr = lr
        self.lambd = lambd

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=self.bias)
        self.FX = None
        self.input = None
        self.g = None
        self.h = None

    def _input_checking_setting(self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        assert isinstance(x, (torch.Tensor, np.ndarray, pd.DataFrame))

        if isinstance(x, np.ndarray):
            x = torch.Tensor(x)
        if isinstance(x, pd.DataFrame):
            x = torch.Tensor(np.array(x))

        if self.training:
            self.input = x.detach().numpy()  # TODO add input checks

        return x

    def forward(self, x: Union[torch.Tensor, np.ndarray, pd.DataFrame]):
        x = self._input_checking_setting(x)

        self.FX = self.linear(x)
        if self.training:
            self.FX.retain_grad()
        return self.FX

    def gb_calc(self):
        """Calculate gradients and stores in the object"""
        if self.FX is None or self.FX.grad is None:
            raise RuntimeError("Backward must be called before gb_step.")

        self.g, self.h = self._get_grad_hess_FX()

    def gb_step(self):
        """Uses stored gradients to update weights"""
        if self.g is None and self.h is None:
            self.gb_calc()

        with torch.no_grad():
            if self.bias:
                X = np.concatenate(
                    [np.ones([self.input.shape[0], 1]), self.input], axis=1
                )
            else:
                X = self.input

            updated_B = ridge_regression(
                X, (self.g / self.h).detach().numpy(), self.lambd
            )

            updated_weight_dir = updated_B[1:, :].T
            self.linear.weight -= self.lr * torch.Tensor(updated_weight_dir)

            if self.bias:
                updated_bias_dir = updated_B[0:1, :].flatten()
                self.linear.bias -= self.lr * torch.Tensor(updated_bias_dir)
        self.g = None
        self.h = None


def ridge_regression(X, y, lambd):
    """Solves ridge regression using Cholesky decomposition.

    Fastest method tested.

    Args:
        X (np.ndarray): Design matrix of shape (n_samples, n_features)
        y (np.ndarray): Target values of shape (n_samples,)
        lambd (float): Ridge regularization parameter

    Returns:
        np.ndarray: Fitted coefficients of shape (n_features,)

    The function solves the ridge regression problem:
    min_beta ||X beta - y||^2 + lambd ||beta||^2
    using the normal equations and Cholesky decomposition for numerical stability.
    """
    n, d = X.shape
    A = X.T @ X + lambd * np.eye(d)
    c = X.T @ y
    L = cho_factor(A)
    beta = cho_solve(L, c)
    return beta
