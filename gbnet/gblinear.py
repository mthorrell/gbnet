from typing import Union

import numpy as np
import pandas as pd

from scipy.linalg import cho_solve, cho_factor
import torch
import torch.nn as nn


class GBLinear(nn.Module):
    def __init__(
        self,
        batch_size,
        input_dim,
        output_dim,
        bias=True,
        lr=0.1,
        min_hess=0.0,
        lambd=0.01,
    ):
        super(GBLinear, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_hess = min_hess
        self.bias = bias
        self.lr = lr
        self.lambd = lambd

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=self.bias)
        self.FX = None
        self.input = None

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

    def _get_grad_hess_FX(self, FX):
        grad = FX.grad * self.batch_size

        hesses = []
        for i in range(self.output_dim):
            hesses.append(
                torch.autograd.grad(grad[:, i].sum(), FX, retain_graph=True)[0][
                    :, i : (i + 1)
                ]
            )
        hess = torch.maximum(torch.cat(hesses, axis=1), torch.Tensor([self.min_hess]))
        return grad, hess

    def gb_calc(self):
        """Calculate gradients and stores in the object"""
        if self.FX is None or self.FX.grad is None:
            raise RuntimeError("Backward must be called before gb_step.")

        self.g, self.h = self._get_grad_hess_FX(self.FX)

    def gb_step(self):
        """Uses stored gradients to update weights"""
        with torch.no_grad():
            if self.bias:
                X = np.concatenate([np.ones([self.batch_size, 1]), self.input], axis=1)
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


def ridge_regression(X, y, lambd):
    n, d = X.shape
    A = X.T @ X + lambd * np.eye(d)
    c = X.T @ y
    L = cho_factor(A)
    beta = cho_solve(L, c)
    return beta
