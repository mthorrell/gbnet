import numpy as np

import torch
import torch.nn as nn


class GBLinear(nn.Module):
    def __init__(
        self, batch_size, input_dim, output_dim, bias=True, lr=0.1, min_hess=0.0
    ):
        super(GBLinear, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.min_hess = min_hess
        self.bias = bias
        self.lr = lr

        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=self.bias)
        self.FX = None  # Output tensor
        self.input = None

    def forward(self, x):
        self.input = x.detach().numpy()  # TODO add input checks
        self.FX = self.linear(x)
        self.FX.retain_grad()
        return self.FX

    def _get_grad_hess_FX(self, FX):
        grad = FX.grad * self.batch_size

        # parameters are independent row by row, so we can
        # at least calculate hessians column by column by
        # considering the sum of the gradient columns
        hesses = []
        for i in range(self.output_dim):
            hesses.append(
                torch.autograd.grad(grad[:, i].sum(), FX, retain_graph=True)[0][
                    :, i : (i + 1)
                ]
            )
        hess = torch.maximum(torch.cat(hesses, axis=1), torch.Tensor([self.min_hess]))
        return grad, hess

    def gb_step(self):
        """Update weights and bias based on the gradient of FX."""
        if self.FX is None or self.FX.grad is None:
            raise RuntimeError("Backward must be called before gb_step.")

        g, h = self._get_grad_hess_FX(self.FX)

        with torch.no_grad():
            if self.bias:
                X = np.concatenate([np.ones([self.batch_size, 1]), self.input], axis=1)
            else:
                X = self.input

            updated_B = np.linalg.lstsq(X, g / h, rcond=None)[0]
            updated_B.shape

            updated_weight_dir = updated_B[1:, :].T
            self.linear.weight -= self.lr * torch.Tensor(updated_weight_dir)

            if self.bias:
                updated_bias_dir = updated_B[0:1, :].flatten()
                self.linear.bias -= self.lr * torch.Tensor(updated_bias_dir)
