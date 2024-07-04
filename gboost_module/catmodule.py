import catboost as cb
import numpy as np
import torch
from torch import nn


class CatModule(nn.Module):
    def __init__(
        self,
        batch_size,
        input_dim,
        output_dim,
        params={},
    ):
        super(CatModule, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.params = params
        self.params.update({'iterations': 1})
        self.bst = None

        self.FX = nn.Parameter(
            torch.tensor(
                np.zeros([batch_size, output_dim]),
                dtype=torch.float,
            )
        )

    def forward(self, input_array, return_tensor=True):
        assert isinstance(input_array, np.ndarray), "Input must be a numpy array"
        # TODO figure out how actual batch training works here
        if self.training:
            if self.bst and self.dtrain:
                preds = self.bst.predict(self.dtrain)
            else:
                preds = np.zeros([self.batch_size, self.output_dim])
        else:
            if self.bst:
                preds = self.bst.predict(input_array)
            else:
                preds = np.zeros(
                    [input_array.shape[0], self.output_dim], dtype=torch.float
                )

        if self.training:
            FX_detach = self.FX.detach()
            FX_detach.copy_(
                torch.tensor(
                    preds.reshape([self.batch_size, self.output_dim]), dtype=torch.float
                )
            )

        if return_tensor:
            if self.training:
                return self.FX
            else:
                return torch.tensor(
                    preds.reshape([-1, self.output_dim]), dtype=torch.float
                )
        return preds

    def gb_step(self, input_array):
        grad = self.FX.grad * self.batch_size

        # parameters are independent row by row, so we can
        # at least calculate hessians column by column by
        # considering the sum of the gradient columns
        hesses = []
        for i in range(self.output_dim):
            hesses.append(
                torch.autograd.grad(grad[:, i].sum(), self.FX, retain_graph=True)[0][
                    :, i : (i + 1)
                ]
            )
        hess = torch.cat(hesses, axis=1)

        obj = CatboostObj(grad, hess)
        input_params = self.params.copy()
        input_params.update(
            {
                "loss_function": obj,
            }
        )

        if self.bst is not None:
            new_bst = cb.CatBoostRegressor(**self.params)
            new_bst.fit(self.train_dat, init_model=self.bst)
            self.bst = new_bst
        else:
            self.train_dat = cb.Pool(
                data=input_array,
                labels=np.random.random([self.batch_size, self.output_dim]),
                weight=range(self.batch_size)  # hack to force custom obj to work correctly
            )
            self.bst = cb.CatBoostRegressor(**self.params)
            self.bst.fit(self.train_dat)


class CatboostObj(cb.MultiRegressionCustomObjective):
    def __init__(self, grad, hess):
        self.grad = grad.detach().numpy()
        self.hess = hess.detach().numpy()

    def calc_ders_multi(self, approxes, targets, weight):
        # TODO catboost works row by row in the multidim output setting.
        # I'm not sure if we can re-calculate grad and hess per row efficiently,
        # so, currently, this uses the `weight` input to identify the prediction
        # row being considered. The todo is to find a better way to do this.
        grad = -self.grad[weight, :]  # uses the negative gradient
        hess = -np.diag(self.hess[weight, :])
        return (grad, hess)
