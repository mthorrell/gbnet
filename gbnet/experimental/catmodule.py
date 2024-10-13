import catboost as cb
import numpy as np
import torch
from torch import nn


class CatModule(nn.Module):
    """
    CatModule mirrors LGBModule and XGBModule but uses catboost as the gbdt
    implementation.

    Currently this is in the experimental section of the code because, while
    this module generally gives results as expected, the catboost connection
    cannot yet work the same as the GBDT connections in LGBModule and
    XGBModule; thus, using CatModule is a fair amount slower compared to the
    other modules. If you have ideas here, please file an issue :)
    """

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
        self.params.update(
            {
                "iterations": 1,
                "eval_metric": ("MultiRMSE" if self.output_dim > 1 else "RMSE"),
                "boost_from_average": False,
            }
        )

        self.bst = None
        self.train_dat = None
        self.obj = None

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
            if self.bst and self.train_dat:
                preds = self.bst.predict(self.train_dat)
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

        self.grad_index = 0
        if self.bst is not None:
            self.obj.reset(grad, hess, self.batch_size)
            self.bst.fit(self.train_dat, init_model=self.bst)

        else:
            self.obj = CatboostObj(grad, hess, self.batch_size)
            input_params = self.params.copy()
            input_params.update({"loss_function": self.obj, "allow_const_label": True})
            self.train_dat = cb.Pool(
                data=input_array, label=np.zeros([self.batch_size, self.output_dim])
            )
            self.bst = cb.CatBoostRegressor(**input_params)
            self.bst.fit(self.train_dat)


class CatboostObj(cb.MultiTargetCustomObjective):
    def __init__(self, grad, hess, n):
        self.grad = grad.detach().numpy()
        self.hess = hess.detach().numpy()
        self.call_index = 0
        self.n = n

    def reset(self, grad, hess, n):
        self.grad = grad.detach().numpy()
        self.hess = hess.detach().numpy()
        self.call_index = 0
        self.n = n

    def calc_ders_multi(self, approxes, targets, weight):
        # TODO catboost works row by row in the multidim output setting.
        # I'm not sure if we can re-calculate grad and hess per row efficiently,
        # so, currently, this relies on the objective function being called
        # sequentially, a very brittle assumption.
        grad = -self.grad[self.call_index % self.n, :]  # uses the negative gradient
        hess = -np.diag(self.hess[self.call_index % self.n, :])
        self.call_index += 1
        return (grad, hess)
