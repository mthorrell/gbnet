import lightgbm as lgb
import numpy as np
import torch
from torch import nn


class LGBModule(nn.Module):
    def __init__(
        self,
        batch_size,
        input_dim,
        output_dim,
        params={},
    ):
        super(LGBModule, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.params = params
        self.bst = None

        self.FX = nn.Parameter(
            torch.tensor(
                np.zeros([batch_size, output_dim]),
                dtype=torch.float,
            )
        )
        self.train_dat = None

    def forward(self, input_array, return_tensor=True):
        assert isinstance(input_array, np.ndarray), "Input must be a numpy array"
        # TODO figure out how actual batch training works here
        if self.training:
            if self.bst:
                preds = self.bst._Booster__inner_predict(0).copy()
            else:
                preds = np.zeros([self.batch_size, self.output_dim])
        else:
            if self.bst:
                preds = self.bst.predict(input_array).copy()
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

        obj = LightGBObj(grad, hess)
        input_params = self.params.copy()
        input_params.update(
            {
                "objective": obj,
                "num_class": self.output_dim,
                "verbose": -1,
                "verbosity": -1,
            }
        )

        if self.bst is not None:
            self.bst.update(train_set=self.train_dat, fobj=obj)
        else:
            self.train_dat = lgb.Dataset(
                input_array,
                params={"verbose": -1},
            )
            self.bst = lgb.train(
                params=input_params,
                train_set=self.train_dat,
                num_boost_round=1,
                keep_training_booster=True,
            )


class LightGBObj:
    def __init__(self, grad, hess):
        self.grad = grad.detach().numpy()
        self.hess = hess.detach().numpy()

    def __call__(self, y_true, y_pred):
        return self.grad, self.hess
