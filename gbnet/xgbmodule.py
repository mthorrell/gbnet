import numpy as np
import torch
import xgboost as xgb
from torch import nn


class XGBModule(nn.Module):
    def __init__(
        self,
        batch_size,
        input_dim,
        output_dim,
        params={},
    ):
        super(XGBModule, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.params = params
        self.params["objective"] = "reg:squarederror"
        self.params["base_score"] = 0
        self.n_completed_boost_rounds = 0

        init_matrix = np.zeros([batch_size, input_dim])
        self.bst = xgb.train(
            self.params,
            xgb.DMatrix(init_matrix, label=np.zeros(batch_size * output_dim)),
            num_boost_round=0,
        )
        self.n_completed_boost_rounds = 0
        self.dtrain = None

        self.FX = nn.Parameter(
            torch.tensor(
                np.zeros([batch_size, output_dim]),
                dtype=torch.float,
            )
        )

    def forward(self, input_dmatrix, return_tensor=True):
        assert isinstance(input_dmatrix, xgb.DMatrix), "Input must be an xgb.DMatrix"
        # TODO figure out actual batch training
        if self.training:
            if self.dtrain is None:
                input_dmatrix.set_label(np.zeros(self.batch_size * self.output_dim))
                self.dtrain = input_dmatrix
            preds = self.bst.predict(self.dtrain)
        else:
            preds = self.bst.predict(input_dmatrix)

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

    def gb_step(self):
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

        obj = XGBObj(grad, hess)
        g, h = obj(np.zeros([self.batch_size, self.output_dim]), None)

        if xgb.__version__ <= "2.0.3":
            self.bst.boost(
                self.dtrain,
                g,
                h,
            )
        else:
            self.bst.boost(
                self.dtrain,
                self.n_completed_boost_rounds + 1,
                g,
                h,
            )
        self.n_completed_boost_rounds = self.n_completed_boost_rounds + 1


class XGBObj:
    def __init__(self, grad, hess):
        self.grad = grad
        self.hess = hess

    def __call__(self, preds, dtrain):
        if len(preds.shape) == 2:
            M = preds.shape[0]
            N = preds.shape[1]
        else:
            M = preds.shape[0]
            N = 1

        if xgb.__version__ >= "2.1.0":
            g = self.grad.detach().numpy().reshape([M, N])
            h = self.hess.detach().numpy().reshape([M, N])
        else:
            g = self.grad.detach().numpy().reshape([M * N, 1])
            h = self.hess.detach().numpy().reshape([M * N, 1])

        return g, h
