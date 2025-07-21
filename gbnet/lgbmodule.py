from typing import Union
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch import nn

from gbnet.base import BaseGBModule


class LGBModule(BaseGBModule):
    """LightGBM Module that wraps LightGBM boosting into a PyTorch Module.

    This module allows integration of LightGBM gradient boosting with PyTorch neural networks.
    It maintains the boosting model state and handles both training and inference.

    Args:
        batch_size (int): Size of training data
        input_dim (int): Dimension of input features
        output_dim (int): Dimension of output predictions
        params (dict, optional): Parameters passed to LightGBM. Defaults to {}.
        min_hess (float, optional): Minimum hessian value submitted to LightGBM. Defaults to 0.

    Attributes:
        batch_size (int): Size of mini-batches
        input_dim (int): Input feature dimension
        output_dim (int): Output prediction dimension
        params (dict): LightGBM parameters
        bst (lightgbm.Booster): The underlying LightGBM booster
        FX (torch.nn.Parameter): Current predictions tensor
        train_dat (lightgbm.Dataset): Training dataset used for caching
        min_hess (float): Minimum hessian threshold
    """

    def __init__(self, batch_size, input_dim, output_dim, params={}, min_hess=0):
        super(LGBModule, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        assert "objective" not in params, "objective should not be specified in params"
        self.params = params.copy()
        self.bst = None

        self.FX = nn.Parameter(
            torch.tensor(
                np.zeros([batch_size, output_dim]),
                dtype=torch.float,
            )
        )
        self.train_dat = None
        self.min_hess = min_hess
        self.grad = None
        self.hess = None

    def _set_train_dat(self, input_dataset: lgb.Dataset):
        if input_dataset.params is None:
            input_dataset.params = {"verbose": -1}
        else:
            input_dataset.params.update({"verbose": -1})
        input_dataset.free_raw_data = False
        self.train_dat = input_dataset

    def _input_checking_setting(
        self, input_dataset: Union[lgb.Dataset, np.ndarray, pd.DataFrame]
    ):
        assert isinstance(input_dataset, (lgb.Dataset, np.ndarray, pd.DataFrame))
        if self.training:
            if self.train_dat is None:
                self._set_train_dat(
                    input_dataset
                    if isinstance(input_dataset, lgb.Dataset)
                    else lgb.Dataset(input_dataset)
                )
            if self.bst is None:
                return self.train_dat
            if isinstance(input_dataset, lgb.Dataset):
                assert (
                    input_dataset._handle is not None
                ), "Changing datasets during training is not supported. If trying to do prediction, call LGBModule.eval() first."
            else:  # NEW
                assert (
                    self.batch_size == input_dataset.shape[0]
                ), "Changing datasets during training is not supported. If trying to do prediction, call LGBModule.eval() first."
            return self.train_dat

        if isinstance(input_dataset, lgb.Dataset):
            # Clunky way to get original data
            input_dataset.free_raw_data = False
            input_dataset.construct()
            return input_dataset.get_data()
        return input_dataset

    def forward(
        self,
        input_dataset: Union[lgb.Dataset, np.ndarray, pd.DataFrame],
        return_tensor=True,
    ):
        """Forward pass through the LightGBM module.

        Args:
            input_dataset (Union[lgb.Dataset, np.ndarray, pd.DataFrame]): Input data for prediction.
                Can be a LightGBM Dataset, numpy array, or pandas DataFrame.
            return_tensor (bool, optional): Whether to return predictions as a PyTorch tensor.
                Defaults to True.

        Returns:
            Union[torch.Tensor, np.ndarray]: Model predictions. Returns a PyTorch tensor if
            return_tensor=True, otherwise returns a numpy array.

        The forward pass handles both train and eval
        - In train mode, maintains state between iterations and updates internal FX tensor
        - In eval mode, generates predictions on new data using the trained model
        """
        input_dataset = self._input_checking_setting(input_dataset)

        # TODO figure out how actual batch training works here
        if self.training:
            if self.bst:
                preds = self.bst._Booster__inner_predict(0).copy()
            else:
                preds = np.zeros([self.batch_size, self.output_dim])
        else:
            if self.bst:
                preds = self.bst.predict(input_dataset).copy()
            else:
                preds = np.zeros(
                    [input_dataset.shape[0], self.output_dim], dtype=torch.float
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

    def gb_calc(self):
        self.grad, self.hess = self._get_grad_hess_FX()

    def gb_step(self):
        """Performs a gradient boosting step to update the model.

        This method:
        1. Computes gradients and hessians from the current predictions
        2. Creates a LightGBM objective using the computed gradients/hessians
        3. Updates the internal boosting model by either:
           - Updating the existing model if one exists
           - Training a new model for 1 boosting round if no model exists

        The gradients are scaled by batch size and hessians are clipped to a minimum value
        to ensure numerical stability.

        Returns:
            None
        """
        if self.grad is None and self.hess is None:
            self.gb_calc()

        obj = LightGBObj(self.grad, self.hess)
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
            self.bst = lgb.train(
                params=input_params,
                train_set=self.train_dat,
                num_boost_round=1,
                keep_training_booster=True,
            )
        self.grad = None
        self.hess = None

    def get_extra_state(self):
        return self.bst.model_to_string() if self.bst else None

    def set_extra_state(self, state):
        if state is not None:
            self.bst = lgb.Booster(model_str=state)
        else:
            self.bst = None


class LightGBObj:
    """Helper class for use with LightGBM as a backend for LGBModule"""

    def __init__(self, grad, hess):
        self.grad = grad.detach().numpy()
        self.hess = hess.detach().numpy()

    def __call__(self, y_true, y_pred):
        if self.grad.shape[1] > 1:
            return self.grad, self.hess
        return self.grad.flatten(), self.hess.flatten()
