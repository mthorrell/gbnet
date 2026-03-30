import inspect
from typing import Union
import warnings
import numpy as np
import pandas as pd
import torch
import xgboost as xgb
from torch import nn

from gbnet.base import BaseGBModule


def _version_tuple(version_str):
    version_values = []
    for token in version_str.replace("-", ".").split("."):
        if token.isdigit():
            version_values.append(int(token))
        if len(version_values) == 3:
            break

    while len(version_values) < 3:
        version_values.append(0)

    return tuple(version_values)


def _xgb_version_at_least(target_version):
    return _version_tuple(xgb.__version__) >= target_version


class XGBModule(BaseGBModule):
    """XGBoost Module that wraps XGBoost boosting into a PyTorch Module.

    This module allows integration of XGBoost gradient boosting with PyTorch neural networks.
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
        super(XGBModule, self).__init__()
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.params = params.copy()

        assert (
            "objective" not in self.params
        ), "objective should not be specified in params"
        assert (
            "base_score" not in self.params
        ), "base_score should not be specified in params"

        self.params["objective"] = "reg:squarederror"
        self.params["base_score"] = 0
        self.n_completed_boost_rounds = 0
        self.min_hess = min_hess

        init_matrix = np.zeros([batch_size, input_dim])
        self.bst = xgb.train(
            self.params,
            xgb.DMatrix(init_matrix, label=np.zeros(batch_size * output_dim)),
            num_boost_round=0,
        )
        self.n_completed_boost_rounds = 0
        self.dtrain = None
        self.training_n = None
        self._boost_with_iteration = None

        self.FX = nn.Parameter(
            torch.tensor(
                np.zeros([batch_size, output_dim]),
                dtype=torch.float,
            )
        )

    def _check_training_data(self):
        if self.dtrain.get_weight().shape[0] > 0:
            warnings.warn(
                "Weights will not work properly when defined as part of the input DMatrix. Weights should be defined in the loss."
            )

    def _input_checking_setting(
        self, input_data: Union[xgb.DMatrix, pd.DataFrame, np.ndarray]
    ):
        assert isinstance(input_data, (xgb.DMatrix, pd.DataFrame, np.ndarray))
        if self.training:
            if self.dtrain is None:
                if isinstance(input_data, xgb.DMatrix):
                    input_data.set_label(np.zeros(self.batch_size * self.output_dim))
                    self.dtrain = input_data
                    self.training_n = input_data.num_row()
                    self._check_training_data()
                else:
                    self.dtrain = xgb.DMatrix(
                        input_data, label=np.zeros(self.batch_size * self.output_dim)
                    )
                    self.training_n = input_data.shape[0]
            compare_n = (
                input_data.num_row()
                if isinstance(input_data, xgb.DMatrix)
                else input_data.shape[0]
            )
            assert (
                compare_n == self.training_n
            ), "Changing datasets while training is not currently supported. If trying to make predictions, set Module to eval mode via `Module.eval()`"
            return self.dtrain
        return (
            input_data
            if isinstance(input_data, xgb.DMatrix)
            else xgb.DMatrix(input_data)
        )

    def forward(
        self,
        input_data: Union[xgb.DMatrix, np.ndarray, pd.DataFrame],
        return_tensor: bool = True,
    ):
        """Forward pass through the XGBoost module.

        Args:
            input_dataset (Union[xgb.DMatrix, np.ndarray, pd.DataFrame]): Input data for prediction.
                Can be a XGBoost DMatrix, numpy array, or pandas DataFrame.
            return_tensor (bool, optional): Whether to return predictions as a PyTorch tensor.
                Defaults to True.

        Returns:
            Union[torch.Tensor, np.ndarray]: Model predictions. Returns a PyTorch tensor if
            return_tensor=True, otherwise returns a numpy array.

        The forward pass handles both train and eval
        - In train mode, maintains state between iterations and updates internal FX tensor
        - In eval mode, generates predictions on new data using the trained model
        """
        input_data = self._input_checking_setting(input_data)
        preds = self.bst.predict(input_data)

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
        3. Updates the internal boosting model

        The gradients are scaled by batch size and hessians are clipped to a minimum value
        to ensure numerical stability.

        Returns:
            None
        """
        if self.grad is None and self.hess is None:
            self.gb_calc()

        self._gb_step_grad_hess(self.grad, self.hess)
        self.grad = None
        self.hess = None

    def _gb_step_grad_hess(self, grad, hess):
        obj = XGBObj(grad, hess)
        g, h = obj(np.zeros([self.batch_size, self.output_dim]), None)

        self._run_boost_compat(g, h)
        self.n_completed_boost_rounds = self.n_completed_boost_rounds + 1

    def _run_boost_compat(self, grad, hess):
        if self._boost_with_iteration is None:
            try:
                boost_signature = inspect.signature(type(self.bst).boost)
                self._boost_with_iteration = "iteration" in boost_signature.parameters
            except (TypeError, ValueError):
                self._boost_with_iteration = _xgb_version_at_least((2, 1, 0))

        if self._boost_with_iteration:
            call_order = ("with_iteration", "legacy")
        else:
            call_order = ("legacy", "with_iteration")

        last_error = None
        for call_style in call_order:
            try:
                if call_style == "with_iteration":
                    self.bst.boost(
                        self.dtrain,
                        self.n_completed_boost_rounds + 1,
                        grad=grad,
                        hess=hess,
                    )
                    self._boost_with_iteration = True
                else:
                    self.bst.boost(
                        self.dtrain,
                        grad,
                        hess,
                    )
                    self._boost_with_iteration = False
                return
            except TypeError as exc:
                error_message = str(exc)
                signature_mismatch = (
                    "positional argument" in error_message
                    or "unexpected keyword argument" in error_message
                )
                if not signature_mismatch:
                    raise
                last_error = exc

        raise last_error

    def get_extra_state(self):
        return self.bst.save_raw()

    def set_extra_state(self, state):
        self.bst = xgb.Booster(model_file=state)


class XGBObj:
    """Helper class for use with XGBoost as a backend for XGBModule"""

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

        if _xgb_version_at_least((2, 1, 0)):
            g = self.grad.detach().numpy().reshape([M, N])
            h = self.hess.detach().numpy().reshape([M, N])
        else:
            g = self.grad.detach().numpy().reshape([M * N, 1])
            h = self.hess.detach().numpy().reshape([M * N, 1])

        return g, h
