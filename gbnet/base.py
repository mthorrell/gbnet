import abc
import warnings
import torch
from torch import nn


class BaseGBModule(nn.Module, abc.ABC):
    """Base class for gradient boosting modules.

    This abstract base class defines the common interface and functionality that all
    gradient boosting modules should implement.

    Attributes:
        min_hess (float) : minimum hessian value
        fixed_hess (float, optional) : fixed hessian value
    """

    def __init__(self, min_hess=0.0, fixed_hess=None):
        super(BaseGBModule, self).__init__()
        if fixed_hess is not None and fixed_hess <= 0:
            raise ValueError("fixed_hess must be positive when provided.")
        if fixed_hess is not None and min_hess > 0:
            warnings.warn(
                "fixed_hess is set and will override min_hess.",
                UserWarning,
                stacklevel=2,
            )
        self.min_hess = min_hess
        self.fixed_hess = fixed_hess
        self.grad = None
        self.hess = None

    @abc.abstractmethod
    def _input_checking_setting(self, input_data):
        """Validate and prepare input data.

        Args:
            input_data: Input data in model-specific format

        Returns:
            Processed input data ready for model
        """
        pass

    @abc.abstractmethod
    def forward(self, input_data, return_tensor: bool = True):
        """Forward pass through the model.

        Args:
            input_data: Input data in model-specific format
            return_tensor: Whether to return predictions as PyTorch tensor

        Returns:
            Model predictions as tensor or numpy array
        """
        pass

    def _get_grad_hess_FX(self):
        grad = self.FX.grad * self.FX.shape[0]

        if self.fixed_hess is not None:
            return grad, torch.full_like(grad, self.fixed_hess)

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
        hess = torch.clamp(torch.cat(hesses, axis=1), min=self.min_hess)
        return grad, hess

    @abc.abstractmethod
    def gb_step(self):
        """Perform one gradient boosting step.

        This method should implement the logic for:
        1. Getting gradients/hessians
        2. Training one boosting iteration
        3. Updating predictions
        """
        pass
