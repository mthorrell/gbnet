import numpy as np
import torch
from unittest import TestCase
from gbnet.base import BaseGBModule


class TestBaseGBModule(TestCase):
    def test_get_grad_hess_FX(self):
        """Test gradient and hessian calculation"""

        class DummyGBModule(BaseGBModule):
            def _input_checking_setting(self, input_data):
                pass

            def forward(self, x):
                pass

            def gb_step(self):
                pass

        # Create dummy module
        module = DummyGBModule()
        module.batch_size = 3
        module.output_dim = 1
        module.min_hess = 1e-6

        # Set up test data
        module.FX = torch.nn.Parameter(torch.tensor([[1.0], [3.0], [5.0]]))
        target = torch.tensor([[0.5], [2.5], [4.5]])
        loss_fn = torch.nn.MSELoss()

        # Calculate loss and backward
        loss = loss_fn(module.FX, target)
        loss.backward(create_graph=True)

        # Get gradients and hessians
        grad, hess = module._get_grad_hess_FX()

        # Check shapes
        self.assertEqual(grad.shape, (3, 1))
        self.assertEqual(hess.shape, (3, 1))

        # Check gradient values (should be 2*(pred - target))
        expected_grad = 2 * (module.FX.data - target)
        np.testing.assert_array_almost_equal(
            grad.detach().numpy(), expected_grad.numpy()
        )

        # Check hessian values (should be 2 for MSE loss)
        expected_hess = torch.full_like(module.FX.data, 2.0)
        np.testing.assert_array_almost_equal(
            hess.detach().numpy(), expected_hess.numpy()
        )

    def test_min_hess_clipping(self):
        """Test minimum hessian clipping"""

        class DummyGBModule(BaseGBModule):
            def _input_checking_setting(self, input_data):
                pass

            def forward(self, x):
                pass

            def gb_step(self):
                pass

        module = DummyGBModule()
        module.batch_size = 2
        module.output_dim = 1
        module.min_hess = 1.5

        # Set up data that would produce hessians < min_hess
        module.FX = torch.nn.Parameter(torch.tensor([[1.0], [2.0]]))
        target = torch.tensor([[1.0], [2.0]])
        loss_fn = torch.nn.MSELoss()

        loss = loss_fn(module.FX, target)
        loss.backward(create_graph=True)

        _, hess = module._get_grad_hess_FX()

        # All hessians should be clipped to min_hess
        self.assertTrue(torch.all(hess >= module.min_hess))
        np.testing.assert_array_almost_equal(
            hess.detach().numpy(),
            np.full((2, 1), 2.0),  # MSE loss has constant hessian of 2.0
        )
