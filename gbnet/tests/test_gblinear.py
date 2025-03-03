import numpy as np
import pytest
import torch
import torch.nn as nn

from gbnet.gblinear import GBLinear, ridge_regression


def test_gblinear_init():
    model = GBLinear(input_dim=5, output_dim=2)
    assert model.input_dim == 5
    assert model.output_dim == 2
    assert model.bias
    assert model.lr == 0.5
    assert model.min_hess == 0.0
    assert model.lambd == 0.01
    assert isinstance(model.linear, nn.Linear)


def test_gblinear_forward():
    model = GBLinear(input_dim=3, output_dim=2)
    x = torch.randn(4, 3)

    out = model(x)
    assert out.shape == (4, 2)
    assert model.FX is not None
    assert model.input is not None


def test_gblinear_training_step():
    model = GBLinear(input_dim=3, output_dim=2)
    x = torch.randn(4, 3)
    y = torch.randn(4, 2)

    criterion = nn.MSELoss()

    # Forward pass
    out = model(x)
    loss = criterion(out, y)

    # Backward pass
    loss.backward(create_graph=True)

    # Store initial weights
    init_weights = model.linear.weight.clone()
    init_bias = model.linear.bias.clone()

    # GB step
    model.gb_calc()
    model.gb_step()

    # Check weights were updated
    assert not torch.allclose(model.linear.weight, init_weights)
    assert not torch.allclose(model.linear.bias, init_bias)


def test_ridge_regression():
    # Create synthetic data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    true_beta = np.array([1, -0.5, 0.2, -0.3, 0.8])
    y = X @ true_beta + np.random.randn(100) * 0.1

    # Solve ridge regression
    lambd = 0.1
    beta = ridge_regression(X, y, lambd)

    # Check shape
    assert beta.shape == true_beta.shape

    # Check solution is reasonable (close to true parameters)
    assert np.allclose(beta, true_beta, atol=0.5)


def test_input_validation():
    model = GBLinear(input_dim=3, output_dim=2)

    # Test numpy input
    x_np = np.random.randn(4, 3)
    out = model(x_np)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 2)

    # Test pandas input
    import pandas as pd

    x_pd = pd.DataFrame(x_np)
    out = model(x_pd)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (4, 2)

    # Test invalid input
    with pytest.raises(AssertionError):
        model("invalid input")


def test_gblinear_no_bias():
    model = GBLinear(input_dim=3, output_dim=2, bias=False)
    assert not model.bias
    assert model.linear.bias is None

    x = torch.randn(4, 3)
    out = model(x)
    assert out.shape == (4, 2)
