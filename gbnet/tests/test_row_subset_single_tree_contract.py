from unittest import mock

import lightgbm as lgb
import numpy as np
import pytest
import torch
import xgboost as xgb

from gbnet import lgbmodule as lgm
from gbnet import xgbmodule as xgm


XGB_SINGLE_TREE_PARAMS = {
    "eta": 1.0,
    "max_depth": 2,
    "lambda": 0.0,
    "alpha": 0.0,
    "gamma": 0.0,
    "min_child_weight": 0.0,
    "tree_method": "hist",
    "nthread": 1,
}

LGB_SINGLE_TREE_PARAMS = {
    "learning_rate": 1.0,
    "num_leaves": 4,
    "max_depth": 2,
    "min_data_in_leaf": 1,
    "min_sum_hessian_in_leaf": 0.0,
    "lambda_l1": 0.0,
    "lambda_l2": 0.0,
    "verbose": -1,
    "verbosity": -1,
    "num_threads": 1,
    "seed": 0,
}


def _row_subset_fixture():
    X = np.arange(8, dtype=np.float32).reshape(-1, 1)
    y = np.array(
        [0.0, 100.0, 1.0, 100.0, 2.0, -100.0, 3.0, -100.0],
        dtype=np.float32,
    ).reshape(-1, 1)
    row_indices = np.array([0, 2, 4, 6], dtype=np.int64)
    heldout_rows = np.setdiff1d(np.arange(X.shape[0]), row_indices)
    full_row_indices = np.arange(X.shape[0], dtype=np.int64)
    return X, y, row_indices, heldout_rows, full_row_indices


def _xgb_reference_predictions(X, y, row_indices, params):
    booster = xgb.train(
        {
            **params,
            "objective": "reg:squarederror",
            "base_score": 0.0,
        },
        xgb.DMatrix(X[row_indices], label=y[row_indices].reshape(-1)),
        num_boost_round=1,
    )
    return booster.predict(xgb.DMatrix(X)).reshape(-1, 1)


def _lgb_reference_predictions(X, y, row_indices, params):
    booster = lgb.train(
        {
            **params,
            "objective": "regression",
            "boost_from_average": False,
        },
        lgb.Dataset(
            X[row_indices],
            label=y[row_indices].reshape(-1),
            init_score=np.zeros(len(row_indices), dtype=np.float32),
            free_raw_data=False,
        ),
        num_boost_round=1,
    )
    return booster.predict(X).reshape(-1, 1)


def _fit_xgbmodule_single_tree(X, y, params, row_indices=None):
    module = xgm.XGBModule(
        batch_size=X.shape[0],
        input_dim=X.shape[1],
        output_dim=1,
        params=params,
    )
    mse = torch.nn.MSELoss()

    module.train()
    preds = module(X)
    subset_preds = preds if row_indices is None else preds[row_indices]
    subset_y = y if row_indices is None else y[row_indices]
    loss = 0.5 * mse(subset_preds, torch.tensor(subset_y, dtype=torch.float))
    loss.backward(create_graph=True)

    if row_indices is None:
        module.gb_step()
    else:
        module.gb_step(row_indices=row_indices)

    module.eval()
    return module(X).detach().numpy()


def _fit_lgbmodule_single_tree(X, y, params, row_indices=None):
    module = lgm.LGBModule(
        batch_size=X.shape[0],
        input_dim=X.shape[1],
        output_dim=1,
        params=params,
    )
    mse = torch.nn.MSELoss()

    module.train()
    preds = module(X)
    subset_preds = preds if row_indices is None else preds[row_indices]
    subset_y = y if row_indices is None else y[row_indices]
    loss = mse(subset_preds, torch.tensor(subset_y, dtype=torch.float))
    loss.backward(create_graph=True)

    if row_indices is None:
        module.gb_step()
    else:
        module.gb_step(row_indices=row_indices)

    module.eval()
    return module(X).detach().numpy()


def test_xgbmodule_gb_step_matches_native_full_training():
    X, y, _, _, full_row_indices = _row_subset_fixture()
    expected_full_preds = _xgb_reference_predictions(
        X, y, full_row_indices, XGB_SINGLE_TREE_PARAMS
    )
    actual_full_preds = _fit_xgbmodule_single_tree(
        X,
        y,
        XGB_SINGLE_TREE_PARAMS,
    )

    np.testing.assert_allclose(
        actual_full_preds, expected_full_preds, rtol=1e-6, atol=1e-6
    )


def test_xgbmodule_gb_step_row_indices_matches_native_subset_training():
    X, y, row_indices, heldout_rows, full_row_indices = _row_subset_fixture()
    expected_subset_preds = _xgb_reference_predictions(
        X, y, row_indices, XGB_SINGLE_TREE_PARAMS
    )
    expected_full_preds = _xgb_reference_predictions(
        X, y, full_row_indices, XGB_SINGLE_TREE_PARAMS
    )

    assert np.max(np.abs(expected_subset_preds - expected_full_preds)) > 1e-3
    assert np.max(np.abs(expected_subset_preds[heldout_rows])) > 1e-3

    actual_subset_preds = _fit_xgbmodule_single_tree(
        X,
        y,
        XGB_SINGLE_TREE_PARAMS,
        row_indices=row_indices,
    )

    np.testing.assert_allclose(
        actual_subset_preds,
        expected_subset_preds,
        rtol=1e-6,
        atol=1e-6,
    )


def test_xgbmodule_gb_step_row_indices_uses_cached_slice_and_incremental_boost():
    class DMatrixSliceSpy:
        def __init__(self, dtrain):
            self._dtrain = dtrain
            self.slice_calls = []

        def slice(self, row_indices, allow_groups=False):
            normalized = np.asarray(row_indices, dtype=np.int64)
            self.slice_calls.append(normalized.copy())
            return self._dtrain.slice(normalized, allow_groups=allow_groups)

        def __getattr__(self, name):
            return getattr(self._dtrain, name)

    X, y, row_indices, _, _ = _row_subset_fixture()
    module = xgm.XGBModule(
        batch_size=X.shape[0],
        input_dim=X.shape[1],
        output_dim=1,
        params=XGB_SINGLE_TREE_PARAMS,
    )
    mse = torch.nn.MSELoss()

    module.train()
    preds = module(X)
    loss = 0.5 * mse(
        preds[row_indices], torch.tensor(y[row_indices], dtype=torch.float)
    )
    loss.backward(create_graph=True)

    dtrain_spy = DMatrixSliceSpy(module.dtrain)
    original_dtrain = module.dtrain
    module.dtrain = dtrain_spy

    with (
        mock.patch("xgboost.train") as m_train,
        mock.patch.object(module.bst, "boost", side_effect=module.bst.boost) as m_boost,
    ):
        module.gb_step(row_indices=row_indices)

    module.dtrain = original_dtrain

    assert len(dtrain_spy.slice_calls) == 1
    np.testing.assert_array_equal(dtrain_spy.slice_calls[0], row_indices)
    m_train.assert_not_called()
    m_boost.assert_called_once()
    assert m_boost.call_args.args[0].num_row() == len(row_indices)
    assert module.n_completed_boost_rounds == 1


def test_lgbmodule_gb_step_matches_native_full_training():
    X, y, _, _, full_row_indices = _row_subset_fixture()
    expected_full_preds = _lgb_reference_predictions(
        X, y, full_row_indices, LGB_SINGLE_TREE_PARAMS
    )
    actual_full_preds = _fit_lgbmodule_single_tree(
        X,
        y,
        LGB_SINGLE_TREE_PARAMS,
    )

    np.testing.assert_allclose(
        actual_full_preds, expected_full_preds, rtol=1e-6, atol=1e-6
    )


@pytest.mark.xfail(
    reason="LightGBM row-subset boosting is not implemented yet.",
    raises=TypeError,
)
def test_lgbmodule_gb_step_row_indices_matches_native_subset_training():
    X, y, row_indices, heldout_rows, full_row_indices = _row_subset_fixture()
    expected_subset_preds = _lgb_reference_predictions(
        X, y, row_indices, LGB_SINGLE_TREE_PARAMS
    )
    expected_full_preds = _lgb_reference_predictions(
        X, y, full_row_indices, LGB_SINGLE_TREE_PARAMS
    )

    assert np.max(np.abs(expected_subset_preds - expected_full_preds)) > 1e-3
    assert np.max(np.abs(expected_subset_preds[heldout_rows])) > 1e-3

    actual_subset_preds = _fit_lgbmodule_single_tree(
        X,
        y,
        LGB_SINGLE_TREE_PARAMS,
        row_indices=row_indices,
    )

    np.testing.assert_allclose(
        actual_subset_preds,
        expected_subset_preds,
        rtol=1e-6,
        atol=1e-6,
    )
