import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
import torch
from torch import nn


def loadModule(module):
    assert module in {"XGBModule", "LGBModule"}
    if module == "XGBModule":
        from gbnet import xgbmodule

        return xgbmodule.XGBModule
    if module == "LGBModule":
        from gbnet import lgbmodule

        return lgbmodule.LGBModule


class GBOrd(BaseEstimator, ClassifierMixin):
    """Gradient Boosting Ordinal Regression model.

    This model combines gradient boosting with ordinal regression to predict ordered
    categorical outcomes. It uses either XGBoost or LightGBM as the underlying boosting
    engine wrapped in a PyTorch module.

    Parameters
    ----------
    num_classes : int
        Number of ordinal classes to predict
    nrounds : int, optional
        Number of boosting rounds. Defaults to 500 for XGBModule and 1000 for LGBModule.
    params : dict, optional
        Additional parameters passed to the gradient boosting model.
    module_type : str, optional
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
        Defaults to "LGBModule".
    min_hess : float, optional
        Minimum hessian value for numerical stability. Defaults to 0.0.

    Attributes
    ----------
    model_ : XGBModule or LGBModule
        Trained gradient boosting module. Set after fitting.
    losses_ : list
        List of loss values recorded at each training iteration.
    min_targets : int
        Minimum value in training targets, used for label normalization.

    Methods
    -------
    fit(X, y)
        Trains the model using input features X and ordinal targets y.
    predict(X)
        Predicts ordinal class labels for input features X.

    Notes
    -----
    The model uses an ordinal logistic loss function to handle ordered categorical outcomes.
    The gradient boosting model learns a single score which is transformed into class
    probabilities via learned thresholds.
    """

    def __init__(
        self,
        num_classes,
        nrounds=None,
        params=None,
        module_type="LGBModule",
        min_hess=0.0,
    ):
        if params is None:
            params = {}
        if nrounds is None:
            if module_type == "XGBModule":
                nrounds = 500
            if module_type == "LGBModule":
                nrounds = 1000
        self.nrounds = nrounds
        self.params = params
        self.model_ = None
        self.losses_ = []
        self.module_type = module_type
        self.Module = loadModule(module_type)
        self.loss_fn = OrdinalLogisticLoss(num_classes=num_classes)
        self.num_classes = num_classes
        self.min_hess = min_hess

    def fit(self, X, y=None):
        self.min_targets = min(y)
        targets = torch.Tensor(y.values).flatten()
        targets = targets.long()
        targets = targets - self.min_targets
        assert len(np.unique(y)) == self.num_classes

        self.model_ = self.Module(
            X.shape[0], X.shape[1], 1, params=self.params, min_hess=self.min_hess
        )
        self.model_.train()

        optimizer = torch.optim.Adam(
            list(self.model_.parameters()) + list(self.loss_fn.parameters()), lr=0.01
        )

        for _ in range(self.nrounds):
            optimizer.zero_grad()
            preds = self.model_(X).flatten()
            loss = self.loss_fn(preds, targets)
            loss.backward(create_graph=True)
            self.losses_.append(loss.detach().item())
            self.model_.gb_step()
            optimizer.step()

        self.model_.eval()
        return self

    def score(self, X, y):
        """
        Return the negative log likelihood score for input X and targets y.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        float
            Negative log likelihood score. Lower values indicate better fit.
        """
        check_is_fitted(self, "model_")
        targets = torch.Tensor(y.values).flatten().long()
        targets = targets - torch.min(targets)
        logits = self.model_(X).flatten()
        neg_log_likelihood = self.loss_fn(logits, targets)
        return neg_log_likelihood.detach().item()

    def predict_proba(self, X):
        """
        Predict class probabilities for input X.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        array-like of shape (n_samples, n_classes)
            Predicted class probabilities.
        """
        check_is_fitted(self, "model_")
        logits = self.model_(X).flatten()
        probs = self.loss_fn.get_pred_probs(logits).detach().numpy()
        return probs

    def predict(self, X, return_logits=True):
        """
        Predict continuous output for input X.
        """
        check_is_fitted(self, "model_")
        if return_logits:
            preds = self.model_(X).detach().numpy()
            return preds.flatten()
        else:
            preds = self.predict_proba(X).argmax(axis=1) + self.min_targets
            return preds


class OrdinalLogisticLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

        self.breakpoints = nn.Parameter(
            torch.arange(num_classes - 1, dtype=torch.float32)
            - (num_classes - 2.0) / 2.0
        )

    def _compute_probabilities(self, logits):
        """
        Compute class probabilities
        """
        if logits.dim() == 2:
            logits = logits.squeeze(1)

        # Compute cumulative probabilities
        cum_probs = torch.sigmoid(self.breakpoints.unsqueeze(0) - logits.unsqueeze(1))

        eps = 1e-8
        probs = torch.diff(
            cum_probs,
            prepend=torch.zeros_like(cum_probs[:, :1]),
            append=torch.ones_like(cum_probs[:, :1]),
        ).clamp(min=eps)

        return probs

    def forward(self, logits, targets):
        """
        Compute loss more efficiently but maintaining numerical equivalence.
        """
        targets = targets.flatten()
        probs = self._compute_probabilities(logits)

        # More efficient than one-hot but numerically equivalent
        target_probs = probs[torch.arange(probs.size(0)), targets]
        loss = -torch.log(target_probs).mean()

        return loss

    def get_pred_probs(self, logits):
        """
        Predict most likely class.
        """
        return self._compute_probabilities(logits)
