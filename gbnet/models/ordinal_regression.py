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
    def __init__(self, nrounds=500, params=None, module_type="XGBModule"):
        if params is None:
            params = {}
        self.nrounds = nrounds
        self.params = params
        self.model_ = None
        self.losses_ = []
        self.module_type = module_type
        self.Module = loadModule(module_type)

    def fit(self, X, y=None):
        targets = torch.Tensor(y.values).flatten()
        num_classes = len(np.unique(y))

        self.model_ = self.Module(X.shape[0], X.shape[1], 1, params=self.params)
        self.model_.train()

        optimizer = torch.optim.Adam(self.model_.parameters(), lr=0.01)
        loss_fn = OrdinalLogisticLoss(num_classes=num_classes)

        for _ in range(self.nrounds):
            optimizer.zero_grad()
            preds = self.model_(X).flatten()
            loss = loss_fn(preds, targets)
            loss.backward(create_graph=True)
            self.losses_.append(loss.detach().item())
            self.model_.gb_step()
            optimizer.step()

        self.model_.eval()
        return self

    def predict(self, X):
        check_is_fitted(self, "model_")
        preds = self.model_(X).detach().numpy()
        return preds.flatten()


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
