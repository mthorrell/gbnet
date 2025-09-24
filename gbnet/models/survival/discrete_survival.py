import torch
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


def loadModule(module):
    """Load the appropriate gradient boosting module."""
    assert module in {"XGBModule", "LGBModule"}
    if module == "XGBModule":
        from gbnet import xgbmodule

        return xgbmodule.XGBModule
    if module == "LGBModule":
        from gbnet import lgbmodule

        return lgbmodule.LGBModule


def create_data_matrix(X, module_type, enable_categorical=True):
    """Create appropriate data matrix based on module type.

    Parameters
    ----------
    X : array-like
        Input features
    module_type : str
        Type of module ("XGBModule" or "LGBModule")
    enable_categorical : bool, optional
        Whether to enable categorical features (XGBoost only)

    Returns
    -------
    data_matrix
        XGBoost DMatrix or LightGBM Dataset depending on module type
    """
    if module_type == "XGBModule":
        import xgboost as xgb

        return xgb.DMatrix(X, enable_categorical=enable_categorical)
    elif module_type == "LGBModule":
        import lightgbm as lgb

        return lgb.Dataset(X)
    else:
        raise ValueError(f"Unsupported module type: {module_type}")


# Custom loss functions using log-gamma for Beta probabilities
def log_p_event(t, alpha, beta):
    """
    log P(T=t | alpha, beta) = log B(alpha+1, beta + t -1) - log B(alpha, beta)
    Corrected denominator: Gamma(alpha + beta + t)
    """
    log_b_event = (
        torch.lgamma(alpha + 1)
        + torch.lgamma(beta + t - 1)
        - torch.lgamma(alpha + beta + t)
    )
    log_b_prior = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return log_b_event - log_b_prior


def log_p_surv(t, alpha, beta):
    """
    log P(T > t | alpha, beta) = log B(alpha, beta + t) - log B(alpha, beta)
    """
    log_b_surv = (
        torch.lgamma(alpha) + torch.lgamma(beta + t) - torch.lgamma(alpha + beta + t)
    )
    log_b_prior = torch.lgamma(alpha) + torch.lgamma(beta) - torch.lgamma(alpha + beta)
    return log_b_surv - log_b_prior


# Geometric distribution functions for ThetaSurvivalModel
def log_p_event_geometric(t, theta):
    """
    log P(T=t | theta) = log(theta) + (t-1) * log(1-theta)
    """
    return torch.log(theta) + (t - 1) * torch.log(1 - theta)


def log_p_surv_geometric(t, theta):
    """
    log P(T > t | theta) = t * log(1-theta)
    """
    return t * torch.log(1 - theta)


class BetaSurvivalModel(BaseEstimator, RegressorMixin):
    """Gradient Boosting Beta Survival Model.

    This model combines gradient boosting with a Beta distribution for discrete
    survival analysis. It uses either XGBoost or LightGBM as the underlying
    boosting engine wrapped in a PyTorch module.

    Parameters
    ----------
    nrounds : int, optional
        Number of boosting rounds. Defaults to 500 for XGBModule and 1000 for LGBModule.
    params : dict, optional
        Additional parameters passed to the gradient boosting model.
    module_type : str, optional
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
        Defaults to "XGBModule".
    min_hess : float, optional
        Minimum hessian value for numerical stability. Defaults to 0.0.

    Attributes
    ----------
    model_ : XGBModule or LGBModule
        Trained gradient boosting module. Set after fitting.
    losses_ : list
        List of loss values recorded at each training iteration.
    n_features_in_ : int
        Number of features seen during fit.

    Methods
    -------
    fit(X, y)
        Trains the model using input features X and survival data y.
    predict_survival(X, times)
        Predicts survival probabilities for given times.
    predict_hazard(X, times)
        Predicts hazard probabilities for given times.
    score(X, y)
        Returns the negative log likelihood score.

    Notes
    -----
    The model uses a Beta distribution to model discrete survival times.
    The gradient boosting model learns parameters alpha and beta for each sample,
    which are then used to compute survival and hazard probabilities.

    For survival data, y should be a structured array or DataFrame with columns:
    - 'time': observed time (discrete)
    - 'event': event indicator (0=censored, 1=event)
    """

    def __init__(
        self,
        nrounds=None,
        params=None,
        module_type="XGBModule",
        min_hess=0.0,
    ):
        if params is None:
            params = {}
        if nrounds is None:
            nrounds = 100

        self.nrounds = nrounds
        self.params = params
        self.module_type = module_type
        self.min_hess = min_hess
        self.model_ = None
        self.losses_ = []
        self.Module = loadModule(module_type)

    def fit(self, X, y):
        """Fit the Beta survival model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples, 2) or structured array
            Survival data. If array-like, should have columns [time, event].
            If structured array, should have 'time' and 'event' fields.
            event: 0 for censored, 1 for event observed.

        Returns
        -------
        self : object
            Returns self.
        """
        # Handle different y formats
        times = y["time"]
        events = y["event"]
        self.max_time = max(times)

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Convert to tensors
        times_torch = torch.tensor(times, dtype=torch.float32)
        events_torch = torch.tensor(events, dtype=torch.float32)

        # Create appropriate data matrix based on module type
        dmatrix = create_data_matrix(X, self.module_type, enable_categorical=True)

        # Initialize model
        self.model_ = self.Module(
            batch_size=n_samples,
            input_dim=self.n_features_in_,
            output_dim=2,  # alpha and beta parameters
            params=self.params,
            min_hess=self.min_hess,
        )
        self.model_.train()

        # Training loop
        self.losses_ = []
        for epoch in range(self.nrounds):
            self.model_.zero_grad()

            pred = self.model_(dmatrix)
            a, b = pred[:, 0], pred[:, 1]
            alpha = torch.exp(a)
            beta = torch.exp(b)

            # Compute NLL per sample
            log_probs = torch.zeros(n_samples)
            uncensored_mask = events_torch == 1
            censored_mask = events_torch == 0

            if uncensored_mask.any():
                log_probs[uncensored_mask] = log_p_event(
                    times_torch[uncensored_mask],
                    alpha[uncensored_mask],
                    beta[uncensored_mask],
                )

            if censored_mask.any():
                log_probs[censored_mask] = log_p_surv(
                    times_torch[censored_mask],
                    alpha[censored_mask],
                    beta[censored_mask],
                )

            loss = -torch.mean(log_probs)  # Negative log-likelihood
            loss.backward(create_graph=True)
            self.model_.gb_step()
            self.losses_.append(loss.detach().item())

        self.model_.eval()
        return self

    def predict_survival(self, X, times):
        """Predict survival probabilities P(T > t) for given times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        times : array-like of shape (n_times,)
            Times at which to predict survival probabilities.

        Returns
        -------
        survival_probs : array-like of shape (n_samples, n_times)
            Survival probabilities for each sample at each time point.
        """
        check_is_fitted(self, "model_")

        dmatrix = create_data_matrix(X, self.module_type, enable_categorical=True)

        with torch.no_grad():
            pred = self.model_(dmatrix)
            a, b = pred[:, 0], pred[:, 1]
            alpha = torch.exp(a)
            beta = torch.exp(b)

            survival_probs = np.zeros((X.shape[0], len(times)))
            times_tensor = torch.tensor(times, dtype=torch.float32)

            for i, t in enumerate(times_tensor):
                log_surv = log_p_surv(t, alpha, beta)
                survival_probs[:, i] = torch.exp(log_surv).cpu().numpy()

        return survival_probs

    def predict(self, X):
        """Predict the expected survival time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        expected_times : array-like of shape (n_samples,)
            Expected survival times for each sample.
        """
        check_is_fitted(self, "model_")

        # Define time range for computing expected survival time
        # Use max training time * 2 to ensure we capture the full survival curve
        max_time = int(self.max_time * 2)
        times = 1 + np.arange(max_time)

        # Get survival probabilities for all time points
        survival_probs = self.predict_survival(X, times)

        # Expected survival time is the sum of survival probabilities
        # This represents the area under the survival curve
        expected_times = np.sum(survival_probs, axis=1)

        return expected_times

    def score(self, X, y):
        """Return the negative log likelihood score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples, 2) or structured array
            Survival data with time and event columns.

        Returns
        -------
        score : float
            Negative log likelihood score. Lower values indicate better fit.
        """
        check_is_fitted(self, "model_")

        times = y["time"]
        events = y["event"]

        dmatrix = create_data_matrix(X, self.module_type, enable_categorical=True)

        with torch.no_grad():
            pred = self.model_(dmatrix)
            a, b = pred[:, 0], pred[:, 1]
            alpha = torch.exp(a)
            beta = torch.exp(b)

            times_torch = torch.tensor(times, dtype=torch.float32)
            events_torch = torch.tensor(events, dtype=torch.float32)

            log_probs = torch.zeros(X.shape[0])
            uncensored_mask = events_torch == 1
            censored_mask = events_torch == 0

            if uncensored_mask.any():
                log_probs[uncensored_mask] = log_p_event(
                    times_torch[uncensored_mask],
                    alpha[uncensored_mask],
                    beta[uncensored_mask],
                )

            if censored_mask.any():
                log_probs[censored_mask] = log_p_surv(
                    times_torch[censored_mask],
                    alpha[censored_mask],
                    beta[censored_mask],
                )

            neg_log_likelihood = -torch.mean(log_probs)

        return neg_log_likelihood.detach().item()


class ThetaSurvivalModel(BaseEstimator, RegressorMixin):
    """Gradient Boosting Theta Survival Model.

    This model combines gradient boosting with a geometric distribution for discrete
    survival analysis. It uses either XGBoost or LightGBM as the underlying
    boosting engine wrapped in a PyTorch module.

    The model learns parameters a and b for each sample, then computes theta = a/(a+b)
    which defines the probability parameter of a geometric distribution for survival times.

    Parameters
    ----------
    nrounds : int, optional
        Number of boosting rounds. Defaults to 100.
    params : dict, optional
        Additional parameters passed to the gradient boosting model.
    module_type : str, optional
        Type of gradient boosting module to use, either "XGBModule" or "LGBModule".
        Defaults to "XGBModule".
    min_hess : float, optional
        Minimum hessian value for numerical stability. Defaults to 0.0.

    Attributes
    ----------
    model_ : XGBModule or LGBModule
        Trained gradient boosting module. Set after fitting.
    losses_ : list
        List of loss values recorded at each training iteration.
    n_features_in_ : int
        Number of features seen during fit.

    Methods
    -------
    fit(X, y)
        Trains the model using input features X and survival data y.
    predict_survival(X, times)
        Predicts survival probabilities for given times.
    predict(X)
        Predicts the expected survival time.
    score(X, y)
        Returns the negative log likelihood score.

    Notes
    -----
    The model uses a geometric distribution to model discrete survival times.
    The gradient boosting model learns parameters a and b for each sample,
    which are used to compute theta = a/(a+b), the success probability.

    Survival probabilities follow:
    - P(T=t) = theta * (1-theta)^(t-1) for event at time t
    - P(T>t) = (1-theta)^t for survival beyond time t

    For survival data, y should be a structured array or DataFrame with columns:
    - 'time': observed time (discrete)
    - 'event': event indicator (0=censored, 1=event)
    """

    def __init__(
        self,
        nrounds=None,
        params=None,
        module_type="XGBModule",
        min_hess=0.0,
    ):
        if params is None:
            params = {}
        if nrounds is None:
            nrounds = 100

        self.nrounds = nrounds
        self.params = params
        self.module_type = module_type
        self.min_hess = min_hess
        self.model_ = None
        self.losses_ = []
        self.Module = loadModule(module_type)

    def fit(self, X, y):
        """Fit the Theta survival model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples, 2) or structured array
            Survival data. If array-like, should have columns [time, event].
            If structured array, should have 'time' and 'event' fields.
            event: 0 for censored, 1 for event observed.

        Returns
        -------
        self : object
            Returns self.
        """
        # Handle different y formats
        times = y["time"]
        events = y["event"]
        self.max_time = max(times)

        self.n_features_in_ = X.shape[1]
        n_samples = X.shape[0]

        # Convert to tensors
        times_torch = torch.tensor(times, dtype=torch.float32)
        events_torch = torch.tensor(events, dtype=torch.float32)

        # Create appropriate data matrix based on module type
        dmatrix = create_data_matrix(X, self.module_type, enable_categorical=True)

        # Initialize model
        self.model_ = self.Module(
            batch_size=n_samples,
            input_dim=self.n_features_in_,
            output_dim=1,
            params=self.params,
            min_hess=self.min_hess,
        )
        self.model_.train()

        # Training loop
        self.losses_ = []
        for epoch in range(self.nrounds):
            self.model_.zero_grad()

            a = self.model_(dmatrix).flatten()
            theta = torch.sigmoid(a)

            # Compute NLL per sample
            log_probs = torch.zeros(n_samples)
            uncensored_mask = events_torch == 1
            censored_mask = events_torch == 0

            if uncensored_mask.any():
                log_probs[uncensored_mask] = log_p_event_geometric(
                    times_torch[uncensored_mask],
                    theta[uncensored_mask],
                )

            if censored_mask.any():
                log_probs[censored_mask] = log_p_surv_geometric(
                    times_torch[censored_mask],
                    theta[censored_mask],
                )

            loss = -torch.mean(log_probs)  # Negative log-likelihood
            loss.backward(create_graph=True)
            self.model_.gb_step()
            self.losses_.append(loss.detach().item())

        self.model_.eval()
        return self

    def predict_survival(self, X, times):
        """Predict survival probabilities P(T > t) for given times.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        times : array-like of shape (n_times,)
            Times at which to predict survival probabilities.

        Returns
        -------
        survival_probs : array-like of shape (n_samples, n_times)
            Survival probabilities for each sample at each time point.
        """
        check_is_fitted(self, "model_")

        dmatrix = create_data_matrix(X, self.module_type, enable_categorical=True)

        with torch.no_grad():
            a = self.model_(dmatrix).flatten()
            theta = torch.sigmoid(a)

            survival_probs = np.zeros((X.shape[0], len(times)))
            times_tensor = torch.tensor(times, dtype=torch.float32)

            for i, t in enumerate(times_tensor):
                log_surv = log_p_surv_geometric(t, theta)
                survival_probs[:, i] = torch.exp(log_surv).cpu().numpy()

        return survival_probs

    def predict(self, X):
        """Predict the expected survival time.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        expected_times : array-like of shape (n_samples,)
            Expected survival times for each sample.
        """
        check_is_fitted(self, "model_")

        # Define time range for computing expected survival time
        # Use max training time * 2 to ensure we capture the full survival curve
        max_time = int(self.max_time * 2)
        times = 1 + np.arange(max_time)

        # Get survival probabilities for all time points
        survival_probs = self.predict_survival(X, times)

        # Expected survival time is the sum of survival probabilities
        # This represents the area under the survival curve
        expected_times = np.sum(survival_probs, axis=1)

        return expected_times

    def score(self, X, y):
        """Return the negative log likelihood score.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        y : array-like of shape (n_samples, 2) or structured array
            Survival data with time and event columns.

        Returns
        -------
        score : float
            Negative log likelihood score. Lower values indicate better fit.
        """
        check_is_fitted(self, "model_")

        times = y["time"]
        events = y["event"]

        dmatrix = create_data_matrix(X, self.module_type, enable_categorical=True)

        with torch.no_grad():
            a = self.model_(dmatrix).flatten()
            theta = torch.sigmoid(a)

            times_torch = torch.tensor(times, dtype=torch.float32)
            events_torch = torch.tensor(events, dtype=torch.float32)

            log_probs = torch.zeros(X.shape[0])
            uncensored_mask = events_torch == 1
            censored_mask = events_torch == 0

            if uncensored_mask.any():
                log_probs[uncensored_mask] = log_p_event_geometric(
                    times_torch[uncensored_mask],
                    theta[uncensored_mask],
                )

            if censored_mask.any():
                log_probs[censored_mask] = log_p_surv_geometric(
                    times_torch[censored_mask],
                    theta[censored_mask],
                )

            neg_log_likelihood = -torch.mean(log_probs)

        return neg_log_likelihood.detach().item()
