import numpy as np
from .base import BaseRegressor
from logger import get_logger

class Lasso(BaseRegressor):
    """
    Lasso Regression with L1 regularization using coordinate descent.
    """
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        super().__init__()
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None
        self.n_features_in_ = None
        self.logger = get_logger(f"{self.__class__.__name__}")

    def fit(self, X, y):
        X, y = self._validate_input(X, y)
        self.n_features_in_ = X.shape[1]

        if self.fit_intercept:
            X_mean = np.mean(X, axis=0)
            y_mean = np.mean(y)
            X = X - X_mean
            y = y - y_mean
        else:
            X_mean = np.zeros(X.shape[1])
            y_mean = 0.0

        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()
            for j in range(n_features):
                residual = y - (X @ self.coef_)
                rho = X[:, j].T @ (residual + self.coef_[j] * X[:, j])
                z = np.sum(X[:, j] ** 2)
                if z == 0:
                    continue
                if rho < -self.alpha / 2:
                    self.coef_[j] = (rho + self.alpha / 2) / z
                elif rho > self.alpha / 2:
                    self.coef_[j] = (rho - self.alpha / 2) / z
                else:
                    self.coef_[j] = 0.0
            # Check convergence
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        if self.fit_intercept:
            self.intercept_ = y_mean - X_mean @ self.coef_
        else:
            self.intercept_ = 0.0

        self.logger.info(f"Lasso regression fitted successfully (iterations: {iteration+1})")
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise ValueError("Model must be fitted before making predictions")
        if isinstance(X, tuple):
            X = X[0]
        X = self._validate_input(X, y=None)
        if X.shape[1] != self.n_features_in_:
            raise ValueError(f"X has {X.shape[1]} features, but Lasso was fitted with {self.n_features_in_} features")
        predictions = X @ self.coef_ + self.intercept_
        return np.asarray(predictions).flatten()

    def get_params(self, deep=True):
        return {
            'alpha': self.alpha,
            'fit_intercept': self.fit_intercept,
            'max_iter': self.max_iter,
            'tol': self.tol
        }

    def set_params(self, **params):
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Invalid parameter {key}")
        return self

    def __str__(self):
        if self.coef_ is not None:
            return f"Lasso(alpha={self.alpha}, fitted=True, n_features={len(self.coef_)})"
        else:
            return f"Lasso(alpha={self.alpha}, fitted=False)"

    def __repr__(self):
        return (f"Lasso(alpha={self.alpha}, fit_intercept={self.fit_intercept}, "
                f"max_iter={self.max_iter}, tol={self.tol})")
