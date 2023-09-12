import warnings
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import pandas as pd

from .util import sigmoid, add_intercept, binary_log_loss, mean_squared_error
from ._numba import _proximal_operator, _group_lasso_penalty


class GroupLassoRegressor(BaseEstimator, RegressorMixin):
    """Linear Regression Model trained with the norm of grouped coefficients as regularizer (aka the GroupLasso)

    Parameters
    ----------
    group_ids : numpy array of int, shape (n_features,)
        Array of group identities for each parameter.
        Each entry of the array should contain an int that specify
        group membership for each parameter.

    random_state : int, RandomState instance or None, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    alpha : float, default: 1e-3
        Constant that multiplies the regularization term.

    eta : float, default: 1e-1
        The learning rate.

    tol : float, default: 1e-3
        The tolerance for the optimization.

    max_iter : int, default: 1000
        The maximum number of iterations taken for the solvers to converge.

    initial_weights : None | numpy array, shape (n_features,), default: None
        The values of initial coefficients.
        if None, the values are initialized by the numpy.random.randn function.

    verbose : bool, default: True
        When set to True, the value of objective function is output in the interval
        that is set with the parameter ``verbose_interval``.

    verbose_interval : int, default: 1
        if ``verbose == True`` and the number of iteration can be divided by
        the ``verbose_interval``, the value of objective function is output.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    n_iter_ : int
        number of iterations run by the solver to reach the specified tolerance.

    Notes
    -----
    The algorithm used to fit the model is proximal gradient method.

    """

    def __init__(self, group_ids, random_state=None,
                 alpha=1e-3, eta=1e-1,
                 tol=1e-3, max_iter=1000,
                 initial_weights=None,
                 verbose=True, verbose_interval=1):
        self.group_ids = group_ids
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self.alpha = alpha
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.initial_weights = initial_weights
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self._losses = []

    def fit(self, X, y):
        if not isinstance(self.group_ids, np.ndarray):
            raise TypeError("group_ids must be numpy.array.")

        if self.group_ids.dtype != np.int_:
            raise TypeError("type of group_id must be int.")

        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero.")

        if len(self.group_ids) != X.shape[1]:
            raise ValueError(
                "X.shape[1] must be the same as the length of group_ids.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        X, y = check_X_y(X, y)
        self._losses.clear()

        alpha = float(self.alpha)
        group_ids = self.group_ids.astype(np.int16)
        n_samples = len(X)
        X = add_intercept(X)
        n_features = X.shape[1]
        if self.initial_weights is not None:
            if self.initial_weights.shape != (n_features, ):
                raise ValueError(
                    "initial_weights must have shape (n_features, ).")
            w = self.initial_weights.copy()
        else:
            w = np.zeros(n_features)
        thresh = self.eta * alpha
        itr = 0
        while itr < self.max_iter:
            w_old = w.copy()
            pred = X @ w
            if self.verbose and itr % self.verbose_interval == 0:
                penalty = _group_lasso_penalty(alpha, w[:-1], group_ids)
                loss = mean_squared_error(y, pred) + penalty
                self._losses.append(loss)
                print("training loss:", loss)

            diff = 1 / n_samples * X.T @ (pred - y)
            out = w - self.eta * diff

            w[:-1] = _proximal_operator(out[:-1], thresh, group_ids)
            w[-1] = out[-1]

            if np.linalg.norm(w_old - w, 2) / self.eta < self.tol:
                if self.verbose:
                    print("Converged. itr={}".format(itr))
                break

            itr += 1

        if itr >= self.max_iter:
            warnings.warn("Failed to converge. Increase the "
                          "number of iterations.")
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        self.n_iter_ = itr
        return self

    def predict(self, X):
        check_is_fitted(self, ['coef_', 'intercept_', 'n_iter_'])
        X = check_array(X)
        return X @ self.coef_ + self.intercept_


class GroupLassoClassifier(BaseEstimator, ClassifierMixin):
    """Logistic Regression Model trained with the norm of grouped coefficients as regularizer (aka the GroupLasso)

    Warnings
    ---------
    This class only solves binary classification problems.

    Parameters
    ----------
    group_ids : numpy array of int, shape (n_features,)
        Array of group identities for each parameter.
        Each entry of the array should contain an int that specify
        group membership for each parameter.

    random_state : int, RandomState instance or None, default: None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used by `np.random`.

    alpha : float, default: 1e-3
        Constant that multiplies the regularization term.

    eta : float, default: 1e-1
        The learning rate.

    tol : float, default: 1e-3
        The tolerance for the optimization.

    max_iter : int, default: 1000
        The maximum number of iterations taken for the solvers to converge.

    initial_weights : None | numpy array, shape (n_features,), default: None
        The values of initial coefficients.
        if None, the values are initialized by the numpy.random.randn function.

    verbose : bool, default: True
        When set to True, the value of objective function is output in the interval
        that is set with the parameter ``verbose_interval``.

    verbose_interval : int, default: 1
        if ``verbose == True`` and the number of iteration can be divided by
        the ``verbose_interval``, the value of objective function is output.

    Attributes
    ----------
    coef_ : array, shape (n_features,)
        parameter vector (w in the cost function formula)

    intercept_ : float
        independent term in decision function.

    n_iter_ : int
        number of iterations run by the solver to reach the specified tolerance.

    Notes
    -----
    The algorithm used to fit the model is proximal gradient method.

    """

    def __init__(self, group_ids, random_state=None,
                 alpha=1e-3, eta=1e-1,
                 tol=1e-3, max_iter=1000,
                 initial_weights=None,
                 verbose=True, verbose_interval=1):
        self.group_ids = group_ids
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
        self.alpha = alpha
        self.eta = eta
        self.tol = tol
        self.max_iter = max_iter
        self.initial_weights = initial_weights
        self.verbose = verbose
        self.verbose_interval = verbose_interval
        self._losses = []

    def fit(self, X, y):
        if not isinstance(self.group_ids, np.ndarray):
            raise TypeError("group_ids must be numpy.array.")

        if self.group_ids.dtype != np.int_:
            raise TypeError("type of group_id must be int.")

        if self.alpha <= 0:
            raise ValueError("alpha must be greater than zero.")

        if len(self.group_ids) != X.shape[1]:
            raise ValueError(
                "X.shape[1] must be the same as the length of group_ids.")

        if isinstance(X, pd.DataFrame):
            X = X.values

        if isinstance(y, pd.Series):
            y = y.values

        X, y = check_X_y(X, y)
        self._losses.clear()

        # binary classification
        assert ((y == 0) | (y == 1)).all()

        alpha = float(self.alpha)
        group_ids = self.group_ids.astype(np.int16)
        n_samples = len(X)
        X = add_intercept(X)
        n_features = X.shape[1]
        if self.initial_weights is not None:
            if self.initial_weights.shape != (n_features, ):
                raise ValueError(
                    "initial_weights must have shape (n_features, ).")
            w = self.initial_weights.copy()
        else:
            w = np.zeros(n_features)
        thresh = self.eta * alpha
        itr = 0
        while itr < self.max_iter:
            w_old = w.copy()
            proba = sigmoid(X @ w)
            if self.verbose and itr % self.verbose_interval == 0:
                penalty = _group_lasso_penalty(alpha, w[:-1], group_ids)
                loss = binary_log_loss(y, proba) + penalty
                self._losses.append(loss)
                print("training loss:", loss)

            diff = 1 / n_samples * X.T @ (proba - y)
            out = w - self.eta * diff

            w[:-1] = _proximal_operator(out[:-1], thresh, group_ids)
            w[-1] = out[-1]

            if np.linalg.norm(w_old - w, 2) / self.eta < self.tol:
                if self.verbose:
                    print("Converged. itr={}".format(itr))
                break

            itr += 1

        if itr >= self.max_iter:
            warnings.warn("Failed to converge. Increase the "
                          "number of iterations.")
        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        self.n_iter_ = itr
        return self

    def predict_proba(self, X):
        check_is_fitted(self, ['coef_', 'intercept_', 'n_iter_'])
        X = check_array(X)
        proba = np.zeros((len(X), 2), dtype=np.float64)
        score = X @ self.coef_ + self.intercept_
        proba[:, 1] = sigmoid(score)
        proba[:, 0] = 1 - proba[:, 1]
        return proba

    def predict(self, X):
        check_is_fitted(self, ['coef_', 'intercept_', 'n_iter_'])
        X = check_array(X)
        proba = self.predict_proba(X)
        return (proba[:, 1] > 0.5).astype(int)
