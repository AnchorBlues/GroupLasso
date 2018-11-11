import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_intercept(X):
    """
    add intercept to X
    """
    return np.c_[X, np.ones(len(X))]


def binary_log_loss(y_true, y_pred, eps=1e-15):
    """
    binary cross entropy loss
    """
    return -np.mean(y_true * np.log(y_pred + eps) +
                    (1 - y_true) * np.log(1 - y_pred + eps))


def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)
