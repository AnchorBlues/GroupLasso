import numpy as np


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def add_intercept(X):
    """
    Xにintercept項を付け加える
    """
    return np.c_[X, np.ones(len(X))]
