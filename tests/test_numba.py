# -*- coding: utf-8 -*-

import unittest
import numpy as np
from sklearn.metrics import log_loss
from .context import grouplasso
from grouplasso._numba import _group_lasso_penalty, _proximal_operator
from grouplasso.util import sigmoid


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_basic(self):
        # set parameters
        np.random.seed(0)
        alpha = 1e-2
        eta = 1e-1
        thresh = eta * alpha
        group_ids = (np.arange(10) // 2).astype(np.int16)

        # create dataset
        x = np.array([[0.27117366, 0.125, 0., 0.01415106, 0.,
                       1., 1., 0., 0., 1.],
                      [0.4722292, 0.125, 0., 0.13913574, 0.,
                       0., 0., 1., 0., 0.],
                      [0.32143755, 0., 0., 0.01546857, 0.,
                       1., 0., 0., 0., 1.],
                      [0.43453129, 0.125, 0., 0.1036443, 0.,
                       0., 0., 0., 0., 1.],
                      [0.43453129, 0., 0., 0.01571255, 0.,
                       1., 1., 0., 0., 1.],
                      [0.34656949, 0., 0., 0.0165095, 0.,
                       1., 1., 0., 1., 0.]])
        y = np.array([0, 1, 1, 1, 0, 0])
        n_samples = len(x)

        # initialize coef
        w = np.random.randn(x.shape[1])

        # initial penalty and loss
        penalty = _group_lasso_penalty(alpha, w, group_ids)
        assert penalty > 0
        proba = sigmoid(x @ w)
        loss = log_loss(y, proba) + penalty

        # update coef
        for _ in range(5):
            diff = 1 / n_samples * x.T @ (proba - y)
            out = w - eta * diff
            w = _proximal_operator(out, thresh, group_ids)
            assert w.shape == out.shape

            # confirm that the loss is getting smaller
            penalty2 = _group_lasso_penalty(alpha, w, group_ids)
            proba = sigmoid(x @ w)
            loss2 = log_loss(y, proba) + penalty2
            assert penalty2 < penalty
            assert loss2 < loss

            # update the values for the next iteration.
            loss = loss2
            penalty = penalty2

    def test_devidebyzero(self):
        w = np.zeros(10).astype(np.float64)
        group_ids = (np.arange(10) // 2).astype(np.int16)
        thresh = 1.0
        result = _proximal_operator(w, thresh, group_ids)
        assert (result == 0).all()

if __name__ == '__main__':
    unittest.main()
