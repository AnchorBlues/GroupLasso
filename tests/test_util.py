# -*- coding: utf-8 -*-

import unittest
import numpy as np
import sklearn.metrics as sm
from .context import grouplasso
from grouplasso.util import sigmoid, binary_log_loss, mean_squared_error


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_log_loss(self):
        np.random.seed(0)
        n_samples = 200
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.uniform(0, 1, size=n_samples)
        loss1 = sm.log_loss(y_true, y_pred)
        loss2 = binary_log_loss(y_true, y_pred)
        assert loss2 > 0
        assert abs(loss1 - loss2) < 1e-12

    def test_mean_squared_error(self):
        np.random.seed(0)
        n_samples = 200
        y_true = np.random.randn(n_samples)
        y_pred = np.random.randn(n_samples)
        loss1 = sm.mean_squared_error(y_true, y_pred)
        loss2 = mean_squared_error(y_true, y_pred)
        assert loss2 > 0
        assert abs(loss1 - loss2) < 1e-12


if __name__ == '__main__':
    unittest.main()
