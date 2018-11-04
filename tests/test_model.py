# -*- coding: utf-8 -*-

import unittest
import time
import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import pandas as pd
from .context import grouplasso
from grouplasso.model import GroupLassoRegressor, GroupLassoClassifier


def _scaling_and_add_noise_feature(x_train, x_test, n_noised_features):
    # scaling
    np.random.seed(0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # add noise feature
    n_noised_features = 5
    noise = np.random.randn(len(x_train), n_noised_features)
    x_train = np.c_[x_train, noise]
    noise = np.random.randn(len(x_test), n_noised_features)
    x_test = np.c_[x_test, noise]

    return x_train, x_test


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_basic(self):
        model = GroupLassoRegressor(np.array([0, 1]), random_state=42)
        model.get_params()

    def test_regressor(self):
        data = load_boston()
        x = data.data
        y = data.target
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=42)

        n_noised_features = 5
        x_train, x_test = _scaling_and_add_noise_feature(
            x_train, x_test, n_noised_features)

        # set group id
        group_ids = np.r_[np.zeros(x.shape[1]), np.ones(
            n_noised_features)].astype(int)
        model = GroupLassoRegressor(group_ids=group_ids,
                                    random_state=42, verbose=False,
                                    alpha=1.0, tol=1e-4, eta=1e-1,
                                    max_iter=100)
        start = time.time()
        model.fit(x_train, y_train)
        print('elapsed time:', time.time() - start)
        print('itr:', model.n_iter_)

        # check that coef of noised feature to be zero
        assert (model.coef_[-n_noised_features:] == 0).all()

        # chek that coef of NOT noised feature not to be zero
        assert (model.coef_[:x.shape[1]] != 0).all()

        # test score is not bad
        score = model.score(x_test, y_test)
        assert score >= 0.65

    def test_classifier(self):
        data = load_breast_cancer()
        x = data.data
        y = data.target
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=42)

        n_noised_features = 5
        x_train, x_test = _scaling_and_add_noise_feature(
            x_train, x_test, n_noised_features)

        # set group id
        group_ids = np.r_[np.zeros(x.shape[1]), np.ones(
            n_noised_features)].astype(int)
        model = GroupLassoClassifier(group_ids=group_ids,
                                     random_state=42, verbose=False,
                                     alpha=1.0, tol=1e-4, eta=1e-1,
                                     max_iter=100)
        start = time.time()
        model.fit(x_train, y_train)
        print('elapsed time:', time.time() - start)
        print('itr:', model.n_iter_)

        # check that coef of noised feature to be zero
        assert (model.coef_[-n_noised_features:] == 0).all()

        # chek that coef of NOT noised feature not to be zero
        assert (model.coef_[:x.shape[1]] != 0).all()

        # test predicted result
        proba = model.predict_proba(x_test)
        pred = model.predict(x_test)
        assert proba.shape == (len(x_test), 2)
        assert (np.sum(proba, axis=1) == 1).all()
        assert ((proba >= 0) & (proba <= 1)).all()
        acc = accuracy_score(y_test, pred)
        assert acc >= 0.9

    def test_GridSearch(self):
        x = pd.DataFrame(np.array([[0.27117366, 0.125, 0., 0.01415106, 0.,
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
                                    1., 1., 0., 1., 0.]]))
        y = pd.Series(np.array([0, 1, 1, 1, 0, 0]))
        group_ids = (np.arange(x.shape[1]) // 2).astype(int)
        for ModelClass in (GroupLassoRegressor, GroupLassoClassifier):
            model = GridSearchCV(
                ModelClass(group_ids,
                           random_state=42,
                           tol=1e-4, eta=1e-1, max_iter=10,
                           verbose=False),
                param_grid={
                    'alpha': np.logspace(-3, -1, 3),
                },
                cv=3,
                n_jobs=1,
                verbose=False
            )
            model.fit(x, y)


if __name__ == '__main__':
    unittest.main()
