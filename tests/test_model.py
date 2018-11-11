# -*- coding: utf-8 -*-

import unittest
import warnings
import time
import inspect
import numpy as np
from sklearn.datasets import load_breast_cancer, load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso, SGDClassifier
import pandas as pd
from .context import grouplasso
from grouplasso.model import GroupLassoRegressor, GroupLassoClassifier

RANDOM_STATE = 42


def _scaling_and_add_noise_feature(x_train, x_test, n_noised_features):
    # scaling
    np.random.seed(0)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)

    # add noise feature
    noise = np.random.randn(len(x_train), n_noised_features)
    x_train = np.c_[x_train, noise]
    noise = np.random.randn(len(x_test), n_noised_features)
    x_test = np.c_[x_test, noise]

    return x_train, x_test


class BasicTestSuite(unittest.TestCase):
    """Basic test cases."""

    def test_basic(self):
        for ModelClass in (GroupLassoRegressor, GroupLassoClassifier):
            # get_params test
            model = ModelClass(
                np.array([0, 1]), random_state=RANDOM_STATE)
            model.get_params()

            # document test (check that the doc has 10 or more lines)
            doc = inspect.getdoc(ModelClass)
            assert doc.count("\n") >= 10

    def test_regressor(self):
        data = load_boston()
        x = data.data
        y = data.target
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=RANDOM_STATE)

        n_noised_features = 5
        x_train, x_test = _scaling_and_add_noise_feature(
            x_train, x_test, n_noised_features)

        # set group id
        group_ids = np.r_[np.zeros(x.shape[1]), np.ones(
            n_noised_features)].astype(int)
        model = GroupLassoRegressor(group_ids=group_ids,
                                    random_state=RANDOM_STATE, verbose=True,
                                    verbose_interval=10,
                                    alpha=1.0, tol=1e-3, eta=1e-1,
                                    max_iter=1000)
        start = time.time()
        model.fit(x_train, y_train)
        print('elapsed time:', time.time() - start)
        print('itr:', model.n_iter_)

        # check that the loss value is getting smaller
        assert len(model._losses) >= model.n_iter_ // model.verbose_interval
        for i in range(1, len(model._losses)):
            assert model._losses[i] < model._losses[i - 1]

        # check that coef of noised feature to be zero
        assert (model.coef_[-n_noised_features:] == 0).all()

        # chek that coef of NOT noised feature not to be zero
        assert (model.coef_[:x.shape[1]] != 0).all()

        # test score is not bad
        score = model.score(x_test, y_test)
        assert score >= 0.65

        # initialize weights
        model_weights = np.r_[model.coef_, model.intercept_]
        # add tiny noise to correct weights
        model_weights += np.random.randn(len(model_weights)) * 0.3
        model2 = GroupLassoRegressor(group_ids=group_ids,
                                     random_state=RANDOM_STATE)
        model2.set_params(**model.get_params())
        model2.set_params(initial_weights=model_weights)
        model2.fit(x_train, y_train)
        assert model2.n_iter_ < model.n_iter_
        assert np.linalg.norm(model2.coef_ - model.coef_, 2) < 5e-2

    def test_regressor_vs_sklearn_Lasso(self):
        """
        compare with lasso of sklearn.
        group lasso become normal lasso if every feature is differenet group with each other.
        """
        data = load_boston()
        x = StandardScaler().fit_transform(data.data)
        y = data.target
        group_ids = np.arange(x.shape[1]).astype(int)
        alpha = 1.0
        group_lasso = GroupLassoRegressor(group_ids=group_ids,
                                          random_state=RANDOM_STATE, verbose=False,
                                          alpha=alpha, tol=1e-3, eta=1e-1,
                                          max_iter=1000)
        group_lasso.fit(x, y)
        print('itr:', group_lasso.n_iter_)
        sklearn_lasso = Lasso(random_state=RANDOM_STATE, alpha=alpha)
        sklearn_lasso.fit(x, y)
        diff_of_coef = np.abs(group_lasso.coef_ - sklearn_lasso.coef_)
        diff_of_intercept = abs(
            group_lasso.intercept_ - sklearn_lasso.intercept_)
        assert (diff_of_coef < 1e-2).all()
        assert diff_of_intercept < 1e-2

    def test_classifier(self):
        data = load_breast_cancer()
        x = data.data
        y = data.target
        x_train, x_test, y_train, y_test = train_test_split(
            x, y, random_state=RANDOM_STATE)

        n_noised_features = 5
        x_train, x_test = _scaling_and_add_noise_feature(
            x_train, x_test, n_noised_features)

        # set group id
        group_ids = np.r_[np.zeros(x.shape[1]), np.ones(
            n_noised_features)].astype(int)
        model = GroupLassoClassifier(group_ids=group_ids,
                                     random_state=RANDOM_STATE, verbose=True,
                                     verbose_interval=10,
                                     alpha=1e-1, tol=1e-3, eta=1e-0,
                                     max_iter=1000)
        start = time.time()
        model.fit(x_train, y_train)
        print('elapsed time:', time.time() - start)
        print('itr:', model.n_iter_)

        # check that the loss value is getting smaller
        assert len(model._losses) >= model.n_iter_ // model.verbose_interval
        for i in range(1, len(model._losses)):
            assert model._losses[i] < model._losses[i - 1]

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

        # initialize weights
        model_weights = np.r_[model.coef_, model.intercept_]
        # add tiny noise to correct weights
        model_weights += np.random.randn(len(model_weights)) * 0.3
        model2 = GroupLassoClassifier(group_ids=group_ids,
                                      random_state=RANDOM_STATE)
        model2.set_params(**model.get_params())
        model2.set_params(initial_weights=model_weights)
        model2.fit(x_train, y_train)
        assert model2.n_iter_ < model.n_iter_
        assert np.linalg.norm(model2.coef_ - model.coef_, 2) < 5e-2

    def test_classifier_vs_sklearn_LogisticRegression(self):
        """
        compare with lasso(L1 logistic regression) of sklearn.
        group lasso become normal lasso if every feature is differenet group with each other.
        """
        data = load_breast_cancer()
        x = StandardScaler().fit_transform(data.data)
        y = data.target
        group_ids = np.arange(x.shape[1]).astype(int)
        alpha = 1e-1
        group_lasso = GroupLassoClassifier(group_ids=group_ids,
                                           random_state=RANDOM_STATE, verbose=False,
                                           alpha=alpha, tol=1e-3, eta=1e-0,
                                           max_iter=1000)
        group_lasso.fit(x, y)
        print('itr:', group_lasso.n_iter_)
        sklearn_lasso = SGDClassifier(loss='log', penalty='l1', alpha=alpha,
                                      l1_ratio=1.0, max_iter=10, random_state=RANDOM_STATE,
                                      learning_rate='invscaling', eta0=1.0, verbose=False)
        sklearn_lasso.fit(x, y)
        diff_of_coef = np.abs(group_lasso.coef_ - sklearn_lasso.coef_[0])
        diff_of_intercept = abs(
            group_lasso.intercept_ - sklearn_lasso.intercept_[0])
        assert (diff_of_coef < 5e-2).all()
        assert diff_of_intercept < 5e-2

    def test_GridSearch(self):
        # ignore "not converge" warning temporarily
        warnings.filterwarnings("ignore")
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
                           random_state=RANDOM_STATE,
                           tol=1e-3, eta=1e-1, max_iter=10,
                           verbose=False),
                param_grid={
                    'alpha': np.logspace(-3, -1, 3),
                },
                cv=3,
                n_jobs=1,
                verbose=False
            )
            model.fit(x, y)
        warnings.filterwarnings("always")

    def test_converge_warning(self):
        """
        check raise warning if not converged.
        """
        data = load_breast_cancer()
        x = StandardScaler().fit_transform(data.data)
        y = data.target
        group_ids = np.arange(x.shape[1]) // 2
        for ModelClass in (GroupLassoRegressor, GroupLassoClassifier):
            model = ModelClass(group_ids=group_ids,
                               random_state=RANDOM_STATE, verbose=False,
                               alpha=1e-1, tol=1e-3, eta=1e-0,
                               max_iter=10)
            with self.assertWarns(UserWarning):
                model.fit(x, y)


if __name__ == '__main__':
    unittest.main()
