# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-03-26 00:33
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

from src.preprocess import load_data
from src.semisup.utility import init_logger


class SelfTrainingWrapper(BaseEstimator):
    def __init__(self, model, max_iter=100, threshold=0.8) -> None:
        super().__init__()
        self.model = model
        self.max_iter = max_iter
        self.threshold = threshold

    def fit(self, X, y, unlabeled_X):
        self.model.fit(X, y)
        unlabeled_y = self.predict(unlabeled_X)
        prob = self.predict_proba(unlabeled_X)
        for it in range(self.max_iter):
            usable_idx = np.where((prob[:, 0] > self.threshold) | (prob[:, 1] > self.threshold))[0]
            self.model.fit(np.vstack((X, unlabeled_X[usable_idx, :])), np.hstack((y, unlabeled_y[usable_idx])))
            pre_unlabeled_y = unlabeled_y
            unlabeled_y = self.predict(unlabeled_X)
            if np.array_equal(unlabeled_y, pre_unlabeled_y):  # converge
                break
            prob = self.predict_proba(unlabeled_X)

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


if __name__ == '__main__':

    logger = init_logger('.', name='self-training.log')
    unlabeled_clinical_X, Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, unlabeled_genomic_X, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(
        True)
    for model_name, model in zip(['logistic', 'self-training'], [LogisticRegression(solver='lbfgs'),
                                                                 SelfTrainingWrapper(
                                                                     LogisticRegression(solver='lbfgs'))]):
        for data_name, (unlabeled_X, train_X, train_y, dev_X, dev_y, test_X, test_y) in zip(['clinical', 'gnomic'], [
            (unlabeled_clinical_X, Ctr_X, Ctr_Y, Cval_X,
             Cval_Y, Ct_X, Ct_Y), (
                    unlabeled_genomic_X, Gtr_X, Gtr_Y,
                    Gval_X,
                    Gval_Y, Gt_X, Gt_Y)]):
            fit_params = {'unlabeled_X': unlabeled_X} if '-' in model_name else {}
            model.fit(train_X, train_y, **fit_params)
            acc = accuracy_score(test_y, model.predict(test_X)) * 100
            all_X = np.concatenate([train_X, dev_X, test_X])
            all_Y = np.concatenate([train_y, dev_y, test_y])
            cv_acc = cross_validate(model, all_X, all_Y, fit_params=fit_params, cv=5)
            cv_acc = cv_acc['test_score'].mean() * 100
            logger.info('{} {} acc {:.2f} cv {:.2f}'.format(model_name, data_name, acc, cv_acc))
