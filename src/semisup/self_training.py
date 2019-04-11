# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-03-26 00:33
from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.preprocess import load_data


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


if __name__ == '__main__':
    unlabeled_clinical_X, Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(
        True)
    base_model = LogisticRegression(solver='lbfgs')
    base_model.fit(Ctr_X, Ctr_Y)
    # print('LogisticRegression Acc: ', end='')
    print(accuracy_score(Cval_Y, base_model.predict(Cval_X)))
    # print('SemiLogisticRegression Acc: ', end='')
    semi_model = SelfTrainingWrapper(base_model)
    semi_model.fit(Ctr_X, Ctr_Y, unlabeled_clinical_X)
    print(accuracy_score(Cval_Y, semi_model.predict(Cval_X)))
