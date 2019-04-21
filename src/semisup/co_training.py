# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-04-10 18:48
import itertools

from sklearn.base import BaseEstimator
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from src.preprocess import load_data
from src.semisup.utility import init_logger


class CoTrainingWrapper(BaseEstimator):
    def __init__(self, model1, model2, features1, features2, max_iter=300, unlabeled_pool_size=75) -> None:
        self.features2 = features2
        self.features1 = features1
        self.model1 = model1
        self.model2 = model2
        self.max_iter = max_iter
        self.unlabeled_pool_size = unlabeled_pool_size

    def choose_samples_most_confidently(self, prob, label, top_k):
        p = prob[:, label]
        indices = p.argsort()
        return indices[:top_k]

    def fit(self, X, y: np.ndarray, unlabeled_X=None):
        if unlabeled_X is not None:
            X = np.concatenate([X, unlabeled_X])
            unlabeled_y = np.ones(len(unlabeled_X)) * -1
            y = np.concatenate([y, unlabeled_y])
        X1, X2 = X[:, self.features1], X[:, self.features2]
        freq = dict(zip(*np.unique(y, return_counts=True)))
        num_p = freq[1]
        num_n = freq[0]
        if num_n > num_p:
            p = 1
            n = int(round(num_n / num_p))
        else:
            n = 1
            p = int(round(num_p / num_n))

        unlabeled_remained = np.where(y == -1)[0]
        u = min(len(unlabeled_remained), self.unlabeled_pool_size)
        unlabeled_pool = unlabeled_remained[:u]
        unlabeled_remained = unlabeled_remained[u:]
        labeled_pool = np.where(y != -1)[0]
        for iter in range(self.max_iter):
            if not len(unlabeled_pool):
                break
            self.model1.fit(X1[labeled_pool], y[labeled_pool])
            self.model2.fit(X2[labeled_pool], y[labeled_pool])

            proba1 = self.model1.predict_proba(X1[unlabeled_pool])
            pos1 = self.choose_samples_most_confidently(proba1, 1, p)
            neg1 = self.choose_samples_most_confidently(proba1, 0, n)

            proba2 = self.model2.predict_proba(X2[unlabeled_pool])
            pos2 = self.choose_samples_most_confidently(proba2, 1, p)
            neg2 = self.choose_samples_most_confidently(proba2, 0, n)

            predict_pos = np.concatenate([pos1, pos2])
            predict_neg = np.concatenate([neg1, neg2])

            # add to labeled samples
            y[predict_pos] = 1
            y[predict_neg] = 0

            # remove from unlabeled pool
            unlabeled_pool = np.delete(unlabeled_pool, np.concatenate([predict_neg, predict_pos]), axis=0)
            # fill unlabeled pool
            if len(unlabeled_remained):
                to_fill = min(len(unlabeled_remained), len(predict_pos) + len(predict_neg))
                unlabeled_pool = np.concatenate([unlabeled_pool, unlabeled_remained[:to_fill]])
                unlabeled_remained = unlabeled_remained[to_fill:]

    def predict(self, X):
        X1, X2 = X[:, self.features1], X[:, self.features2]
        proba1 = self.model1.predict_proba(X1)
        proba2 = self.model2.predict_proba(X2)
        ensemble: np.ndarray = proba1 + proba2
        return ensemble.argmax(axis=1)

    def score(self, X, y):
        return accuracy_score(y, self.predict(X))


if __name__ == '__main__':
    unlabeled_clinical_X, Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, unlabeled_genomic_X, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(
        True)

    num_unlabeled_samples = len(unlabeled_genomic_X)
    num_features = Gtr_X.shape[1]
    unlabeled_y = np.ones(num_unlabeled_samples) * -1
    Gtr_X = np.concatenate([Gtr_X, unlabeled_genomic_X])
    Gtr_Y = np.concatenate([Gtr_Y, unlabeled_y])
    features = set(range(0, num_features))
    logger = init_logger(name='genomic_feature.log')
    best_score, best_features = 0, None
    for size in range(1, int(num_features / 2) + 1):
        for features1 in set(itertools.combinations(features, size)):
            features1 = set(features1)
            features2 = features - features1
            features1 = np.array(list(features1), dtype=np.int)
            features2 = np.array(list(features2), dtype=np.int)
            cotraining = CoTrainingWrapper(LogisticRegression(solver='lbfgs', max_iter=300), LogisticRegression(solver='lbfgs', max_iter=300),
                                           features1,
                                           features2)
            cotraining.fit(Gtr_X, Gtr_Y)
            assert Gtr_X.shape[0] == Gtr_Y.shape[0]
            score = accuracy_score(Gval_Y, cotraining.predict(Gval_X))
            score *= 100
            if score > best_score:
                best_score = score
                best_features = features1
            logger.info('[{}] score={:.2f} best={:.2f}'.format(','.join(str(i) for i in features1), score, best_score))
    logger.info('Best score={} feature={}'.format(best_score, ','.join(str(i) for i in best_features)))
