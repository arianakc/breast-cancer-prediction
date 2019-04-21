# -*- coding:utf-8 -*-
# Author: hankcs
# Date: 2019-04-21 14:21
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate

from src.preprocess import load_data
from src.semisup.co_training import CoTrainingWrapper
from src.semisup.utility import init_logger

logger = init_logger('.', name='co-training.log')
unlabeled_clinical_X, Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, unlabeled_genomic_X, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(
    True)
for data_name, (unlabeled_X, train_X, train_y, dev_X, dev_y, test_X, test_y) in zip(['clinical', 'gnomic'], [
    (unlabeled_clinical_X, Ctr_X, Ctr_Y, Cval_X,
     Cval_Y, Ct_X, Ct_Y), (
            unlabeled_genomic_X, Gtr_X, Gtr_Y,
            Gval_X,
            Gval_Y, Gt_X, Gt_Y)]):
    num_features = train_X.shape[1]
    all_X = np.concatenate([train_X, dev_X, test_X])
    all_y = np.concatenate([train_y, dev_y, test_y])

    features = set(range(0, num_features))
    features1 = {58, 43, 31}
    features2 = features - features1
    features1 = np.array(list(features1), dtype=np.int)
    features2 = np.array(list(features2), dtype=np.int)
    cotraining = CoTrainingWrapper(LogisticRegression(solver='lbfgs', max_iter=300),
                                   LogisticRegression(solver='lbfgs', max_iter=300),
                                   features1,
                                   features2)
    cotraining.fit(train_X, train_y, unlabeled_X)
    acc = accuracy_score(test_y, cotraining.predict(test_X)) * 100
    fit_params = {'unlabeled_X': unlabeled_X}
    cv_acc = cross_validate(cotraining, all_X, all_y, fit_params=fit_params, cv=5)
    cv_acc = cv_acc['test_score'].mean() * 100
    logger.info('{} {} acc {:.2f} cv {:.2f}'.format('Co-Training', data_name, acc, cv_acc))
