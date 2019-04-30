import numpy as np
from random import randint
from xgboost import XGBClassifier
from src.Xgboost.preprocess import Preprocessor
from src.Xgboost.Baseline import devide
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
#Xgboost implementation--------------------------
from sklearn.metrics import *
from sklearn.ensemble import GradientBoostingClassifier
from multiprocessing import freeze_support
from src.Xgboost.gbm import XGB
def result_printer(test_Y, pred_Y, type):
    fpr, tpr, thresholds = roc_curve(test_Y, pred_Y)
    auc1 = auc(fpr, tpr)
    print('The AUC of dealing '+type+' data with Xgboost is: ', auc1)
    print("Mean Absolute Error : " + str(mean_absolute_error(pred_Y, test_Y)))
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr, lw=1)
    plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.title(type +' data with Xgboost')
    plt.show()

def run_myboost(tr_X, tr_Y, val_X, val_Y, test_X, test_Y, n_est, early_stop, type):
    xgb = XGB()
    params1 = {'loss': "logisticloss", # 0.7 eta + 20 max depth 0.23 val present best.
              'eta': 0.7, #0.7 0.68
              'max_depth': 6, #5 0.25, 0.71 #6 - Val 0.23  #8 - Val 0.12 error. #9 - Val 0.28 error. #10 -Val 0.26 #11 -Val 0.29
              'num_boost_round': n_est,
              'subsample': 0.8,
              'colsample': 0.8,
              'min_sample_split': 20, #15 0.66 #20 0.68, 0.25 #25 0.65, 0.29
              'min_child_weight': 1,
              'reg_lambda': 0.8,
              'gamma': 0.2,
              'eval_metric': "error",
              'early_stopping_rounds': early_stop,
              'num_thread':3}
    params2 = {'loss': "logisticloss",
               'eta': 0.7,
               'max_depth': 12, #15 may tend to overfit. Val-error goes around 0.43 #12 significantly improve. Best round val 0.30 #13 overfit spotted
               'num_boost_round': n_est,
               'subsample': 0.8,
               'colsample': 0.8,
               'min_sample_split': 20,
               'min_child_weight': 1,
               'reg_lambda': 0.8,
               'gamma': 0.2,
               'eval_metric': "error",
               'early_stopping_rounds': early_stop,
               'num_thread': 3}
    if type == 'Clinical':
        params = params1
    else:
        params = params2
    xgb.fit(tr_X, tr_Y, validation_data=(val_X, val_Y), **params)
    predictions = xgb.predict(test_X)
    result_printer(test_Y, predictions, type)



def run_myboost_CV(tr_X, tr_Y, val_X, val_Y, test_X, test_Y, n_est, early_stop, type):
    xgb = XGB()
    params1 = {'loss': "logisticloss", # 0.7 eta + 20 max depth 0.23 val present best.
              'eta': 0.7, #0.7 0.68
              'max_depth': 6, #5 0.25, 0.71 #6 - Val 0.23  #8 - Val 0.12 error. #9 - Val 0.28 error. #10 -Val 0.26 #11 -Val 0.29
              'num_boost_round': n_est,
              'subsample': 0.8,
              'colsample': 0.8,
              'min_sample_split': 20, #15 0.66 #20 0.68, 0.25 #25 0.65, 0.29
              'min_child_weight': 1,
              'reg_lambda': 0.8,
              'gamma': 0.2,
              'eval_metric': "error",
              'early_stopping_rounds': early_stop,
              'num_thread':3}
    params2 = {'loss': "logisticloss",
               'eta': 0.7,
               'max_depth': 15, #15 may tend to overfit. Val-error goes around 0.43 #12 significantly improve. Best round val 0.30 #13 overfit spotted
               'num_boost_round': n_est,
               'subsample': 0.8,
               'colsample': 0.8,
               'min_sample_split': 25,
               'min_child_weight': 1,
               'reg_lambda': 0.8,
               'gamma': 0.2,
               'eval_metric': "error",
               'early_stopping_rounds': early_stop,
               'num_thread': 3}
    if type == 'Clinical':
        params = params1
    else:
        params = params2
    xgb.fit(tr_X, tr_Y, validation_data=(val_X, val_Y), **params)
    pred_Y = xgb.predict(test_X)
    fpr, tpr, thresholds = roc_curve(test_Y, pred_Y)
    auc1 = auc(fpr, tpr)
    return auc1


def sk_boost(tr_X, tr_Y, val_X, val_Y, test_X, test_Y, n_est, early_stop, type):
    sk_model = GradientBoostingClassifier(loss='deviance', learning_rate=0.1,
            n_estimators=100, subsample=1.0, criterion='friedman_mse',
            min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
            max_depth=3, min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=None, max_features=None, verbose=0,
            max_leaf_nodes=None, warm_start=False, presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=0.0001)
    sk_model.fit(tr_X,tr_Y)
    predY = sk_model.predict(val_X)
    result_printer(val_Y,predY,type)
    predY = sk_model.predict(test_X)
    result_printer(test_Y, predY, type)
def my_testing():
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_Y
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = devide(clinical_X, clinical_Y)
    Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = devide(genomic_X, genomic_Y)
    #run_myboost(Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, 1000, 5, 'Clinical')
    run_myboost(Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y, 1000, 5, 'Genomics')




def _devide(X,Y,kFolds):
    devided = [[],[],[],[],[],[]]
    i=0
    while i <len(Y):
        assign = randint(0,kFolds-1)
        if assign == kFolds - 1:
            devided[4].append(X[i])
            devided[5].append(Y[i])
        elif assign == kFolds -2:
            devided[2].append(X[i])
            devided[3].append(Y[i])
        else:
            devided[0].append(X[i])
            devided[1].append(Y[i])
        i += 1
    return [ np.array(_) for _ in devided ]

def my_testing_CV(KFolds):
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_Y
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y
    C_scores = []
    G_scores = []
    for i in  range(KFolds):
        Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = _devide(clinical_X, clinical_Y, 5)
        Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = _devide(genomic_X, genomic_Y, 5)
        C_score = run_myboost_CV(Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, 1000, 5, 'Clinical')
        C_scores.append(C_score)
        #G_score = run_myboost_CV(Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y, 1000, 5, 'Genomics')
        #G_scores.append(G_score)
    #print('Clinical Xgboost AUC ',KFolds,'avg score: ',sum(C_scores)/KFolds,C_scores)
    #print('Genomic Xgboost AUC ',KFolds,'avg score: ',sum(G_scores)/KFolds,G_scores)


def sk_testing():
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_Y
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = devide(clinical_X, clinical_Y)
    sk_boost(Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, 1000, 5, 'Clinical')

#Official Xgboost result
def pack_Xgboost(tr_X, tr_Y, val_X, val_Y, test_X, test_Y, n_est, early_stop, type):
    model = XGBClassifier(n_estimators=n_est,learning_rate=0.1, max_depth=10,
                         gamma=0.2, subsample=0.6, colsample_bytree=1.0,
                        objective='binary:logistic', nthread=4)
    model.fit(tr_X, tr_Y, early_stopping_rounds=early_stop,
             eval_set=[(val_X, val_Y)], verbose=False)
    predictions = model.predict(test_X)
    result_printer(test_Y, predictions, type)

if __name__=='__main__':
    freeze_support()
    my_testing()





'''


if __name__ == '__main__':
    # preprocessing data
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_YA
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y

    # devide data set into 8:1:1 as train,validate,test set
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = devide(clinical_X, clinical_Y)
    Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = devide(genomic_X, genomic_Y)

    # make predictions

    pack_Xgboost(Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, 1000, 5, 'Clinical')
    pack_Xgboost(Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y, 1000, 5, 'Genetic')


'''