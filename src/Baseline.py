from src.preprocess import Preprocessor
from src.preprocess import load_data
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle

#Using 7 baseline methods to predict and compute results' AUC.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import  MLPClassifier
import warnings
warnings.filterwarnings("ignore")

def classify(classifier, X, Y, valX, valY, name, type):
    classifier.fit(X, Y)
    Y_pred = classifier.predict(valX)
    fpr, tpr, thresholds = roc_curve(valY, Y_pred)
    auc1 = auc(fpr, tpr)
    print('The AUC of dealing '+type+' data with ' + name + 'is: ', auc1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.plot(fpr, tpr, lw=1)
    plt.text(0.5, 0.3, 'ROC curve (area = %0.2f)' % auc1)
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.title(type+' data with ' + name)
    plt.show()


def baseline(CtrX,CtrY,CvalX,CvalY,GtrX,GtrY,GvalX,GvalY):
    printer = ['MLPClassifier', 'LogisticRegression', 'KNeighborsClassifier', 'Linear SVC', 'rbf SVC', 'GaussianNB',
               'DecisionTreeClassifier', 'RandomForestClassifier']
    inner = [MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=(70), random_state=1),
              LogisticRegression(random_state=0),
              KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2),
             SVC(kernel='linear', random_state=0),
              SVC(kernel='rbf', random_state=0),
             GaussianNB(),
             DecisionTreeClassifier(criterion='entropy', random_state=0),
             RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0)]
    for i in range(len(printer)):
        print('----------------------'+printer[i]+'-----------------------')
        classify(inner[i], CtrX, CtrY, CvalX, CvalY, printer[i], "clinical")
        classify(inner[i], GtrX, GtrY, GvalX, GvalY, printer[i], "genetic")

def fine_tuning_mlp(CtrX,CtrY,CvalX,CvalY,GtrX,GtrY,GvalX,GvalY):
    max_auc_ctr = 0
    max_hidden_size_ctr = 1
    for i in range(100):
        mlp = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5, hidden_layer_sizes=(i+1), random_state=1)
        mlp.fit(CtrX, CtrY)
        Y_pred = mlp.predict(CvalX)
        fpr, tpr, thresholds = roc_curve(CvalY, Y_pred)
        auc1 = auc(fpr, tpr)
        if auc1>max_auc_ctr:
            max_auc_ctr = auc1
            max_hidden_size_ctr = i+1
    print(max_hidden_size_ctr)
    print(max_auc_ctr)
    max_auc_gtr = 0
    max_hidden_size_gtr = 1
    for i in range(100):
        print(i+1)
        mlp = MLPClassifier(solver='sgd', activation="relu", alpha=1e-5, hidden_layer_sizes=(i+1), random_state=1)
        mlp.fit(GtrX, GtrY)
        Y_pred = mlp.predict(GvalX)
        fpr, tpr, thresholds = roc_curve(GvalY, Y_pred)
        auc1 = auc(fpr, tpr)
        if auc1>max_auc_gtr:
            max_auc_gtr = auc1
            max_hidden_size_gtr = i+1
    print(max_hidden_size_gtr)
    print(max_auc_gtr)
    return max_hidden_size_ctr, max_auc_ctr,  max_auc_gtr, max_hidden_size_gtr

if __name__ == '__main__':
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data()
    baseline(Ctr_X, Ctr_Y, Cval_X, Cval_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y)
    # max_hidden_size_ctr, max_auc_ctr, max_auc_gtr, max_hidden_size_gtr = fine_tuning_mlp(Ctr_X, Ctr_Y, Cval_X, Cval_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y)
    # print(max_hidden_size_ctr)
    # print(max_auc_ctr)
    # print(max_hidden_size_gtr)
    # print(max_auc_gtr)
