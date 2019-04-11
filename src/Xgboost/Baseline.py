from preprocess import Preprocessor
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


def devide(X, Y):
    X, Y = shuffle(X, Y, random_state=0)
    x_train = X[0:int(len(X)*0.8)]
    y_train = Y[0:int(len(Y)*0.8)]
    x_val = X[int(len(X)*0.8):int(len(X)*0.9)]
    y_val = Y[int(len(Y)*0.8):int(len(Y)*0.9)]
    x_tst = X[int(len(X)*0.9):len(X)]
    y_tst = Y[int(len(Y)*0.9):len(Y)]
    return x_train, y_train, x_val, y_val, x_tst, y_tst


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
    printer = ['MLPClassifier', 'LogisticRegression','KNeighborsClassifier','Linear SVC', 'rbf SVC','GaussianNB',
               'DecisionTreeClassifier','RandomForestClassifier']
    inner = [MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1),
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


if __name__ == '__main__':
    # preprocessing data
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_Y
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y

    # devide data set into 8:1:1 as train,validate,test set
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = devide(clinical_X, clinical_Y)
    Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = devide(genomic_X, genomic_Y)
    baseline(Ctr_X,Ctr_Y,Cval_X,Cval_Y,Gtr_X,Gtr_Y,Gval_X,Gval_Y)
