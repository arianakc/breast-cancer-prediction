from src.preprocess import Preprocessor
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt

#Using 7 baseline methods to predict and compute results' AUC.
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import  MLPClassifier

def devide(X,Y,kFolds):
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
    return [ _ for _ in devided ]


def baseline(CtrX,CtrY,CvalX,CvalY,GtrX,GtrY,GvalX,GvalY):

    printer = ['MLPClassifier', 'LogisticRegression','KNeighborsClassifier','Linear SVC', 'rbf SVC','GaussianNB',
               'DecisionTreeClassifier','RandomForestClassifier']
    inner = [ MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1), LogisticRegression(random_state = 0),KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2),
             SVC(kernel = 'linear', random_state = 0),SVC(kernel = 'rbf', random_state = 0),GaussianNB(),
             DecisionTreeClassifier(criterion = 'entropy', random_state = 0),
             RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)]
    for i in range(len(printer)):
        print('----------------------'+printer[i]+'-----------------------')
        classifier = inner[i]
        classifier.fit(CtrX, CtrY)
        Y_pred = classifier.predict(CvalX)
        fpr, tpr, thresholds = roc_curve(CvalY, Y_pred)
        auc1 = auc(fpr, tpr)
        print('The AUC of dealing clinical data with '+ printer[i]+'is: ',auc1)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(fpr, tpr,lw=1)
        plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % auc1)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.title('Clinical data with '+ printer[i])
        plt.show()

        classifier = inner[i]
        classifier.fit(GtrX,GtrY)
        Y_pred = classifier.predict(GvalX)
        fpr, tpr, thresholds = roc_curve(GvalY, Y_pred)
        auc2 = auc(fpr, tpr)
        print('The AUC of dealing genetic data with '+ printer[i]+'is: ',auc2)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.plot(fpr, tpr,lw=1)
        plt.text(0.5,0.3,'ROC curve (area = %0.2f)' % auc1)
        plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
        plt.title('Genetic data with ' +  printer[i])
        plt.show()


if __name__ == '__main__':
    # preprocessing data
    preprocessor = Preprocessor()
    clinical_X = preprocessor.clinical_X
    clinical_Y = preprocessor.clinical_Y
    genomic_X = preprocessor.genomic_X
    genomic_Y = preprocessor.genomic_Y

    # devide data set into 8:1:1 as train,validate,test set
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y = devide(clinical_X, clinical_Y, 10)
    Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = devide(genomic_X, genomic_Y, 10)
    baseline(Ctr_X,Ctr_Y,Cval_X,Cval_Y,Gtr_X,Gtr_Y,Gval_X,Gval_Y)
