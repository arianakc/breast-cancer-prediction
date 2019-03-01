from preprocess import Preprocessor
import numpy as np
from random import randint
import matplotlib.pyplot as plt

#preprocessing data

preprocessor = Preprocessor()
clinical_X = np.array(preprocessor.clinical_X)
clinical_Y = np.array(preprocessor.clinical_Y)

genomic_X = np.array(preprocessor.genomic_X)
genomic_Y = np.array(preprocessor.genomic_Y)


#devide data set into 8:1:1 as train,validate,test set

#print(clinical_X[0])
#print(len(clinical_X),len(genomic_X))
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

Ctr_X,Ctr_Y,Cval_X,Cval_Y,Ct_X,Ct_Y = devide(clinical_X,clinical_Y,10)
Gtr_X,Gtr_Y,Gval_X,Gval_Y,Gt_X,Gt_Y = devide(genomic_X,genomic_Y,10)


#print(len(Ctr_X),len(Ctr_Y),len(Cval_X),len(Cval_Y),len(Ct_X),len(Ct_Y))

#Using 7 baseline method to predict and compute results' AUC.
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
print(fpr, tpr)
auc1 = auc(fpr, tpr)
print(auc1)
plt.plot(fpr,tpr,lw=1,label='ROC fold %d (area = %0.2f)')
plt.show()
#Using KNeighborsClassifier Method of neighbors class to use Nearest Neighbor algorithm
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
auc1 = auc(fpr, tpr)
print(auc1)
#Using SVC method of svm class to use Support Vector Machine Algorithm

from sklearn.svm import SVC
classifier = SVC(kernel = 'linear', random_state = 0)
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
auc1 = auc(fpr, tpr)
print(auc1)
#Using SVC method of svm class to use Kernel SVM Algorithm

from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
auc1 = auc(fpr, tpr)
print(auc1)
#Using GaussianNB method of naïve_bayes class to use Naïve Bayes Algorithm

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
auc1 = auc(fpr, tpr)
print(auc1)
#Using DecisionTreeClassifier of tree class to use Decision Tree Algorithm

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
auc1 = auc(fpr, tpr)
print(auc1)
#Using RandomForestClassifier method of ensemble class to use Random Forest Classification algorithm

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(Ctr_X, Ctr_Y)
Y_pred = classifier.predict(Cval_X)
fpr, tpr, thresholds = roc_curve(Cval_Y, Y_pred)
auc1 = auc(fpr, tpr)
print(auc1)