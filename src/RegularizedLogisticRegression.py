#
# Author: Caleb Ziems
#

import preprocess
from preprocess import load_data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from Baseline import classify

"""Pr(G=1|x)"""
def pr(x, beta):
    exponent = x.dot(beta)
    return 1 / (1 + np.exp(-exponent))

"""Clamped pr within a distance of 1e-5"""
def pr_cap(x, beta):
    p = pr(x, beta)
    p[p<1e-5] = 1e-5
    p[1-p<1e-5] = 1-1e-5
    return p

"""Weights"""
def w_arr(x, beta, p):
    return p*(1-p)

"""Working response"""
def z_arr(x, y, beta, p):
    return (x.dot(beta) + ( (y-p) / (p*(1-p)) ))

"""Soft thresholding"""
def S_arr(z, gamma):
    n = len(z)
    vector = np.zeros(n)
    for j in range(n):
        if gamma[j] >= np.abs(z[j]):
            vector[j] = 0
        else:
            if z[j] > 0:
                vector[j] = z[j] - gamma[j]
            else:
                vector[j] = z[j] + gamma[j]
    return vector

"""Faster version of S_arr"""
def S(z, gamma):
    gamma_less = (gamma < np.abs(z))
    z_greater = (z > 0)
    
    return gamma_less*(z + gamma * (-1)**z_greater)

"""The fitted value excluding the contribution from x_ij"""
def y_tilda_j(x, beta, j):
    return x.dot(beta) - x[:, j]*beta[j]

"""Least squares coefficient for IRLS"""
def LS_coef(x, y, w, beta, j):
    return w.T.dot(x[:, j] * (y - y_tilda_j(x, beta, j)))

"""Matrix version of y_tilda_j (possibly faster?)"""
def y_tilda(x, beta):
    prod = np.dot(x, beta)
    return prod.reshape(len(prod), 1) - (beta.T * x)

"""Matrix version of LS_coef (possibly faster?)"""
def LS(x, y, w, beta):
    return np.dot(w, x*(y.reshape(len(y), 1) - y_tilda(x, beta)))

"""Weighted sum of squares"""
def SOS(x, w):
    return np.inner(x.T**2, w)

"""Weighted least squares coordinate update due to Friedman 2010"""
def coord_update_(x, y, w, beta, lambd, alpha):
    n = len(beta)
    #LS_ = [LS_coef(x, y, w, beta, j) for j in range(n)]
    LS_ = LS(x, y, w, beta)
    #numerator = S_arr(LS_, n*[lambd*alpha])
    numerator = S(LS_, n*[lambd*alpha])
    denominator = SOS(x, w) + lambd*(1-alpha)
    return numerator / denominator

"""Regularized Logistic Regression Classifier due to Friedman 2010"""
class LogisticRegressionClassifier:

    def __init__(self, reg_coeff, l1_ratio, iterate=50):
        self.lambd = reg_coeff
        self.alpha = l1_ratio
        self.beta_0 = 0
        self.beta = np.zeros(1)
        self.iter = iterate
	
    def fit(self, X, Y, verbose=False):
        X = np.array(X)
        Y = np.array(Y)

        m,n = X.shape

        X = np.concatenate((X, np.array([np.ones(m)]).T), 1) # column for beta_0
        m,n = X.shape

        beta = np.zeros(n)

        for loop in range(self.iter): # fixed number of iterations for determining solution
            p = pr_cap(X, beta)
            z = z_arr(X, Y, beta, p)
            w = w_arr(X, beta, p)
            beta = coord_update_(X, z, w, beta, self.lambd, self.alpha)
            if verbose:
                print(loop)

        self.beta_0 = beta[-1]
        self.beta = beta[:-1]

    def predict(self, X, threshold=0.5):
        exponent = X.dot(self.beta) + self.beta_0
        p = 1 / (1 + np.exp(-exponent))
        return (p > threshold).astype(int)

if __name__ == '__main__':
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(False)

    classifier = LogisticRegressionClassifier(reg_coeff=0.1, l1_ratio=0.95)
    classifier.fit(Ctr_X, Ctr_Y)
    classify(classifier, Ctr_X, Ctr_Y, Cval_X, Cval_Y, "LogisticRegressionClassifier", "clinical")
    classifier.fit(Gtr_X, Gtr_Y)
    classify(classifier, Gtr_X, Gtr_Y, Gval_X, Gval_Y, "LogisticRegressionClassifier", "genomic")
    print("done.")
