
#
# Authors: Changmao Li



from src.preprocess import Preprocessor
from src.Baseline import devide
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import  MLPClassifier
from scipy.special import expit as logistic_sigmoid
from scipy import linalg, sparse
from numpy.matlib import repmat
import math

def safe_sparse_dot(a, b, dense_output=False):
    if sparse.issparse(a) or sparse.issparse(b):
        ret = a * b
        if dense_output and hasattr(ret, "toarray"):
            ret = ret.toarray()
        return ret
    else:
        return np.dot(a, b)


def logistic(X):
    return logistic_sigmoid(X, out=X)


def tanh(X):
    return np.tanh(X, out=X)


def relu(X):
    np.clip(X, 0, np.finfo(X.dtype).max, out=X)
    return X


ACTIVATIONS = {'tanh': tanh, 'logistic': logistic, 'relu': relu}


def initialization(N, dim, up, down):
    if np.size(up, 0) == 1:
        X = np.random.rand(N, dim)*(up-down)+down
    elif np.size(up, 0) > 1:
        X = np.zeros(N, dim)
        for i in range(dim):
            high = up[i]
            low = down[i]
            X[:, i] = np.random.rand(1, N)*(high - low)+low
    else:
        X = np.zeros(N, dim)
        print("wrong size")
        exit(0)
    return X


def distance(a,b):
    assert(np.size(a,0) == np.size(b,0),"A and B should be of same dimensionality")
    aa = np.sum(a * a, 0)
    bb = np.sum(b* b, 0)
    ab = np.dot(a,b)
    d = np.sqrt(np.abs(np.tile(np.transpose(aa)[..., None], [1, np.size(bb, 1)])
                       + np.tile(bb[..., None], [np.size(aa, 1), 1])-2*ab))
    return d


def eps(z):
    zre = np.real(z)
    zim = np.imag(z)
    return np.spacing(np.max([zre, zim]))


def s_func(r):
    return 0.5*np.exp(-r/1.5)-np.exp(-r)


def GOA(N, max_iter, lb, ub, dim):
    flag=0
    if np.size(ub, 0) == 1:
        ub = np.ones(dim, 1)*ub
        lb = np.ones(dim, 1)*lb
    if dim%2 != 0:
        dim = dim+1
        ub = np.append(ub, 100)
        lb = np.append(lb, -100)
        flag = 1
    grasshopper_positions = initialization(N, dim, ub, lb)
    grasshopper_fitness = np.zeros(1, N)
    cmax = 1
    cmin = 0.00004
    for i in range(np.size(grasshopper_positions, 0)):
        if flag ==1:
            # TODO: decode weights and bias from grasshopper_positions and calulate fitness based on loss function
            grasshopper_fitness[0, i] = grasshopper_positions[i, 0:-1]
        else:
            grasshopper_fitness[0, i] = grasshopper_positions[i, :]

    sorted_fitness = grasshopper_fitness.sort()
    sorted_indexes = grasshopper_fitness.argsort()
    sorted_grasshopper = np.zeros(N, dim)
    for new_index in range(N):
        sorted_grasshopper[new_index, :] = grasshopper_positions[sorted_indexes[new_index], :]

    target_position = sorted_grasshopper[0, :]
    target_fitness = sorted_fitness[0]
    l=2
    while l<max_iter+1:
        cc = cmax-l*((cmax-cmin)/max_iter)
        for i in range(np.size(grasshopper_positions, 0)):
            temp = np.transpose(grasshopper_positions)
            s_i = np.zeros(dim, 1)
            for j in range(N):
                if i != j:
                    dist = distance(temp[:, j], temp[:, i])
                    r_ij_vec = (temp[:, j] - temp[:, i]) / (dist + eps(1))
                    xj_xi = 2 + dist % 2
                    s_ij = ((ub - lb) * cc / 2) * s_func(xj_xi)* r_ij_vec
                    s_i = s_i + s_ij
            X_new = cc * np.transpose(s_i) + target_position
            grasshopper_positions[i, :] = np.transpose(X_new)

    for i in range(N):
        # Relocate grasshoppers that go outside the search space
        tp = np.greater(grasshopper_positions[i, :], np.transpose(ub))
        tm = np.less(grasshopper_positions[i, :], np.transpose(lb))
        grasshopper_positions[i, :] = grasshopper_positions[i, :]*np.logical_not(tp+tm)+np.transpose(ub)*tp+np.transpose(lb)*tm
        if flag ==1:
            # TODO: decode weights and bias from grasshopper_positions and calulate fitness based on loss function
            grasshopper_fitness[0, i] = grasshopper_positions[i, 0:-1]
        else:
            grasshopper_fitness[0, i] = grasshopper_positions[i, :]

        if grasshopper_fitness[0,i] < target_fitness:
            target_position = grasshopper_positions[i, :]
            target_fitness = grasshopper_fitness[0, i]
    if flag == 1:
        target_position = target_position[0:-1]
    return target_position



class MultilayerPerceptron:
    def __init__(self, hidden_layer_sizes, activation, max_iter,random_state):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = random_state

    def forward_pass(self, activations):
        """Perform a forward pass on the network by computing the values
        of the neurons in the hidden layers and the output layer.

        Parameters
        ----------
        activations : list, length = n_layers - 1
            The ith element of the list holds the values of the ith layer.
        """
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], self.coefs_[i])
            activations[i + 1] += self.intercepts_[i]

            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])

        # For the last layer
        activations[i + 1] = logistic(activations[i + 1])
        return activations




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
