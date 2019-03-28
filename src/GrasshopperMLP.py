from src.preprocess import load_data
from src.Baseline import classify
import numpy as np
from random import randint
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.neural_network import  MLPClassifier
from sklearn.utils.validation import check_random_state
from scipy.special import expit as logistic_sigmoid
from scipy import linalg, sparse
from numpy.matlib import repmat
import math
from sklearn.utils.validation import check_array
from sklearn.neural_network._base import binary_log_loss
from sklearn.preprocessing.label import LabelBinarizer
from sklearn.utils.multiclass import unique_labels
from sklearn.utils.validation import check_X_y, column_or_1d

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


def distance(a, b):
    assert(np.size(a, 0) == np.size(b, 0), "A and B should be of same dimensionality")
    aa = np.sum(a * a, 0)
    bb = np.sum(b * b, 0)
    ab = np.dot(a, b)
    i = np.tile(np.transpose(aa)[..., None], [1, np.size(bb)])
    j = np.tile(bb[..., None], [np.size(aa), 1])
    k =2*ab
    d = np.sqrt(np.abs(i+j-k))
    return d


def eps(z):
    zre = np.real(z)
    zim = np.imag(z)
    return np.spacing(np.max([zre, zim]))


def s_func(r):
    return 0.5*np.exp(-r/1.5)-np.exp(-r)

class GOAMultilayerPerceptron:
    def __init__(self, N, hidden_layer_sizes, max_iter, random_state, activation="relu"):
        self.N = N
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
    def _forward_pass(self, activations, coefs, intercepts):
        hidden_activation = ACTIVATIONS[self.activation]
        # Iterate over the hidden layers
        for i in range(self.n_layers_ - 1):
            activations[i + 1] = safe_sparse_dot(activations[i], coefs[i])
            activations[i + 1] += intercepts[i]
            # For the hidden layers
            if (i + 1) != (self.n_layers_ - 1):
                activations[i + 1] = hidden_activation(activations[i + 1])
        # For the last layer
        activations[self.n_layers_-1] = logistic(activations[self.n_layers_-1])
        return activations

    def initialize(self, y, layer_units, coefs_, intercepts_):
        self.n_outputs_ = y.shape[1]
        self.n_layers_ = len(layer_units)
        self.out_activation_ = 'logistic'
        self.n_coefs = []
        self.n_intercepts = []
        self.bound = 0
        bound = 0
        self.coefs_ = coefs_
        self.intercepts_ = intercepts_
        grasshopper_vector = self.encode(coefs_, intercepts_)
        for x in grasshopper_vector:
            if abs(x) > bound:
                bound = abs(x)
        bound = math.ceil(bound)
        self.grasshopper_vector = grasshopper_vector
        self.dim = len(grasshopper_vector)
        self.ub = bound
        self.lb = -bound

    def fit(self, X, y):
        inicial_mlp = MLPClassifier(solver='sgd', alpha=1e-5, hidden_layer_sizes=self.hidden_layer_sizes, random_state=1)
        inicial_mlp.fit(X, y)
        N = self.N
        max_iter = self.max_iter
        hidden_layer_sizes = self.hidden_layer_sizes
        hidden_layer_sizes = list(hidden_layer_sizes)
        X, y = self.validate_input(X, y)
        n_samples, n_features = X.shape
        if y.ndim == 1:
            y = y.reshape((-1, 1))
        self.n_outputs_ = y.shape[1]
        layer_units = ([n_features] + hidden_layer_sizes +
                       [self.n_outputs_])
        self.initialize(y, layer_units, inicial_mlp.coefs_, inicial_mlp.intercepts_)
        y = self.label_binarizer.inverse_transform(y)
        flag = 0
        dim = self.dim
        print("dim:", dim)
        lb = self.lb
        ub = self.ub
        ub = np.ones((dim, 1)) * ub
        lb = np.ones((dim, 1)) * lb
        if dim % 2 != 0:
            dim = dim + 1
            ub = np.append(ub, self.ub)
            lb = np.append(lb, self.lb)
            flag = 1
        if flag == 1:
            self.grasshopper_vector.append(0)
        grasshopper_positions = []
        for i in range(N):
            grasshopper_positions.append(self.grasshopper_vector)
        grasshopper_positions = np.array(grasshopper_positions)
        grasshopper_fitness = []
        cmax = 1
        cmin = 0.00004
        for i in range(np.size(grasshopper_positions, 0)):
            if flag == 1:
                grasshopper_position = grasshopper_positions[i][0:-1]
                coefs, intercepts = self.decode(grasshopper_position)
                y_pred = self._predict(X, coefs, intercepts)
                grasshopper_fitness.append(binary_log_loss(y, y_pred))
            else:
                grasshopper_position = grasshopper_positions[i]
                coefs, intercepts = self.decode(grasshopper_position)
                y_pred = self._predict(X, coefs, intercepts)
                grasshopper_fitness.append(binary_log_loss(y, y_pred))
        sorted_indexes = list(np.array(grasshopper_fitness).argsort())
        grasshopper_fitness.sort(reverse=True)
        sorted_grasshopper = []
        for new_index in range(N):
            sorted_grasshopper.append(grasshopper_positions[sorted_indexes[new_index]])
        target_position = sorted_grasshopper[0]
        target_fitness = grasshopper_fitness[0]
        print("target_position:",  target_position)
        print("target_fitness:", target_fitness)
        l = 2
        grasshopper_positions = np.array(grasshopper_positions)
        print(np.shape(grasshopper_positions))
        while l < max_iter + 1:
            print("iteration ", l)
            tp = np.array(target_position)
            cc = cmax - l * ((cmax - cmin) / max_iter)
            for i in range(np.size(grasshopper_positions, 0)):
                temp = np.transpose(grasshopper_positions)
                s_i = np.zeros((dim, 1))
                for j in range(N):
                    if i != j:
                        dist = distance(temp[:, j], temp[:, i])
                        r_ij_vec = (temp[:, j] - temp[:, i]) / (dist + eps(1))
                        xj_xi = 2 + dist % 2
                        s_ij = np.multiply((ub - lb)*cc/2*s_func(xj_xi), r_ij_vec)
                        s_i = s_i + np.transpose(s_ij)
                X_new = cc * np.transpose(s_i) + tp
                grasshopper_positions[i, :] = np.squeeze(np.transpose(X_new))
            for i in range(N):
                # Relocate grasshoppers that go outside the search space
                tp = np.greater(grasshopper_positions[i, :], np.transpose(ub))
                tm = np.less(grasshopper_positions[i, :], np.transpose(lb))
                grasshopper_positions[i, :] = grasshopper_positions[i, :] * np.logical_not(tp + tm) + np.transpose(
                    ub) * tp + np.transpose(lb) * tm
                if flag == 1:
                    grasshopper_position = grasshopper_positions[i][0:-1]
                    coefs, intercepts = self.decode(grasshopper_position)
                    y_pred = self._predict(X, coefs, intercepts)
                    grasshopper_fitness = binary_log_loss(y, y_pred)
                else:
                    grasshopper_position = grasshopper_positions[i]
                    coefs, intercepts = self.decode(grasshopper_position)
                    y_pred = self._predict(X, coefs, intercepts)
                    grasshopper_fitness = binary_log_loss(y, y_pred)
                if grasshopper_fitness < target_fitness:
                    target_position = grasshopper_positions[i]
                    target_fitness = grasshopper_fitness
                    print("new_fitness:", target_fitness)
            l=l+1
        if flag == 1:
            target_position = target_position[0:-1]
        coefss, interceptss = self.decode(target_position)
        self.coefs_ = coefss
        self.intercepts_ = coefss

    def init_coef(self, fan_in, fan_out):
        # Use the initialization method recommended by
        # Glorot et al.
        factor = 6.
        if self.activation == 'logistic':
            factor = 2.
        init_bound = np.sqrt(factor / (fan_in + fan_out))

        # Generate weights and bias:
        coef_init = self.random_state.uniform(-init_bound, init_bound, (fan_in, fan_out))
        intercept_init = self.random_state.uniform(-init_bound, init_bound, fan_out)
        return coef_init, intercept_init, init_bound
    def encode(self, coefs, intercepts):
        self.n_coefs = []
        self.n_intercepts = []
        grasshopper_position = []
        for array in coefs:
            self.n_coefs.append(np.shape(array))
            for line in array:
                grasshopper_position += list(line)
        for array in intercepts:
            self.n_intercepts.append(np.shape(array))
            grasshopper_position += list(array)
        # print(grasshopper_position)
        # print(self.n_coefs)
        # print(self.n_intercepts)
        # tcoefs, tintercepts = self.decode(grasshopper_position)
        # for i, array in enumerate(tcoefs):
        #     print(np.array_equal(array, coefs[i]))
        #     print(np.shape(array))
        # for i, array in enumerate(tintercepts):
        #     print(np.array_equal(array, intercepts[i]))
        #     print(np.shape(array))
        return grasshopper_position
    def decode(self, grasshopper_position:list):
        coefs = []
        intercepts = []
        pos = 0
        for shape in self.n_coefs:
            coef = []
            for j in range(shape[0]):
                coe = []
                for k in range(shape[1]):
                    coe.append(grasshopper_position[pos])
                    pos = pos+1
                coef.append(coe)
            coefs.append(np.array(coef))
        for shape in self.n_intercepts:
            intercept = []
            for j in range(shape[0]):
                intercept.append(grasshopper_position[pos])
                pos = pos+1
            intercepts.append(np.array(intercept))
        return coefs, intercepts

    def _predict(self, X, coefs, intercepts):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations, coefs, intercepts)
        y_pred = activations[-1]
        return y_pred

    def predict(self, X):
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'])

        # Make sure self.hidden_layer_sizes is a list
        hidden_layer_sizes = self.hidden_layer_sizes
        if not hasattr(hidden_layer_sizes, "__iter__"):
            hidden_layer_sizes = [hidden_layer_sizes]
        hidden_layer_sizes = list(hidden_layer_sizes)

        layer_units = [X.shape[1]] + hidden_layer_sizes + [self.n_outputs_]

        # Initialize layers
        activations = [X]

        for i in range(self.n_layers_ - 1):
            activations.append(np.empty((X.shape[0], layer_units[i + 1])))
        # forward propagate
        self._forward_pass(activations, self.coefs_, self.intercepts_)
        y_pred = activations[-1]
        if self.n_outputs_ == 1:
            y_pred = y_pred.ravel()
        return self.label_binarizer.inverse_transform(y_pred)

    def validate_input(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=['csr', 'csc', 'coo'],
                         multi_output=True)
        if y.ndim == 2 and y.shape[1] == 1:
            y = column_or_1d(y, warn=True)
        classes = unique_labels(y)
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(classes)
        y = self.label_binarizer.transform(y)
        return X, y


if __name__ == '__main__':
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data()
    goamlp_ctr = GOAMultilayerPerceptron(N=5000, hidden_layer_sizes=[70], max_iter=1000, random_state=1)
    classify(goamlp_ctr, Ctr_X, Ctr_Y, Cval_X, Cval_Y, "GOAMLPClassifier", "clinical")
    goamlp_gtr = GOAMultilayerPerceptron(N=5000, hidden_layer_sizes=[36], max_iter=1000, random_state=1)
    classify(goamlp_gtr, Gtr_X, Gtr_Y, Gval_X, Gval_Y, "GOAMLPClassifier", "genetic")
