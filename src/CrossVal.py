"""

I use CrossVal to perform 5-fold nested cross validation to find the optimal regularization coefficient for my Logistic Regression classifier

-Caleb

"""


import RegularizedLogisticRegression
from preprocess import load_data
import matplotlib.pyplot as plt
#from Baseline import classify
from preprocess import load_data
import numpy as np
from sklearn.metrics import roc_curve, auc
import random

"""Synchronizes a shuffle on X and Y, then partitions k folds (removing the remainder) 
and returns a list of [X_i,Y_i] lists for each fold i"""
def k_fold_shuffle(X, Y, k, seed=None):
    if seed:
        random.seed(seed)
    
    m = X.shape[0]
    assert m == Y.shape[0]
    assert k > 0
    
    X_shuff = X.copy()
    Y_shuff = Y.copy()
    
    indices = list(range(m))
    random.shuffle(indices)
    
    X_shuff = X_shuff[indices]
    Y_shuff = Y_shuff[indices]
    
    fold_size = int(m/k)
    folds = [ [X_shuff[i:i+fold_size], Y_shuff[i:i+fold_size]] for i in range(0, m-fold_size+1, fold_size) ]
    
    return folds[:k]

"""Aggregates X and Y data from @folds, excluding those with indices from @hold_out"""
def training_data(folds, hold_out):
    training_X = []
    training_Y = []
    for fold_index in range(len(folds)):
        if fold_index not in hold_out:
            training_X.extend(folds[fold_index][0])
            training_Y.extend(folds[fold_index][1])
    return [np.array(training_X), np.array(training_Y)]

"""Finds the greatest i such that y[i] is within one std of y[index]"""
def find_optimal(y, index, std):
    optimal_index = index
    upper_bound = y[index]+std
    lower_bound = y[index]-std
    for i in range(index+1, len(y)):
        if (y[i] <= upper_bound) and (y[i] >= lower_bound):
            optimal_index = i
    return optimal_index

def plot_errors(means, stds, lambdas, min_index, opt_index, name="error_plot", type=None):
    # prepare datapoints
    means_min = means[min_index]
    lambda_min = lambdas[min_index]
    std_min = stds[min_index]
    means_opt = means[opt_index]
    lambda_opt = lambdas[opt_index]

    # setup plot
    log_10 = np.log10(lambdas)
    plt.figure(figsize=(7, 7))
    ax = plt.axes()

    # plot mu and mu_err
    plt.plot(log_10, means, color='black', label="$\mu_{err}$")
    plt.plot(log_10, means + stds, color='red', label="$\mu_{err} \pm \sigma_{err}$")
    plt.plot(log_10, means - stds, color='red')

    # plot lambda min
    plt.plot(log_10, [means_min + std_min]*len(log_10), color='blue', linestyle='dashed')
    plt.plot(log_10, [means_min - std_min]*len(log_10), color='blue', linestyle='dashed')
    plt.plot(np.log10([lambda_min]*10), np.linspace(*ax.get_ylim(), 10), color='blue', linestyle='dashed')
    plt.scatter(np.log10([lambda_min]), [means_min], s=80, facecolors='none', edgecolors='b', label="$\lambda_{min}$ = %2.4f" % lambda_min)

    # plot optimal
    plt.plot(np.log10([lambda_opt]*10), np.linspace(*ax.get_ylim(), 10), color='green', linestyle='dashed')
    plt.scatter(np.log10([lambda_opt]), [means_opt], s=80, facecolors='none', edgecolors='g', label="$\lambda^*$ = %2.4f" % lambda_opt)

    # finish
    plt.legend(loc='lower right')
    plt.xlabel("$\log_{10}(\lambda)$")
    plt.ylabel("Classification Error")
    if type:
        plt.title("Validation Errors for %s" % type)
    else:
        plt.title("Validation Errors")
    plt.savefig('%s.png' % name)
    
    with open('%s.txt' % name, 'w') as f:
        f.write(str(lambda_opt))
    
"""Returns the MSE of binary @predictions from the @true values"""
def error(predictions, true):
    n = len(predictions)
    er = 0
    for i in range(n):
        if predictions[i] != true[i]:
            er += 1
    return er / n

def cross_validation(X, Y, k=5, save_name='', type=None):
    training_errors = [] # for each outer loop
    validation_errors = [] # for each outer loop
    test_errors = [] # for each outer loop

    trn_errors = [] # for each inner loop
    val_errors = [] # for each inner loop
    
    opt_lambdas = []

    lambdas = np.logspace(-5, 2, 500)
    #classifiers = [ SGDClassifier(alpha=lambda_i, loss='log', penalty='elasticnet', l1_ratio=0.95, shuffle=False) for lambda_i in lambdas ]
    classifiers = [ RegularizedLogisticRegression.LogisticRegressionClassifier(reg_coeff=lambda_i, l1_ratio=0.95)
                    for lambda_i in lambdas ]

    # nested cross validation, outer loop
    folds = k_fold_shuffle(X, Y, k, seed=448)
    i=0
    for test_index in range(k):
        i+=1
        print("t", test_index)
        test_X, test_Y = folds[test_index]

        # nested cross validation, inner loop
        for validation_index in range(k):
            print('v:', validation_index)
            if validation_index != test_index:
                validation_X, validation_Y = folds[validation_index]
                training_X, training_Y = training_data(folds, hold_out=[validation_index, test_index])

                trn_row = []
                val_row = []
                for classifier in classifiers:
                    classifier.fit(training_X, training_Y)

                    # Test Error
                    pred_Y = classifier.predict(training_X)
                    tsterr = error(pred_Y, training_Y)
                    #print('tsterr:', tsterr)
                    trn_row.append(tsterr)

                    # Validation Error
                    pred_Y = classifier.predict(validation_X)
                    valerr = error(pred_Y, validation_Y)
                    #print('tsterr:', valerr)
                    val_row.append(valerr)
                    print('.', end='')

                # errors for each choice of lambda
                trn_errors.append(trn_row)
                val_errors.append(val_row)

        # prepare datapoints
        error_array = np.array(val_errors)
        means = np.mean(error_array, axis=0)
        stds = np.std(error_array, axis=0)

        # find optimal
        min_index = np.argmin(means)
        std_min = stds[min_index]
        opt_index = find_optimal(means, min_index, std_min)
        lambda_opt = lambdas[opt_index]
        
        opt_lambdas.append(lambda_opt)
        print("")
        print("Optimal Lambda:", lambda_opt)

        # plot validation errors
        plot_errors(means, stds, lambdas, min_index, opt_index, name='errors_%s%d' % (save_name, i), type=type)
        
    return opt_lambdas

if __name__ == '__main__':
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(False)
    
    opt_lambdas_clin = cross_validation(Ctr_X, Ctr_Y, save_name='clinical', type='Clinical Data')
    opt_lambdas = cross_validation(Gtr_X, Gtr_Y, save_name='genomic', type='Genomic Data')