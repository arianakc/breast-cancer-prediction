from preprocess import load_data
import matplotlib.pyplot as plt
from preprocess import load_data
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc
from random import shuffle
from sklearn.svm import SVC
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold
from sklearn.metrics import roc_curve, auc, accuracy_score
from sklearn.ensemble import RandomForestClassifier

def cv_roc(X, y, params, classifiers, get_max=True, save_path="cv_roc.png", param_name="\lambda", title="Areas Under ROC Curve", log=True, n_splits=5):
    kf = KFold(n_splits=n_splits)
    
    scores = []
    
    print("start")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        score_row = []
        for clf in classifiers:
            clf.fit(X_train, y_train)
            print(".", end='')
            y_pred = clf.predict(X_test)
            fpr, tpr, thresholds = roc_curve(y_test, y_pred)
            score = auc(fpr, tpr)
            score_row.append(score)
            print(",", end='')
        scores.append(score_row)
    
    scrs = np.array(scores)
    
    means = np.mean(scrs, axis=0)
    stds = np.std(scrs, axis=0)
    
    max_index = np.argmax(means)
    std_max = stds[max_index]
    if get_max:
        opt_index = find_maximum(means, max_index, std_max)
    else:
        opt_index = find_minimum(means, max_index, std_max)
    
    opt_param = params[opt_index]
    
    plot_errors(means, stds, params, max_index, opt_index, y_lbl="ROC AUC", title=title, save=save_path+".png", param_name=param_name, log=log)
    
    # save everything to file
    scrs_X = pd.DataFrame(params)
    scrs_Y = pd.DataFrame(scrs)
    scrs_X.T.to_csv(save_path+"_X.csv", index=False, header=False)
    scrs_Y.to_csv(save_path+"_Y.csv", index=False, header=False)
    
    with open(save_path+".txt", 'w') as f:
        f.write("%s %s" % (str(opt_param), str(params[max_index])))

def cv(X, y, params, classifiers, get_max=True, save_path="cv.png", param_name="\lambda", title="Errors", log=True, n_splits=5):
    kf = KFold(n_splits=n_splits)
    
    errs = []
    
    print("start")
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        err_row = []
        for clf in classifiers:
            clf.fit(X_train, y_train)
            print(".", end='')
            y_pred = clf.predict(X_test)
            err = 1 - accuracy_score(y_test, y_pred)
            err_row.append(err)
            print(",", end='')
        errs.append(err_row)
    
    errors = np.array(errs)
    
    means = np.mean(errors, axis=0)
    stds = np.std(errors, axis=0)
    
    min_index = np.argmin(means)
    std_min = stds[min_index]
    if get_max:
        opt_index = find_maximum(means, min_index, std_min)
    else:
        opt_index = find_minimum(means, min_index, std_min)
    
    opt_param = params[opt_index]
    
    plot_errors(means, stds, params, min_index, opt_index, y_lbl="Classification Error", title=title, save=save_path+".png", param_name=param_name, log=log)
    
    # save everything to file
    errors_X = pd.DataFrame(params)
    errors_Y = pd.DataFrame(errors)
    errors_X.T.to_csv(save_path+"_X.csv", index=False, header=False)
    errors_Y.to_csv(save_path+"_Y.csv", index=False, header=False)
    
    with open(save_path+".txt", 'w') as f:
        f.write("%s %s" % (str(opt_param), str(params[min_index])))
    
    return opt_param, params[min_index]

def plot_errors(means, stds, lambdas, min_index, opt_index, y_lbl="Classification Error", title="Errors", param_name="\lambda", save="errs.png", log=True):
    # prepare datapoints
    means_min = means[min_index]
    lambda_min = lambdas[min_index]
    std_min = stds[min_index]
    means_opt = means[opt_index]
    lambda_opt = lambdas[opt_index]

    # setup plot
    x_ax = lambdas
    if log:
        x_ax = np.log10(lambdas)
    plt.figure(figsize=(7, 7))
    ax = plt.axes()

    # plot mu and mu_err
    plt.plot(x_ax, means, color='black', label="$\mu$")
    plt.plot(x_ax, means + stds, color='red', label="$\mu \pm \sigma$")
    plt.plot(x_ax, means - stds, color='red')

    # plot lambda min
    plt.plot(x_ax, [means_min + std_min]*len(x_ax), color='blue', linestyle='dashed')
    plt.plot(x_ax, [means_min - std_min]*len(x_ax), color='blue', linestyle='dashed')
    if log:
        plt.plot(np.log10([lambda_min]*10), np.linspace(*ax.get_ylim(), 10), color='blue', linestyle='dashed')
        plt.scatter(np.log10([lambda_min]), [means_min], s=80, facecolors='none', edgecolors='b', label="$%s_{opt}$" % param_name)
    else:
        plt.plot([lambda_min]*10, np.linspace(*ax.get_ylim(), 10), color='blue', linestyle='dashed')
        plt.scatter([lambda_min], [means_min], s=80, facecolors='none', edgecolors='b', label="$%s_{opt}$" % param_name)

    # plot optimal
    if log:
        plt.plot(np.log10([lambda_opt]*10), np.linspace(*ax.get_ylim(), 10), color='green', linestyle='dashed')
        plt.scatter(np.log10([lambda_opt]), [means_opt], s=80, facecolors='none', edgecolors='g', label="$%s^*$" % param_name)
    else:
        plt.plot([lambda_opt]*10, np.linspace(*ax.get_ylim(), 10), color='green', linestyle='dashed')
        plt.scatter([lambda_opt], [means_opt], s=80, facecolors='none', edgecolors='g', label="$%s^*$" % param_name)

       
    x_lbl = "$%s$" % param_name
    if log:
        x_lbl="$\log_{10}(%s)$" % param_name
    # finish
    plt.legend(loc='lower right')
    plt.xlabel(x_lbl)
    plt.ylabel(y_lbl)
    plt.title(title)
    plt.savefig(save)

def find_maximum(y, index, std):
    optimal_index = index
    upper_bound = y[index]+std
    lower_bound = y[index]-std
    for i in range(index+1, len(y)):
        if (y[i] <= upper_bound) and (y[i] >= lower_bound):
            optimal_index = i
    return optimal_index

def find_minimum(y, index, std):
    optimal_index = index
    upper_bound = y[index]+std
    lower_bound = y[index]-std
    for i in range(index-1, 0, -1):
        if (y[i] <= upper_bound) and (y[i] >= lower_bound):
            optimal_index = i
    return optimal_index