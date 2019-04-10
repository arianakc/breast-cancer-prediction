import numpy as np


def accuracy(preds, labels):
    return np.mean(labels == preds.round())


def error(preds, labels):
    return 1.0 - accuracy(preds,labels)


def mean_square_error(preds, labels):
    return np.mean(np.square(preds - labels))


def mean_absolute_error(preds, labels):
    return np.mean(np.abs(preds - labels))


def tied_rank(x):
    sorted_x = sorted(zip(x,range(len(x))))
    r = [0 for k in x]
    cur_val = sorted_x[0][0]
    last_rank = 0
    for i in range(len(sorted_x)):
        if cur_val != sorted_x[i][0]:
            cur_val = sorted_x[i][0]
            for j in range(last_rank, i):
                r[sorted_x[j][1]] = float(last_rank+1+i)/2.0
            last_rank = i
        if i==len(sorted_x)-1:
            for j in range(last_rank, i+1):
                r[sorted_x[j][1]] = float(last_rank+i+2)/2.0
    return r



metrics = {"acc": accuracy,
           "error": error,
           "mse": mean_square_error,
           "mae": mean_absolute_error,
           }


def get_metric(eval_metric):
    return metrics[eval_metric]