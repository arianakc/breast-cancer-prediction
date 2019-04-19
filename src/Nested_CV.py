from CV import *

def nested_cv_roc(X, y, params, classifiers, get_max=True, save_path="CV/cv_roc", param_name="\lambda", title="Areas Under ROC Curve", log=True, n_splits=5):
    kf = KFold(n_splits=n_splits)
    
    i=1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        cv_roc(X_train, y_train, params, classifiers, get_max=get_max, save_path='%s_%d'%(save_path, i), param_name=param_name, title=title, log=log, n_splits=n_splits-1)
        
        i+=1
        
def nested_cv(X, y, params, classifiers, get_max=True, save_path="CV/cv", param_name="\lambda", title="Areas Under ROC Curve", log=True, n_splits=5):
    kf = KFold(n_splits=n_splits)
    
    i=1
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        cv(X_train, y_train, params, classifiers, get_max=get_max, save_path='%s_%d'%(save_path, i), param_name=param_name, title=title, log=log, n_splits=n_splits-1)
        
        i+=1
        
        