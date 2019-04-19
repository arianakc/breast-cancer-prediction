import RegularizedLogisticRegression
from CV import *
from Nested_CV import nested_cv_roc, nested_cv
import os

if __name__ == '__main__':
    Ctr_X, Ctr_Y, Cval_X, Cval_Y, Ct_X, Ct_Y, Gtr_X, Gtr_Y, Gval_X, Gval_Y, Gt_X, Gt_Y = load_data(False)
    
    params = np.logspace(-5, 2, 200)
    classifiers = [ RegularizedLogisticRegression.LogisticRegressionClassifier(reg_coeff=lambda_i, l1_ratio=0.95)
                    for lambda_i in params ]
    
    if not os.path.exists("CV"):
        os.makedirs("CV")
        
    if not os.path.exists("CV/RegLR"):
        os.makedirs("CV/RegLR")
    
    nested_cv(Ctr_X, Ctr_Y, params, classifiers, get_max=True, save_path='CV/RegLR/reg_lr_lambda', param_name="\lambda", title="L1 Logistic Regression Validation Errors", log=True)