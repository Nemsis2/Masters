from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

def grid_search_lr(X, y):
    """
    Inputs:
    ---------
        X: np.array of melspecs for each cough
    
        y: list or array which contains a label for each value in the data np.array

    Outputs:
    --------
        best_clf: lr model with the best performing architecture as found by GridSearchCV

        best_clf.params: Optimal hyperparameters determined by the GridSearch
    """
    X = np.array([np.mean(x, axis=0) for x in X])
    y = y.astype("int")
    
    param_grid = {
        'C':[0.01, 0.1, 1, 10],
        'l1_ratio':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }

    model = LogisticRegression(C = 0.2782559402207126, 
    l1_ratio = 1, max_iter=1000000, 
    solver='saga', 
    penalty='elasticnet', 
    multi_class = 'multinomial', 
    n_jobs = -1,
    tol=0.001)
    clf = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_clf = clf.fit(X, y)

    return best_clf, best_clf.best_params_


def gather_results(results, labels, names):
    """
    Inputs:
    ---------
        results: multiple model prob predictions for each value in the data with shape num_models x num_data_samples
    
        labels: list or array which contains a label for each value in the data

        names: list or array of patient_id associated with each value in the data

    Outputs:
    --------
        out[:,1]: averaged model prob predictions for each unique patient_id in names

        out[:,2]: label associated with each value in out[:,1]
    """
    unq,ids,count = np.unique(names,return_inverse=True,return_counts=True)
    out = np.column_stack((unq,np.bincount(ids,results[:,1])/count, np.bincount(ids,labels)/count))
    return out[:,1], out[:,2]