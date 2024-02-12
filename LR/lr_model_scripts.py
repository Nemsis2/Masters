#libraries
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np
import pickle

# custom scripts
from data_grab import extract_dev_data, normalize_mfcc
from helper_scripts import get_EER_threshold


def grid_search_lr(X, y):
    """
    Description:
    ---------

    Inputs:
    ---------
        X: np.array of melspecs for each cough
    
        y: list or array which contains a label for each value in the data np.array

        feature_type: 

    Outputs:
    --------
        best_clf: lr model with the best performing architecture as found by GridSearchCV

        best_clf.params: Optimal hyperparameters determined by the GridSearch
    """
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
                                tol=0.0001)
    clf = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_clf = clf.fit(X, y)

    return best_clf, best_clf.best_params_


def gather_results(results, labels, names):
    """
    Description:
    ---------

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


def get_decision_threshold(feature_type, n_feature, n_outer, n_inner):
    """
    Description:
    ---------

    Inputs:
    ---------
    feature_type: (string) The type of feature used to train the model (melspec, mfcc, lfb)

    n_features: (int) The number of features corresponding to the feature type (eg. 180 melspec features or 39 mfccs)

    n_outer: (int) The number of outer folds in the k-fold cross validation

    n_inner: (int) The number of inner folds in the k-fold cross validation

    Outputs:
    --------
    outer_threshold: (list) The average EER decision threshold over all inner models in each outer fold. Has length = num outer folds
    """
    outer_threshold = []
    for outer in range(n_outer): 
        outer_avg_threshold = 0
        for inner in range(n_inner):
            # grab the dev models
            dev_model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            dev_model = (pickle.load(open(dev_model_path, 'rb'))) # load in the model
        
            # grab the dev data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_dev_data(k_fold_path, inner)
            
            data = normalize_mfcc(data)
            X = np.array([np.mean(x, axis=0) for x in data])
            labels = labels.astype("int")

            # make predictions and calculate the threshold based off the EER
            results, val_labels = gather_results(dev_model.predict_proba(X), labels, names) # do a forward pass through the model
            outer_avg_threshold += get_EER_threshold(val_labels, results)
        outer_threshold.append(outer_avg_threshold/n_inner)

    return outer_threshold


def labels_per_frame(data, labels):
    """
    Inputs:
    ---------
        data: np.array which contains melspec samples of each cough
    
        labels: list or array which contains a label for each value in the data np.array

    Outputs:
    --------
        per_frame_label: np.array which contains a label for each frame 
    """
    per_frame_label = []
    for i in range(len(labels)):
        for j in range(data[i].shape[0]):
            per_frame_label.append(labels[i])
    return np.array(per_frame_label)