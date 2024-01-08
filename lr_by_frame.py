# libraries
import torch as th
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from pruning import *
from model_scripts import *
from get_best_features import *



"""
date: 19/12/2023 

author: Michael Knight
"""

# declare global variables

# set paths
K_FOLD_PATH = "../data/tb/combo/multi_folds/"
MODEL_PATH = "../models/tb/lr_by_frame/"

# choose which melspec we will be working on
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"
TEST_PATH = "test/test_dataset_mel_180_fold_"

# set hyperpaperameters
BATCH_SIZE = 64
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# Flags
TRAIN = 0
TEST = 0

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)

def grid_search_lr(X, y):
    """
    Inputs:
    ---------
        X: np.array of melspecs for each cough
    
        y: list or array which contains a label for each value in the data np.array

    Outputs:
    --------
        best_clf: lr model with the best performing architecture as found by GridSearchCV
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
    tol=0.001)
    clf = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_clf = clf.fit(X, y)

    return best_clf


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


for outer in range(NUM_OUTER_FOLDS):
    if TRAIN == 1:
        working_folder = create_new_folder(str(MODEL_PATH))
        # get training data
        data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, outer)
        labels = labels_per_frame(data, labels)
        data = np.vstack(data)

        # train the model
        model = grid_search_lr(data, labels)
        pickle.dump(model, open(MODEL_PATH + "/" + working_folder + "/" + "lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer), 'wb')) # save the model

    if TEST == 1:
        # get test data
        data, labels, names = extract_test_data(K_FOLD_PATH + TEST_PATH, outer)
        labels = labels_per_frame(data, labels)
        names = labels_per_frame(data, names) # misusing this function to keep num_names = num_labels
        data = np.vstack(data)

        # test on data
        model = pickle.load(open(MODEL_PATH + str(outer+1).zfill(4) + "/lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer), 'rb'))
        results = model.predict_proba(data)
        results, labels = gather_results(results, labels, names)
        auc_ = roc_auc_score(labels, results)
        print(auc_)