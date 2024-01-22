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
from get_best_features import *
from lr_model_scripts import *



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