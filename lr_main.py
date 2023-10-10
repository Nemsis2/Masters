# libraries
import torch as th
import os
import pickle

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from pruning import *
from model_scripts import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score


"""
date: 21/02/2023 

author: Michael Knight
"""

# declare global variables

# set paths
K_FOLD_PATH = "../data/tb/combo/multi_folds/"
MODEL_PATH = "../models/tb/lr/"

# choose which melspec we will be working on
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"
TEST = "test/test_dataset_mel_180_fold_"


# set hyperpaperameters
BATCH_SIZE = 64
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# training options for the models
TRAIN_INNER_MODEL_FLAG = 0

# testing options for the models
TEST_GROUP_DECISION_FLAG = 1
VAL_MODEL_TEST_FLAG = 1

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)

    
def grid_search_lr(X, y):
    param_grid = {
        'C':[0.01, 0.1, 1, 10],
        'l1_ratio':[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    }

    model = LogisticRegression(C = 0.2782559402207126, 
    l1_ratio = 1, max_iter=1000000, 
    solver='saga', 
    penalty='elasticnet', 
    multi_class = 'multinomial', 
    n_jobs = -1)
    clf = GridSearchCV(model, param_grid=param_grid, cv=3, verbose=True, n_jobs=-1)
    best_clf = clf.fit(X, y)

    return best_clf, clf.best_params_


def main():
    """
    train a model for each inner fold within each outer fold resulting in inner_fold*outer_fold number of models.
    trains only on training data
    """
    if TRAIN_INNER_MODEL_FLAG == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "val/"))
        for train_outer_fold in range(NUM_OUTER_FOLDS):
                print("train_outer_fold=", train_outer_fold)
                
                for train_inner_fold in range(NUM_INNER_FOLDS):
                    print("train_inner_fold=", train_inner_fold)
                    data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)
                    X = np.array([np.mean(x, axis=0) for x in data])
                    labels = labels.astype("int")

                    model, params = grid_search_lr(X, labels)
                    print(params)

                    pickle.dump(model, open(MODEL_PATH + "val/" + working_folder + "/" + "lr_" + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                    "_inner_fold_" + str(train_inner_fold), 'wb')) # save the model


    """
    validates each model by assessing its performance on its corresponding validation set.
    """
    if VAL_MODEL_TEST_FLAG == 1:
        print("Beginning Validation")
        folder_names = os.listdir(MODEL_PATH + "val/")
        folder_names.sort()
        print(folder_names)
        best_fold0, best_fold1, best_fold2 = 0, 0, 0

        for folder in folder_names:
            auc = 0
            for val_outer_fold in range(NUM_OUTER_FOLDS):
                print("val_outer_fold=", val_outer_fold)
                models = []

                # for each outer fold pass through all the inner folds
                threshold = 0
                for val_inner_fold in range(NUM_INNER_FOLDS):
                    print("val_inner_fold=", val_inner_fold)
                    model = pickle.load(open(MODEL_PATH + "val/" + folder + "/lr_" + MODEL_MELSPEC + "_outer_fold_" + str(val_outer_fold) + 
                                    "_inner_fold_" + str(val_inner_fold), 'rb')) # load in the model
                    
                    val_data, val_labels, val_names = extract_val_data(K_FOLD_PATH + MELSPEC, val_outer_fold, val_inner_fold)
                    X = np.array([np.mean(x, axis=0) for x in val_data])
                    val_labels = val_labels.astype("int")

                    results = model.predict_proba(X) # do a forward pass through the models
                    unq,ids,count = np.unique(val_names,return_inverse=True,return_counts=True)
                    out = np.column_stack((unq,np.bincount(ids,results[:,1])/count, np.bincount(ids,val_labels)/count))

                    results = out[:,1]
                    val_labels = out[:,2]
                    threshold = get_optimal_threshold(val_labels, results)
                    #results = (np.array(results)>threshold).astype(np.int8)
                    auc += roc_auc_score(val_labels, results)
                    print("Threshold", threshold)

                if best_fold0 < auc and val_outer_fold == 0:
                    best_fold0 = auc/4
                    folder0 = folder
                
                if best_fold1 < auc and val_outer_fold == 1:
                    best_fold1 = auc/4
                    folder1 = folder

                if best_fold2 < auc and val_outer_fold == 2:
                    best_fold2 = auc/4
                    folder2 = folder
                    
                auc = 0

            print("Folder", folder0,  "AUC:", best_fold0)
            print("Folder", folder1,  "AUC:", best_fold1)
            print("Folder", folder2,  "AUC:", best_fold2)
     

    """
    test all inner models on the corresponding outer test set

    Implement: Add the threshold calculation from the val set and get sens and spec
    """
    if TEST_GROUP_DECISION_FLAG == 1:
        print("Beginning Testing")
        folder_names = os.listdir(MODEL_PATH + "GD/")
        folder_names.sort()
        optimal_threshold = [0.3076, 0.4809, 0.4520]
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        for working_folder in folder_names:
            # pass through all the outer folds
            print(working_folder)
            
            for outer in range(NUM_OUTER_FOLDS):
                print("test_outer_fold_ensemble=", outer)
                
                # for each outer fold pass through all the inner folds
                models = []
                for inner in range(NUM_INNER_FOLDS):
                    models.append(pickle.load(open(MODEL_PATH + "GD/" + working_folder + "/lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                        "_inner_fold_" + str(inner), 'rb'))) # load in the model
               
                test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + TEST, outer)
                X = np.array([np.mean(x, axis=0) for x in test_data])
                test_labels = test_labels.astype("int")

                results = []
                for model in models:
                    results.append(model.predict_proba(X)) # do a forward pass through the models

                for i in range(len(results)):
                    unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
                    out = np.column_stack((unq,np.bincount(ids,results[i][:,1])/count, np.bincount(ids,test_labels)/count))
                    results[i] = out[:,1]

                test_labels = out[:,2]

                # total the predictions over all models
                results = sum(results)/4
                auc[outer] = roc_auc_score(test_labels, results)
                results = (np.array(results)>optimal_threshold[outer]).astype(np.int8)
                sens[outer], spec[outer] = calculate_sens_spec(test_labels, results)

            print("AUC:", np.mean(auc), "var:", np.var(auc))
            print("sens:", np.mean(sens), "var:", np.var(sens))
            print("spec:", np.mean(spec), "var:", np.var(spec))



if __name__ == "__main__":
    main()