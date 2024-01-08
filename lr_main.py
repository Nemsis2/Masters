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
from get_best_features import *


"""
date: 21/02/2023 

author: Michael Knight
"""

# declare global variables

# set paths
#K_FOLD_PATH = "../data/tb/combo/multi_folds/"
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
TRAIN_INNER_FSS_MODEL_FLAG = 0
DO_FSS = 0

# testing options for the models
TEST_GROUP_DECISION_FLAG = 1
TEST_GROUP_FSS_DECISION_FLAG = 0
VAL_MODEL_TEST_FLAG = 1

# Hyperparameters
BEST_C = [[10, 10, 10, 0.1],[0.01, 1, 10, 10],[1, 10, 10, 0.01]]
BEST_L1_RATIO = [[0.9, 1, 0.9, 0.6],[0.5, 1, 0.9, 0.9],[0.2, 0.7, 1, 0.2]]
NUM_FEATURES = 20

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


def get_oracle_thresholds(results, labels, threshold):
    sens_threshold, spec_threshold = np.zeros(len(threshold)), np.zeros(len(threshold))
    for i in range(len(threshold)):
        thresholded_results = (np.array(results)>threshold[i]).astype(np.int8)
        sens, spec = calculate_sens_spec(labels, thresholded_results)
        sens_threshold[i] = np.abs(sens-0.9)
        spec_threshold[i] = np.abs(spec-0.7)
    
    sens = np.nanargmin(sens_threshold)
    spec = np.nanargmin(spec_threshold)

    return threshold[sens], threshold[spec]


def validate_model(folder, outer_fold):
    """
    Inputs:
    ---------
        folder: current folder being worked on. String consisting of 4 digits e.g. 0001
    
        outer_fold: current outer fold being considered. Int between 1 and 3

    Outputs:
    --------
        threshold: average optimal threshold over all models in the outer fold

        auc: average auc over all models in the outer fold
    """
    threshold, auc = 0, 0
    for inner_fold in range(NUM_INNER_FOLDS):
        val_data, val_labels, val_names = extract_val_data(K_FOLD_PATH + MELSPEC, outer_fold, inner_fold)
        X = np.array([np.mean(x, axis=0) for x in val_data])
        val_labels = val_labels.astype("int")
        model = pickle.load(open(MODEL_PATH + "val/" + folder + "/lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer_fold) + 
                                        "_inner_fold_" + str(inner_fold), 'rb')) # load in the model

        results, val_labels = gather_results(model.predict_proba(X), val_labels, val_names) # do a forward pass through the model
        fpr, tpr, thresholds = roc_curve(val_labels, results, pos_label=1)
        sens_threshold, spec_threshold = get_oracle_thresholds(results, val_labels, thresholds)
        threshold += sens_threshold
        # get EER threshold
        # threshold +=  get_optimal_threshold(val_labels, results)

        auc += roc_auc_score(val_labels, results)

    return threshold/4, auc/4


def test_model(folder, outer, optimal_threshold):
    """
    Inputs:
    ---------
        folder: current folder being worked on. String consisting of 4 digits e.g. 0001
    
        outer_fold: current outer fold being considered. Int between 1 and 3

        optimal_threshold: the optimal threshold for the current outer fold

    Outputs:
    --------
        auc: auc for the outer fold

        sens: sens for the outer fold

        spec: spec for the outer fold
    """
    # for each outer fold pass through all the inner folds
    models = []
    for inner in range(NUM_INNER_FOLDS):
        models.append(pickle.load(open(MODEL_PATH + "GD/" + folder + "/lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                        "_inner_fold_" + str(inner), 'rb'))) # load in the model
               
    data, labels, names = extract_test_data(K_FOLD_PATH + TEST, outer)
    X = np.array([np.mean(x, axis=0) for x in data])
    labels = labels.astype("int")

    results = []
    for model in models:
        results.append(model.predict_proba(X)) # do a forward pass through the models

    output = []
    for i in range(len(results)):
        new_results, new_labels = gather_results(results[i], labels, names)
        output.append(new_results)

    # total the predictions over all models
    results = sum(output)/4
    auc = roc_auc_score(new_labels, results)
    results = (np.array(results)>optimal_threshold).astype(np.int8)
    sens, spec = calculate_sens_spec(new_labels, results)

    return auc, sens, spec


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


    if TRAIN_INNER_FSS_MODEL_FLAG == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/"))
        for outer in range(NUM_OUTER_FOLDS):
                print("train_outer_fold=", outer)
                
                for inner in range(NUM_INNER_FOLDS):
                    print("train_inner_fold=", inner)
                    #data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, outer, inner)
                    data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, outer)
                    X = np.array([np.mean(x, axis=0) for x in data])
                    labels = labels.astype("int")
                    selected_features = dataset_fss(NUM_FEATURES)
                    
                    chosen_features = []
                    for i in range(NUM_FEATURES):
                        chosen_features.append(np.asarray(X[:,selected_features[i]]))
                    chosen_features = th.as_tensor(np.stack(chosen_features, -1))

                    print(chosen_features.shape)
                    model, params = grid_search_lr(X, labels)
                    
                    model.fit(chosen_features, labels)
                    pickle.dump(model, open(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/" + working_folder + "/" + "lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                    "_inner_fold_" + str(inner), 'wb')) # save the model


    if DO_FSS == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "FSS/"))
        print(working_folder)

        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)
            for inner in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", inner)

                feature_priority = []
                auc_priority = []
                features = np.arange(0,180)
                
                # do the FSS here
                # Load in Data and val data for current outer and inner
                train_data, train_labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, outer, inner)
                train_data = np.array([np.mean(x, axis=0) for x in train_data])
                train_labels = train_labels.astype("int")
                
                val_data, val_labels, val_names = extract_val_data(K_FOLD_PATH + MELSPEC, outer, inner)
                val_data = np.array([np.mean(x, axis=0) for x in val_data])
                val_labels = val_labels.astype("int")

                # iterate feature-1 times
                while len(feature_priority) != 180:
                    performance = []
                    # Pass through all unselected features
                    for feature in features:
                        # create new model
                        model = LogisticRegression(C = BEST_C[outer][inner], l1_ratio = BEST_L1_RATIO[outer][inner],
                                                    max_iter=1000000, solver='saga', penalty='elasticnet', multi_class = 'multinomial', n_jobs = -1)
                            
                        # create array of chosen features    
                        chosen_features, chosen_features_val = [], []
                        for prev_select_feature in feature_priority:
                            chosen_features.append(np.asarray(train_data[:,int(prev_select_feature)]))
                            chosen_features_val.append(np.asarray(val_data[:,int(prev_select_feature)]))
                        chosen_features.append(np.asarray(train_data[:,feature]))
                        chosen_features_val.append(np.asarray(val_data[:,feature]))
                        chosen_features = th.as_tensor(np.stack(chosen_features, -1))
                        chosen_features_val = th.as_tensor(np.stack(chosen_features_val, -1))

                        print("Chosen features:", chosen_features.shape)
                        print("Val chosen features:", chosen_features_val.shape)

                        model.fit(chosen_features, train_labels)
                        results = model.predict_proba(chosen_features_val)
                        auc = roc_auc_score(val_labels, results[:,1])
                        performance.append(auc)
                        print("Feature:", feature, "AUC:", auc)

                        # force delete loaded in model
                        del model
                        gc.collect()

                    # select best performing feature from list
                    best_feature = np.argmax(np.array(performance))
                    print("Features array:", features)
                    print("Best feature:", best_feature)
                    print("Array selection", features[best_feature])
                    
                    feature_priority.append(str(features[best_feature]))
                    auc_priority.append(str(performance[best_feature]))
                    print("Best performing feature:", best_feature, "with an auc of:", performance[best_feature])

                    # train model on best performing features and save
                    model = LogisticRegression(C = BEST_C[outer][inner], l1_ratio = BEST_L1_RATIO[outer][inner], max_iter=1000000, 
                                               solver='saga', penalty='elasticnet', multi_class = 'multinomial', n_jobs = -1)                      
                    model.fit(chosen_features, train_labels)
                    
                    # save the new model
                    pickle.dump(model, open(MODEL_PATH + "FSS/" + working_folder + "/" + "lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                    "_inner_fold_" + str(inner) + "_features_" + str(len(feature_priority)), 'wb'))
                    
                    # delete the previous model
                    previous_model_path = str(MODEL_PATH + "FSS/" + working_folder + "/" + "lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                    "_inner_fold_" + str(inner) + "_features_" + str(len(feature_priority)-1))              
                    if os.path.exists(previous_model_path):
                        os.remove(previous_model_path)

                    # delete the chosen feature so it cannot be reselected
                    features = np.delete(features, best_feature)
                    
                    # force delete loaded in model
                    del model
                    gc.collect()

                    #save current feature list
                    file_name = MODEL_PATH + "FSS/" + working_folder + "/features_outer_" + str(outer) + "_inner_" + str(inner) + ".txt"
                    with open(file_name, 'w') as f:
                        for feature in feature_priority:
                            f.write("%s\n" % feature)

                    # save current auc list
                    file_name = MODEL_PATH + "FSS/" + working_folder + "/auc_outer_" + str(outer) + "_inner_" + str(inner) + ".txt"
                    with open(file_name, 'w') as f:
                        for auc in auc_priority:
                            f.write("%s\n" % auc)


    """
    validates each model by assessing its performance on its corresponding validation set.
    """
    if VAL_MODEL_TEST_FLAG == 1:
        print("Beginning Validation")
        folder_names = os.listdir(MODEL_PATH + "val/")
        folder_names.sort()
        print(folder_names)

        for folder in folder_names:
            for outer_fold in range(NUM_OUTER_FOLDS):
                print("val_outer_fold=", outer_fold)
                threshold, auc = validate_model(folder, outer_fold)
                print("Threshold", threshold)
                print("AUC", auc)

     
    """
    test all inner models on the corresponding outer test set

    Implement: Add the threshold calculation from the val set and get sens and spec
    """
    if TEST_GROUP_DECISION_FLAG == 1:
        print("Beginning Testing")
        folder_names = os.listdir(MODEL_PATH + "GD/")
        folder_names.sort()
        
        for folder in folder_names:
            # pass through all the outer folds
            print(folder)
            
            auc, sens, spec = np.zeros(3), np.zeros(3), np.zeros(3)
            for outer in range(NUM_OUTER_FOLDS):
                print("test_outer_fold_ensemble=", outer)
                val_folder = os.listdir(MODEL_PATH + "val/")
                threshold, val_auc = validate_model(val_folder[0], outer)
                auc[outer], sens[outer], spec[outer] = test_model(folder, outer, threshold)        
                
            print("AUC:", np.mean(auc), "var:", np.var(auc))
            print("sens:", np.mean(sens), "var:", np.var(sens))
            print("spec:", np.mean(spec), "var:", np.var(spec))


    if TEST_GROUP_FSS_DECISION_FLAG == 1:
        print("Beginning Testing")
        folder_names = os.listdir(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/")
        folder_names.sort()
        optimal_threshold = [0.3076, 0.4809, 0.4520]
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        for working_folder in folder_names:
            # pass through all the outer folds
            print(working_folder)
            
            for outer in range(NUM_OUTER_FOLDS):
                print("test_outer_fold_ensemble=", outer)
                
                # for each outer fold pass through all the inner folds
                results = []
                for inner in range(NUM_INNER_FOLDS):
                    model = pickle.load(open(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/" + working_folder + "/lr_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                        "_inner_fold_" + str(inner), 'rb')) # load in the model
               
                    test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + TEST, outer)
                    X = np.array([np.mean(x, axis=0) for x in test_data])
                    test_labels = test_labels.astype("int")
                    selected_features = dataset_fss(NUM_FEATURES) 

                    chosen_features = []
                    for i in range(NUM_FEATURES):
                        chosen_features.append(np.asarray(X[:,selected_features[i]]))
                    chosen_features = th.as_tensor(np.stack(chosen_features, -1))

                    results.append(model.predict_proba(chosen_features)) # do a forward pass through the models

                print(len(results))
                for i in range(len(results)):
                    unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
                    out = np.column_stack((unq,np.bincount(ids,results[i][:,1])/count, np.bincount(ids,test_labels)/count))
                    results[i] = out[:,1]

                test_labels = out[:,2]

                # total the predictions over all models
                results = sum(results)/4
                auc[outer] = roc_auc_score(test_labels, results)
            print(sum(auc)/3)

if __name__ == "__main__":
    main()