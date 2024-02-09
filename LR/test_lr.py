# libraries
import torch as th
import pickle

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from get_best_features import *
from lr_model_scripts import *

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 64
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)



def test_lr(feature_type, n_feature, threshold):
    """
    Description:
    ---------
    Calculates the auc, sens and spec for a LR model on the given features.
    
    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.

    threshold: (float) decision threshold calculated on the development set.

    Outputs:
    --------
    auc: average auc over all outer folds.

    sens: avearge sensitivity over all outer folds.

    spec: average specificity over all outer folds.
    """

    auc, sens, spec = 0, 0, 0
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model
        

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)

        if feature_type=="mfcc":
            data = normalize_mfcc(data)
        
        # for by frame
        #labels = labels_per_frame(data, labels)
        #names = labels_per_frame(data, names) # misusing this function to keep num_names = num_labels
        #X = np.vstack(data)
        
        # for averaging
        X = np.array([np.mean(x, axis=0) for x in data])
        labels = labels.astype("int")

        results = []
        for model in models:
            results.append(model.predict_proba(X)) # do a forward pass through the models

        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        # total the predictions over all models in the outer fold
        # for by frame
        #results, labels = gather_results(results, labels, names)

        results = sum(output)/4
        inner_auc = roc_auc_score(new_labels, results)
        results = (np.array(results)>threshold[outer]).astype(np.int8)
        inner_sens, inner_spec = calculate_sens_spec(new_labels, results)
        
        # add to the total auc, sens and spec
        auc += inner_auc
        sens += inner_sens
        spec += inner_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS



def test_lr_multi_feature():
    """
    Description:
    ---------
    Calculates the auc, sens and spec by averaging the decision making over all different feature types.
    
    Outputs:
    --------
    auc: average auc over all outer folds.
    """

    auc = 0
    for outer in range(NUM_OUTER_FOLDS):
        results = []
        for feature_type in ['mfcc', 'melspec', 'lfb']:
            if feature_type == 'mfcc':
                n_feature = '13'
            if feature_type == 'melspec' or feature_type == 'lfb':
                n_feature = '180'
            
            # grab all models to be tested for that outer fold
            models = []
            for inner in range(NUM_INNER_FOLDS):
                # get the testing models
                model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
                models.append(pickle.load(open(model_path, 'rb'))) # load in the model
            

            # grab the testing data
            k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
            data, labels, names = extract_test_data(k_fold_path)
            X = np.array([np.mean(x, axis=0) for x in data])
            labels = labels.astype("int")
        
            if feature_type=="mfcc":
                data = normalize_mfcc(data)
        
            for model in models:
                results.append(model.predict_proba(X)) # do a forward pass through the models

        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        results = sum(output)/4
        inner_auc = roc_auc_score(new_labels, results)
        
        # add to the total auc, sens and spec
        auc += inner_auc

    return auc/NUM_OUTER_FOLDS


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            # get the optimal threshold based off the EER
            threshold = get_decision_threshold(feature_type, n_feature, NUM_OUTER_FOLDS, NUM_INNER_FOLDS)

            # test the em setup
            auc, sens, spec = test_lr(feature_type, n_feature, threshold)

            print(f'AUC for {n_feature}_{feature_type}: {auc}')
            print(f'Sens for {n_feature}_{feature_type}: {sens}')
            print(f'Spec for {n_feature}_{feature_type}: {spec}')

    auc = test_lr_multi_feature()

    print(f'AUC for multi feature: {auc}')


if __name__ == "__main__":
    main()