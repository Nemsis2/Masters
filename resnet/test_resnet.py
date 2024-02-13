# libraries
import torch as th
import gc
import os

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 64
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
EPOCHS = 16

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


def test_resnet(feature_type, n_feature, model, threshold):
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

    sens: average sensitivity over all outer folds.

    spec: average specificity over all outer folds.
    """
    auc, sens, spec = 0, 0, 0
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/{model}/{feature_type}/{n_feature}_{feature_type}/dev/{model}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path, feature_type)
        if feature_type == 'mfcc':
            data = normalize_mfcc(data)
        data, labels, lengths = create_batches(data, labels, 'image', BATCH_SIZE)

        results = []
        for model in models:
            results.append(test(data, model, lengths)) # do a forward pass through the models

        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        results = sum(output)/4
        inner_auc = roc_auc_score(new_labels, results)
        results = (np.array(results)>threshold[outer]).astype(np.int8)
        inner_sens, inner_spec = calculate_sens_spec(new_labels, results)
        
        # add to the total auc, sens and spec
        auc += inner_auc
        sens += inner_sens
        spec += inner_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS


def get_resnet_threshold(model, feature_type, n_feature):
    thresholds = []
    for outer in range(NUM_OUTER_FOLDS):
        
        # grab all models to be tested for that outer fold
        threshold = 0
        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/{model}/{feature_type}/{n_feature}_{feature_type}/dev/{model}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            model = pickle.load(open(model_path, 'rb')) # load in the model

            # grab and preprocess data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_dev_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths = create_batches(data, labels, 'image', BATCH_SIZE)
            
            # calculate inner decision threshold
            results = test(data, model, lengths)
            results, labels = gather_results(results, labels, names)
            threshold += get_EER_threshold(labels, results)

        thresholds.append(threshold/NUM_INNER_FOLDS)

    return thresholds



def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            for model in ['resnet_18', 'resnet_10', 'resnet6_2Deep', 'resnet6_4Deep']:
                threshold = get_resnet_threshold(feature_type, n_feature)
                auc, sens, spec = test_resnet(feature_type, n_feature, model, threshold)

                print(f'AUC for {n_feature}_{feature_type}: {auc}')
                print(f'Sens for {n_feature}_{feature_type}: {sens}')
                print(f'Spec for {n_feature}_{feature_type}: {spec}')
            

if __name__ == "__main__":
    main()