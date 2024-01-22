# libraries
import torch as th
import os
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

def test_lr(feature_type, n_feature, model_type='val'):
    """
    Inputs:
    ---------

    Outputs:
    --------

    """
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        models = []
        for inner in range(NUM_INNER_FOLDS):
            print("Inner fold=", inner)
            model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/{model_type}/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)
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
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13,26,39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            #create models for validation (threshold calculation)
            test_lr(feature_type, n_feature,'val')
            
            # create models for ensemble testing
            test_lr(feature_type, n_feature,'em')