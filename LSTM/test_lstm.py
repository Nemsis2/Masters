# libraries
import torch as th
import os

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from lstm_model_scripts import *
from get_best_features import *

# declare global variables
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)


def test_lstm(feature_type, n_feature, model_type='em'):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    auc, sens, spec = 0, 0, 0
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        outer_results, threshold = [], []
        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/{model_type}/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            model = pickle.load(open(model_path, 'rb')) # load in the model
        
            threshold.append(model.val()) # get the decision threshold for this inner fold
            results, labels, names = model.test() # do a forward pass through the models
            results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
            outer_results.append(results)

        results = sum(outer_results)/NUM_INNER_FOLDS # average prediction over the number of models in the outer fold
        auc = roc_auc_score(labels, results)
        results = (np.array(results)>threshold[outer]).astype(np.int8)
        sens, spec = calculate_sens_spec(labels, results)

    return auc, sens, spec


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            test_lstm(feature_type, n_feature,'em')
        
    
if __name__ == "__main__":
    main()