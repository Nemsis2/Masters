# libraries
import torch as th

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


def test_em_lstm(feature_type, n_feature):
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

        outer_results = []
        threshold = 0
        for inner in range(NUM_INNER_FOLDS):
            # get the dev model
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            threshold += model.dev(return_threshold=True) # get the decision threshold for this inner fold
            results, labels, names = model.test() # do a forward pass through the models
            results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
            outer_results.append(results)

        # get auc
        results = sum(outer_results)/NUM_INNER_FOLDS # average prediction over the number of models in the outer fold
        auc += roc_auc_score(labels, results)

        # use threshold and get sens and spec
        results = (np.array(results)>(threshold/NUM_INNER_FOLDS)).astype(np.int8)
        sens_, spec_ = calculate_sens_spec(labels, results)
        sens += sens_
        spec += spec_

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS


def test_sm_lstm(feature_type, n_feature):
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
        # find best performing inner for this outer
        best_auc = 0
        for inner in range(NUM_INNER_FOLDS):
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            dev_auc = model.dev(return_auc=True) # get the dev auc for this inner model
            
            if dev_auc > best_auc:
                best_auc = auc
                best_inner = inner
        
        model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{best_inner}')
        threshold = model.dev(return_threshold=True)
        results, labels, names = model.test() # do a forward pass through the models
        
        # get auc
        results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
        auc += roc_auc_score(labels, results)

        # use threshold and get sens and spec
        results = (np.array(results)>(threshold)).astype(np.int8)
        sens_, spec_ = calculate_sens_spec(labels, results)
        sens += sens_
        spec += spec_

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS


def test_ts_lstm(feature_type, n_feature):
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
        # get the threshold from the dev set
        threshold = 0
        for inner in range(NUM_INNER_FOLDS):
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            threshold += model.dev(return_threshold=True)
        
        # load in and test ts model
        model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/ts/lstm_{feature_type}_{n_feature}_outer_fold_{outer}')
        results, labels, names = model.test() # do a forward pass through the models
        
        # get auc
        results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
        auc += roc_auc_score(labels, results)

        # use threshold and get sens and spec
        results = (np.array(results)>(threshold/NUM_INNER_FOLDS)).astype(np.int8)
        sens_, spec_ = calculate_sens_spec(labels, results)
        sens += sens_
        spec += spec_

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS


def test_ts_lstm_2(feature_type, n_feature):
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
        # get the threshold from the dev set
        threshold = 0
        for inner in range(NUM_INNER_FOLDS):
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            threshold += model.dev(return_threshold=True)
        
        # load in and test ts model
        model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/ts_2/lstm_{feature_type}_{n_feature}_outer_fold_{outer}')
        results, labels, names = model.test() # do a forward pass through the models
        
        # get auc
        results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
        auc += roc_auc_score(labels, results)

        # use threshold and get sens and spec
        results = (np.array(results)>(threshold/NUM_INNER_FOLDS)).astype(np.int8)
        sens_, spec_ = calculate_sens_spec(labels, results)
        sens += sens_
        spec += spec_

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS


def test_fss_lstm(feature_type, n_feature, fss_feature):
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

        outer_results = []
        threshold = 0
        for inner in range(NUM_INNER_FOLDS):
            # get the dev model
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/fss/lstm_{feature_type}_{n_feature}_fss_{fss_feature}_outer_fold_{outer}_inner_fold_{inner}')
            threshold += model.dev_fss(fss_feature) # get the decision threshold for this inner fold
            results, labels, names = model.test_fss(fss_feature) # do a forward pass through the models
            results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
            outer_results.append(results)

        # get auc
        results = sum(outer_results)/NUM_INNER_FOLDS # average prediction over the number of models in the outer fold
        auc += roc_auc_score(labels, results)

        # use threshold and get sens and spec
        results = (np.array(results)>(threshold/NUM_INNER_FOLDS)).astype(np.int8)
        sens_, spec_ = calculate_sens_spec(labels, results)
        sens += sens_
        spec += spec_

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS




def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            auc, sens, spec = test_em_lstm(feature_type, n_feature)

            print(f'AUC for em {n_feature}_{feature_type}: {auc}')
            print(f'Sens for em {n_feature}_{feature_type}: {sens}')
            print(f'Spec for em {n_feature}_{feature_type}: {spec}')

            auc, sens, spec = test_sm_lstm(feature_type, n_feature)

            print(f'AUC for sm {n_feature}_{feature_type}: {auc}')
            print(f'Sens for sm {n_feature}_{feature_type}: {sens}')
            print(f'Spec for sm {n_feature}_{feature_type}: {spec}')

            auc, sens, spec = test_ts_lstm(feature_type, n_feature)

            print(f'AUC for ts {n_feature}_{feature_type}: {auc}')
            print(f'Sens for ts {n_feature}_{feature_type}: {sens}')
            print(f'Spec for ts {n_feature}_{feature_type}: {spec}')

            auc, sens, spec = test_ts_lstm_2(feature_type, n_feature)

            print(f'AUC for ts_2 {n_feature}_{feature_type}: {auc}')
            print(f'Sens for ts_2 {n_feature}_{feature_type}: {sens}')
            print(f'Spec for ts_2 {n_feature}_{feature_type}: {spec}')

            for fraction_of_feature in [0.1, 0.2, 0.5]:
                if feature_type == 'mfcc':
                    auc, sens, spec = test_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature*3))
                    print(f'AUC for fss {n_feature}_{feature_type}_fss_{int(fraction_of_feature*n_feature*3)}: {auc}')
                    print(f'Sens for fss {n_feature}_{feature_type}_fss_{int(fraction_of_feature*n_feature*3)}: {sens}')
                    print(f'Spec for fss {n_feature}_{feature_type}_fss_{int(fraction_of_feature*n_feature*3)}: {spec}')
                else:
                    auc, sens, spec = test_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature))
                    print(f'AUC for fss {n_feature}_{feature_type}_fss_{int(fraction_of_feature*n_feature)}: {auc}')
                    print(f'Sens for fss {n_feature}_{feature_type}_fss_{int(fraction_of_feature*n_feature)}: {sens}')
                    print(f'Spec for fss {n_feature}_{feature_type}_fss_{int(fraction_of_feature*n_feature)}: {spec}')

    
if __name__ == "__main__":
    main()



"""
def get_oracle_thresholds(results, labels, threshold):
    sens_threshold, spec_threshold = np.zeros(len(threshold)), np.zeros(len(threshold))
    for i in range(len(threshold)):
        thresholded_results = (np.array(results)>threshold[i]).astype(np.int8)
        sens, spec = calculate_sens_spec(labels, thresholded_results)
        sens_threshold[i] = np.abs(sens-0.9)
        spec_threshold[i] = np.abs(spec-0.7)

    print(sens_threshold)
    sens = np.nanargmin(sens_threshold)
    spec = np.nanargmin(spec_threshold)
    print("sens", sens)
    return threshold[sens], threshold[spec]
"""