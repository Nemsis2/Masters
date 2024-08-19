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


def post_results(feature_type, n_feature, performance_metrics):
    print(f'AUC for {n_feature}_{feature_type}: {performance_metrics[0]}')
    print(f'Sens for {n_feature}_{feature_type}: {performance_metrics[1]}')
    print(f'Spec for {n_feature}_{feature_type}: {performance_metrics[2]}')
    print(f'Oracle sens for {n_feature}_{feature_type}: {performance_metrics[3]}')
    print(f'Oracle spec for {n_feature}_{feature_type}: {performance_metrics[4]}')
    print(f'{feature_type} & {round(n_feature,4)} & {round(performance_metrics[0],4)} & {round(performance_metrics[1],4)} & {round(performance_metrics[2],4)} & {round(performance_metrics[3],4)} & {round(performance_metrics[4],4)}')



def post_fss_results(feature_type, n_feature, fraction_of_feature, performance_metrics):
    print(f'AUC for {n_feature}_{feature_type} with {int(fraction_of_feature*n_feature)}: {performance_metrics[0]}')
    print(f'Sens for {n_feature}_{feature_type} with {int(fraction_of_feature*n_feature)}: {performance_metrics[1]}')
    print(f'Spec for {n_feature}_{feature_type} with {int(fraction_of_feature*n_feature)}: {performance_metrics[2]}')
    print(f'Oracle sens for frame_skip fss {n_feature}_{feature_type}: {performance_metrics[3]}')
    print(f'Oracle spec for frame_skip fss {n_feature}_{feature_type}: {performance_metrics[4]}')
    print(f'{feature_type} & {round(n_feature,4)} & {round(performance_metrics[0],4)} & {round(performance_metrics[1],4)} & {round(performance_metrics[2],4)} & {round(performance_metrics[3],4)} & {round(performance_metrics[4],4)}')



def dev_lstm(feature_type, n_feature):
    """
    Description:
    ---------

    Inputs:
    ---------

    Outputs:
    --------

    """
    total_auc = 0
    count  = 0
    valid_folds = []
    for outer in range(NUM_OUTER_FOLDS):
        valid_folds.append([])
        for inner in range(NUM_INNER_FOLDS):
            # get the dev model
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            auc = model.dev() # do a forward pass through the models
            if auc > 0.5:
                total_auc += auc
                count +=1
                valid_folds[outer].append(1)
            else:
                valid_folds[outer].append(0)
    
    print(total_auc/count)
    print(valid_folds)
    return valid_folds


def dev_lstm_fss(feature_type, n_feature, fss_features):
    """
    Description:
    ---------

    Inputs:
    ---------

    Outputs:
    --------

    """
    total_auc = 0
    count  = 0
    valid_folds = []
    for outer in range(NUM_OUTER_FOLDS):
        valid_folds.append([])
        for inner in range(NUM_INNER_FOLDS):
            # get the dev model
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/fss/lstm_{feature_type}_{n_feature}_fss_{fss_features}_outer_fold_{outer}_inner_fold_{inner}')

            auc = model.dev_on_select_features(fss_features) # do a forward pass through the models
            if auc > 0.5:
                total_auc += auc
                count +=1
                valid_folds[outer].append(1)
            else:
                valid_folds[outer].append(0)
    
    print(total_auc/count)
    print(valid_folds)
    return valid_folds


def test_em_lstm(feature_type, n_feature, valid_folds):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    performance_metrics = np.zeros(5)
    count  = 0
    for outer in range(NUM_OUTER_FOLDS):
        outer_results = []
        if sum(valid_folds[outer]) != 0:
            count +=1
            print(f'outer: {outer}')
            for inner in range(NUM_INNER_FOLDS):
                # get the dev model
                if valid_folds[outer][inner] == 1:
                    print(f'inner: {inner}')
                    model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
                    results, labels, names = model.test() # do a forward pass through the models
                    results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
                    outer_results.append(results)

            # get auc
            results = sum(outer_results)/len(outer_results) # average prediction over the number of models in the outer fold
            performance_metrics += calculate_metrics(labels, results)

    performance_metrics = performance_metrics/count

    return performance_metrics



def test_sm_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):
        # find best performing inner for this outer
        best_auc = 0
        for inner in range(NUM_INNER_FOLDS):
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            dev_auc = model.dev() # get the dev auc for this inner model
            
            if dev_auc > best_auc:
                best_auc = dev_auc
                best_inner = inner
        
        model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{best_inner}')
        
        # get results gather by patient and calculate auc
        results, labels, names = model.test() # do a forward pass through the models
        results, labels = gather_results(results, labels, names) # gather results by patient so all a patients cough predictions are averaged
        performance_metrics += calculate_metrics(labels, results)

    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

    return performance_metrics




def test_exclusive_ts_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):  
        # get the threshold from the dev set
        threshold = 0
        for inner in range(NUM_INNER_FOLDS):
            model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
            threshold += model.dev(return_threshold=True)
        
        # load in and test ts model
        model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/ts/lstm_{feature_type}_{n_feature}_outer_fold_{outer}')
        results, labels, names = model.test() # do a forward pass through the models
        
        # get results gather by patient and calculate auc
        results, labels = gather_results(results, labels, names) # gather results by patient so all a patients cough predictions are averaged
        performance_metrics += calculate_metrics(labels, results)

    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

    return performance_metrics




def test_ts_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):   
        # load in and test ts model
        model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/ts_2/lstm_{feature_type}_{n_feature}_outer_fold_{outer}')
        results, labels, names = model.test() # do a forward pass through the models

        # get results gather by patient and calculate auc
        results, labels = gather_results(results, labels, names) # gather results by patient so all a patients cough predictions are averaged
        performance_metrics += calculate_metrics(labels, results)

    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

    return performance_metrics




def test_fss_lstm(feature_type, n_feature, fss_feature, valid_folds):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    performance_metrics = np.zeros(5)
    count  = 0
    for outer in range(NUM_OUTER_FOLDS):
        outer_results = []
        if sum(valid_folds[outer]) != 0:
            count +=1
            for inner in range(NUM_INNER_FOLDS):
                if valid_folds[outer][inner] == 1:
                # get the dev model
                    model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/fss/lstm_{feature_type}_{n_feature}_fss_{fss_feature}_outer_fold_{outer}_inner_fold_{inner}')
                    results, labels, names = model.test_fss(fss_feature) # do a forward pass through the models
                    results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
                    outer_results.append(results)

            # get results gather by patient and calculate auc
            outer_results = sum(outer_results)/len(outer_results) # average prediction over the number of models in the outer fold
            performance_metrics += calculate_metrics(labels, outer_results)

    performance_metrics = performance_metrics/count

    return performance_metrics



def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            #valid_folds = dev_lstm(feature_type, n_feature)
            
            # test the em setup
            # performance_metrics = test_em_lstm(feature_type, n_feature, valid_folds)
            # post_results(feature_type, n_feature, performance_metrics)

            # test the sm setup
            # performance_metrics = test_sm_lstm(feature_type, n_feature)
            # post_results(feature_type, n_feature, performance_metrics)

            # test the ts setup
            # performance_metrics = test_ts_lstm(feature_type, n_feature)
            # post_results(feature_type, n_feature, performance_metrics)


            for fraction_of_feature in [0.1, 0.2, 0.5]:
                if feature_type == 'mfcc':
                    valid_folds = dev_lstm_fss(feature_type, n_feature, int(fraction_of_feature*n_feature*3))
                    performance_metrics = test_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature*3), valid_folds)
                    post_fss_results(feature_type, n_feature, fraction_of_feature, performance_metrics)
                else:
                    valid_folds = dev_lstm_fss(feature_type, n_feature, int(fraction_of_feature*n_feature))
                    performance_metrics = test_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature), valid_folds)
                    post_fss_results(feature_type, n_feature, fraction_of_feature, performance_metrics)


if __name__ == "__main__":
    main()