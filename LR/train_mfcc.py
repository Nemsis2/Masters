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

def normalize(mfcc):
    mfcc = (mfcc-np.max(mfcc))/(np.max(mfcc)-np.min(mfcc))
    return mfcc



def create_inner_lr(feature_type, n_feature, model_type='dev'):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/{model_type}/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating {model_type} models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
                print("Outer fold=", outer)
                
                for inner in range(NUM_INNER_FOLDS):
                    print("Inner fold=", inner)

                    k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
                    print(k_fold_path)
                    if model_type == 'dev':
                        data, labels = extract_inner_fold_data(k_fold_path, inner)
                    elif model_type =='em':
                        data, labels = extract_outer_fold_data(k_fold_path)

                    for i in range(data.shape[0]):
                       for j in range(data[i].shape[0]):
                            if np.all(data[i][j]) != 0:
                                data[i][j] = normalize(data[i][j])



                    # for averaging
                    data = np.array([np.mean(x, axis=0) for x in data])
                    labels = labels.astype("int")

                    model, params = grid_search_lr(data, labels)
                    pickle.dump(model, open(f'{model_path}lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model
    
    else:
        print(f'Models already exist for type:{model_type}_{n_feature}_{feature_type}. Skipping...')



def test_lr(feature_type, n_feature, threshold, model_type='em'):
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

        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/{model_type}/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model
        

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)

        for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                if np.all(data[i][j]) != 0:
                    data[i][j] = normalize(data[i][j])

        
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
        if feature_type == 'mfcc':
            print(results)
        inner_auc = roc_auc_score(new_labels, results)
        results = (np.array(results)>threshold[outer]).astype(np.int8)
        inner_sens, inner_spec = calculate_sens_spec(new_labels, results)
        
        # add to the total auc, sens and spec
        auc += inner_auc
        sens += inner_sens
        spec += inner_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS


def main():
    for feature_type in ['mfcc']:    
        for n_feature in  [13, 26, 39]:
            #create models for development (threshold calculation)
            create_inner_lr(feature_type, n_feature,'dev')
            
            # create models for ensemble testing
            create_inner_lr(feature_type, n_feature,'em')

            # get the optimal threshold based off the EER
            threshold = get_decision_threshold(feature_type, n_feature, NUM_OUTER_FOLDS, NUM_INNER_FOLDS)

            # test the em setup
            auc, sens, spec = test_lr(feature_type, n_feature, threshold, 'em')

            print(f'AUC for {n_feature}_{feature_type}: {auc}')
            print(f'Sens for {n_feature}_{feature_type}: {sens}')
            print(f'Spec for {n_feature}_{feature_type}: {spec}')


if __name__ == "__main__":
    main()