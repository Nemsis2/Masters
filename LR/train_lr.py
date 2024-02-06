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


def create_inner_lr(feature_type, n_feature, model_type='dev'):
    """
    Description:
    ---------
    
    Inputs:
    ---------
    feature_type: 

    n_feature:

    model_type: 
    
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
                    data, labels = extract_inner_fold_data(k_fold_path, inner)

                    # for by frame
                    #labels = labels_per_frame(data, labels)
                    #data = np.vstack(data)

                    if feature_type=="mfcc":
                        data = normalize_mfcc(data)

                    # for averaging
                    data = np.array([np.mean(x, axis=0) for x in data])
                    labels = labels.astype("int")

                    model, params = grid_search_lr(data, labels)
                    pickle.dump(model, open(f'{model_path}lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model
    
    else:
        print(f'Models already exist for type:{model_type}_{n_feature}_{feature_type}. Skipping...')


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            create_inner_lr(feature_type, n_feature,'dev')
            


if __name__ == "__main__":
    main()