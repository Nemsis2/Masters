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
    Trains models for the outer and inner folds. Skips a folder if it contains anything.

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.

    model_type: (string) type of model. Specifies data to be trained on as well as which folder the modesl will be saved too.
    
    """
    model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/{model_type}/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating {model_type} models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
            print("Outer fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("Inner fold=", inner)

                k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
                data, labels = load_inner_data(k_fold_path, feature_type, inner)

                model, params = grid_search_lr(data, labels)
                pickle.dump(model, open(f'{model_path}lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model
    
    else:
        print(f'Models already exist for type:{model_type}_{n_feature}_{feature_type}. Skipping...')


def create_fss_lr(feature_type, n_feature, fss_feature):
    """
    Description:
    ---------

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.

    model_type: (string) type of model. Specifies data to be trained on as well as which folder the modesl will be saved too.
    """
    # set the model path
    model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/'
    
    # select only the relevant features
    feature_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
    if feature_type == 'mfcc':
        selected_features = dataset_fss(n_feature*3, fss_feature, feature_path)
    else:
        selected_features = dataset_fss(n_feature, fss_feature, feature_path)

    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        for inner in range(NUM_INNER_FOLDS):
            print("Inner fold=", inner)

            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels = load_inner_data(k_fold_path, feature_type, inner)

            chosen_features = []
            print('selected_features:', selected_features)
            for i in range(len(selected_features)):
                chosen_features.append(np.asarray(data[:,selected_features[i]]))
            chosen_features = th.as_tensor(np.stack(chosen_features, -1))

            model, params = grid_search_lr(chosen_features, labels)
            pickle.dump(model, open(f'{model_path}lr_{feature_type}_{n_feature}_fss_{fss_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model



def create_inner_per_frame_lr(feature_type, n_feature):
    """
    Description:
    ---------
    Trains models for the outer and inner folds. Skips a folder if it contains anything.

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.

    model_type: (string) type of model. Specifies data to be trained on as well as which folder the modesl will be saved too.
    
    """
    model_path = f'../../models/tb/lr_per_frame/{feature_type}/{n_feature}_{feature_type}/dev/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating per_frame dev models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
            print("Outer fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("Inner fold=", inner)

                k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
                data, labels = load_inner_per_frame_data(k_fold_path, feature_type, inner)

                model, params = grid_search_lr(data, labels)
                pickle.dump(model, open(f'{model_path}lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model

    else:
        print(f'models already exist for {feature_type}_{n_feature} skipping to next...')


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 

        
        for n_feature in features:
            #create_inner_lr(feature_type, n_feature,'dev')
            create_inner_per_frame_lr(feature_type, n_feature)
            
            """
            for fraction_of_feature in [0.1, 0.2, 0.5]:
                
                if feature_type == 'mfcc':
                    create_fss_lr(feature_type, n_feature, int(fraction_of_feature*n_feature*3))
                    create_inner_per_frame_lr(feature_type, n_feature)
                else:
                    create_fss_lr(feature_type, n_feature, int(fraction_of_feature*n_feature))
            """
            

if __name__ == "__main__":
    main()