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

# set hyperpaperameters
BATCH_SIZE = 64
EPOCHS = 16
#geoff
HIDDEN_DIM = [32, 64, 32]
LSTM_LAYERS = [2, 1, 1]

#own
#HIDDEN_DIM = [128, 64, 32]
#LSTM_LAYERS = [3, 2, 2]

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)


def create_dev_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    model_path = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating dev models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
                print("Outer fold=", outer)
                
                for inner in range(NUM_INNER_FOLDS):
                    print("Inner fold=", inner)

                    if feature_type == 'melspec' or feature_type == 'lfb':
                        model = bi_lstm_package(n_feature, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)
                    elif feature_type == 'mfcc':
                        model = bi_lstm_package(n_feature*3, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)
                    
                    model.train()
                    model.save()
    
    else:
        print(f'Models already exist for type:dev_{n_feature}_{feature_type}. Skipping...')


def create_ts_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    model_path = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/ts/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating ts models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
            print("Outer fold=", outer)
            
            if feature_type == 'melspec' or feature_type == 'lfb':
                model = bi_lstm_package(n_feature, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts', n_feature, feature_type)
            elif feature_type == 'mfcc':
                model = bi_lstm_package(n_feature*3, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts', n_feature, feature_type)

            models = []
            for inner in range(NUM_INNER_FOLDS):
                  models.append(load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'))
            
            model.train_ts(models)
            model.save()
    
    else:
        print(f'Models already exist for type:ts_{n_feature}_{feature_type}. Skipping...')


def create_ts_lstm_2(feature_type, n_feature):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    model_path = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/ts_2/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating ts_2 models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
            print("Outer fold=", outer)
            
            if feature_type == 'melspec' or feature_type == 'lfb':
                model = bi_lstm_package(n_feature, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts_2', n_feature, feature_type)
            elif feature_type == 'mfcc':
                model = bi_lstm_package(n_feature*3, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts_2', n_feature, feature_type)

            models = []
            for inner in range(NUM_INNER_FOLDS):
                  models.append(load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'))

            model.train_ts_2(models)
            model.save()
    
    else:
        print(f'Models already exist for type:ts_2_{n_feature}_{feature_type}. Skipping...')



def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            #create models for development (threshold calculation)
            create_dev_lstm(feature_type, n_feature)

            create_ts_lstm(feature_type, n_feature)

            create_ts_lstm_2(feature_type, n_feature)
    
if __name__ == "__main__":
    main()