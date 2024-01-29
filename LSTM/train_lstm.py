# libraries
import torch as th
import torch.nn as nn
import torch.optim as optim
import os
import pickle

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
HIDDEN_DIM = [128, 64, 32]
LSTM_LAYERS = [3, 2, 2]

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)


def create_inner_lstm(feature_type, n_feature, model_type='dev'):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    model_path = f'../../models/tb/bi_lstm/{feature_type}/{n_feature}_{feature_type}/{model_type}/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating {model_type} models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
                print("Outer fold=", outer)
                
                for inner in range(NUM_INNER_FOLDS):
                    print("Inner fold=", inner)

                    model = bi_lstm_package(n_feature, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, inner, EPOCHS, BATCH_SIZE, model_type, n_feature, feature_type)
    
    else:
        print(f'Models already exist for type:{model_type}_{n_feature}_{feature_type}. Skipping...')


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            #create models for development (threshold calculation)
            create_inner_lstm(feature_type, n_feature,'dev')
            
            # create models for ensemble testing
            create_inner_lstm(feature_type, n_feature,'em')
