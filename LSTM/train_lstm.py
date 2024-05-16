# libraries
import torch as th
import os
from tqdm import tqdm

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

"""
for hidden_dim in [32, 64, 128]:
for num_layers in [1, 2, 3]:

hyperparameters:
[['mfcc', 13, 1, 1, 5],  = hidden_dim[32, 32, 64], num_layers[2, 2, 3]
['mfcc', 26, 8, 2, 4], = hidden_dim[128, 32, 64], num_layers[3, 3, 2]
['mfcc', 39, 4, 3, 2],  = hidden_dim[64, 64, 32], num_layers[2, 1, 3]
['melspec', 80, 1, 4, 8], = hidden_dim[32, 64, 128], num_layers[2, 2, 3]
['melspec', 128, 0, 3, 0], = hidden_dim[32, 64, 32], num_layers[1, 1, 1]
['melspec', 180, 8, 6, 8], = hidden_dim[128, 64, 128], num_layers[3, 1, 3]
['lfb', 80, 7, 7, 8], = hidden_dim[128, 128, 128], num_layers[2, 2, 3]
['lfb', 128, 2, 6, 3], = hidden_dim[32, 128, 64], num_layers[3, 1, 1]
['lfb', 180, 7, 7, 7]] = hidden_dim[128, 128, 128], num_layers[2, 2, 2]
"""

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)


# create hyperparameter dictionary


def hyperparameter_dev_check(feature_type, n_feature, hidden_dim, num_layers):
    outer_auc = []
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        auc = 0
        for inner in tqdm(range(NUM_INNER_FOLDS)):
            print("Inner fold=", inner)

            if feature_type == 'melspec' or feature_type == 'lfb':
                model = bi_lstm_package(n_feature, hidden_dim, num_layers, outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)
            elif feature_type == 'mfcc':
                model = bi_lstm_package(n_feature*3, hidden_dim, num_layers, outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)
            
            model.train()
            auc += model.dev()
        
        outer_auc.append(auc/NUM_INNER_FOLDS)

    return outer_auc


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


def create_exclusive_ts_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    Creates an lstm model for teacher student training where the student is taught only by the teacher
    and never makes use of the true labels.
    
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


def create_ts_lstm(feature_type, n_feature):
    """
    Description:
    ---------
    Creates an lstm model for teacher student training where the student is taught by the teacher
    and the true labels.
    
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

            model.train_ts(models)
            model.save()
    
    else:
        print(f'Models already exist for type:ts_2_{n_feature}_{feature_type}. Skipping...')


def create_fss_lstm(feature_type, n_feature, fss_feature):
    # set the model path
    model_path = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/fss/'
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        for inner in range(NUM_INNER_FOLDS):
            print("Inner fold=", inner)

            if feature_type == 'melspec' or feature_type == 'lfb':
                model = bi_lstm_package(fss_feature, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, inner, EPOCHS, BATCH_SIZE, 'fss', n_feature, feature_type)
            elif feature_type == 'mfcc':
                model = bi_lstm_package(fss_feature, HIDDEN_DIM[outer], LSTM_LAYERS[outer], outer, inner, EPOCHS, BATCH_SIZE, 'fss', n_feature, feature_type)

            model.train_fss(fss_feature)
            model.save(fss_feature)


def main():
    # grid search for lstm
    # this is a one time run to determine optimal hyperparameters for the given data
    # values should be interpreted manually and updated at the top of the script
    # hyperparameters = []
    # for feature_type in ['mfcc', 'melspec', 'lfb']:
    #     if feature_type == 'mfcc':
    #         features = [13, 26, 39]
    #     elif feature_type == 'melspec' or feature_type == 'lfb':
    #         features = [80, 128, 180] 

    #     for n_feature in features:
    #         auc = []
    #         hyperparameter = []
    #         for hidden_dim in [32, 64, 128]:
    #             for num_layers in [1, 2, 3]:
    #                 auc.append(hyperparameter_dev_check(feature_type, n_feature, hidden_dim, num_layers))

    #         auc = np.array(auc)
    #         hyperparameter = [feature_type, n_feature, np.argmax(auc[:,0]), np.argmax(auc[:,1]), np.argmax(auc[:,2])]
    #         print(hyperparameter)
    #         hyperparameters.append(hyperparameter)

    # print(hyperparameters)

    hyperparameters = {'mfcc':{'13': {'hidden_dim': [32, 32, 64], 'num_layers': [2, 2, 3]},
                            '26': {'hidden_dim': [128, 32, 64], 'num_layers': [3, 3, 2]}}
                            
                            }
    

#     hyperparameters:
# [['mfcc', 13, 1, 1, 5],  = hidden_dim[32, 32, 64], num_layers[2, 2, 3]
# ['mfcc', 26, 8, 2, 4], = hidden_dim[128, 32, 64], num_layers[3, 3, 2]
# ['mfcc', 39, 4, 3, 2],  = hidden_dim[64, 64, 32], num_layers[2, 1, 3]
# ['melspec', 80, 1, 4, 8], = hidden_dim[32, 64, 128], num_layers[2, 2, 3]
# ['melspec', 128, 0, 3, 0], = hidden_dim[32, 64, 32], num_layers[1, 1, 1]
# ['melspec', 180, 8, 6, 8], = hidden_dim[128, 64, 128], num_layers[3, 1, 3]
# ['lfb', 80, 7, 7, 8], = hidden_dim[128, 128, 128], num_layers[2, 2, 3]
# ['lfb', 128, 2, 6, 3], = hidden_dim[32, 128, 64], num_layers[3, 1, 1]
# ['lfb', 180, 7, 7, 7]] = hidden_dim[128, 128, 128], num_layers[2, 2, 2]
    
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            create_dev_lstm(feature_type, n_feature)

            create_ts_lstm(feature_type, n_feature)

            for fraction_of_feature in [0.1, 0.2, 0.5]:
                if feature_type == 'mfcc':
                    create_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature*3))
                else:
                    create_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature))
    
if __name__ == "__main__":
    main()