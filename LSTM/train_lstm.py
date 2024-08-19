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


def create_dev_lstm(feature_type, n_feature, hidden_dim, num_layers):
    """
    Description:
    ---------
    
    Inputs:
    ---------

    Outputs:
    --------

    """
    model_path = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/'
    
    print(f'Creating dev models for {n_feature}_{feature_type}')

    for outer in range(NUM_OUTER_FOLDS):
            print("Outer fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("Inner fold=", inner)

                if feature_type == 'melspec' or feature_type == 'lfb':
                    model = bi_lstm_package(n_feature, hidden_dim[outer], num_layers[outer], outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)
                elif feature_type == 'mfcc':
                    model = bi_lstm_package(n_feature*3, hidden_dim[outer], num_layers[outer], outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)
                
                model.train()
                model.save()


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
        
        if sum(valid_folds[outer]) == 0:
            best_auc = 0
            for inner in range(NUM_INNER_FOLDS):
                # get the dev model
                model = load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}')
                auc = model.dev() # do a forward pass through the models
                if auc > best_auc:
                    best_auc = auc
                    valid_inner_fold = inner
        
            count += 1
            total_auc += best_auc
            valid_folds[outer][valid_inner_fold] = 1
    
    print(total_auc/count)
    print(valid_folds)
    return valid_folds
    

def create_exclusive_ts_lstm(feature_type, n_feature, hidden_dim, num_layers):
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
    
    print(f'Creating ts models for {n_feature}_{feature_type}')

    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        if feature_type == 'melspec' or feature_type == 'lfb':
            model = bi_lstm_package(n_feature, hidden_dim[outer], num_layers[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts', n_feature, feature_type)
        elif feature_type == 'mfcc':
            model = bi_lstm_package(n_feature*3, hidden_dim[outer], num_layers[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts', n_feature, feature_type)

        models = []
        for inner in range(NUM_INNER_FOLDS):
            models.append(load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'))
        
        model.train_ts(models)
        model.save()
    


def create_ts_lstm(feature_type, n_feature, hidden_dim, num_layers):
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
    print(f'Creating ts_2 models for {n_feature}_{feature_type}')
    valid_folds = dev_lstm(feature_type, n_feature)
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        if feature_type == 'melspec' or feature_type == 'lfb':
            model = bi_lstm_package(n_feature, hidden_dim[outer], num_layers[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts_2', n_feature, feature_type)
        elif feature_type == 'mfcc':
            model = bi_lstm_package(n_feature*3, hidden_dim[outer], num_layers[outer], outer, None, EPOCHS, BATCH_SIZE, 'ts_2', n_feature, feature_type)

        models = []
        for inner in range(NUM_INNER_FOLDS):
            if valid_folds[outer][inner] == 1:
                models.append(load_model(f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'))

        model.train_ts(models)
        model.save()



def create_fss_lstm(feature_type, n_feature, fss_feature, hidden_dim, num_layers):
    # set the model path
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        for inner in range(NUM_INNER_FOLDS):
            print("Inner fold=", inner)

            if feature_type == 'melspec' or feature_type == 'lfb':
                model = bi_lstm_package(fss_feature, hidden_dim[outer], num_layers[outer], outer, inner, EPOCHS, BATCH_SIZE, 'fss', n_feature, feature_type)
            elif feature_type == 'mfcc':
                model = bi_lstm_package(fss_feature, hidden_dim[outer], num_layers[outer], outer, inner, EPOCHS, BATCH_SIZE, 'fss', n_feature, feature_type)

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

    hyperparameters = {'mfcc':{13: {'hidden_dim': [32, 32, 64], 'num_layers': [2, 2, 3]},
                            26: {'hidden_dim': [128, 32, 64], 'num_layers': [3, 3, 2]},
                            39: {'hidden_dim': [64, 64, 32], 'num_layers': [2, 1, 3]}},
                        'melspec': {80: {'hidden_dim': [32, 64, 128], 'num_layers': [2, 2, 3]},
                                    128: {'hidden_dim': [32, 64, 32], 'num_layers': [1, 1, 1]},
                                    180: {'hidden_dim': [128, 64, 128], 'num_layers': [3, 1, 3]}},
                        'lfb':{80: {'hidden_dim': [128, 128, 128], 'num_layers': [2, 2, 3]},
                                128: {'hidden_dim': [32, 128, 64], 'num_layers': [3, 1, 1]},
                                180: {'hidden_dim': [128, 128, 128], 'num_layers': [2, 2, 2]}}}
    
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            # create_dev_lstm(feature_type, n_feature, hyperparameters[feature_type][n_feature]['hidden_dim'], hyperparameters[feature_type][n_feature]['num_layers'])

            create_ts_lstm(feature_type, n_feature, hyperparameters[feature_type][n_feature]['hidden_dim'], hyperparameters[feature_type][n_feature]['num_layers'])

            # for fraction_of_feature in [0.1, 0.2, 0.5]:
            #     if feature_type == 'mfcc':
            #         create_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature*3), hyperparameters[feature_type][n_feature]['hidden_dim'], hyperparameters[feature_type][n_feature]['num_layers'])
            #     else:
            #         create_fss_lstm(feature_type, n_feature, int(fraction_of_feature*n_feature), hyperparameters[feature_type][n_feature]['hidden_dim'], hyperparameters[feature_type][n_feature]['num_layers'])
    
if __name__ == "__main__":
    main()