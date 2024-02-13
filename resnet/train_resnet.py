# libraries
import torch as th
import gc
import os

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 64
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
EPOCHS = 16


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


def create_inner_resnet(feature_type, n_feature, model, model_type='dev'):
    """
    Description:
    ---------
    Trains models for the outer and inner folds

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.

    model_type: (string) type of model. Specifies data to be trained on as well as which folder the modesl will be saved too.
    
    """
    model_path = f'../../models/tb/{model.name}/{feature_type}/{n_feature}_{feature_type}/{model_type}/'
    
    if len(os.listdir(model_path)) == 0: # if the folder is empty
        print(f'Creating {model_type} models for {n_feature}_{feature_type}')

        for outer in range(NUM_OUTER_FOLDS):
            print("Outer fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("Inner fold=", inner)

                k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
                data, labels = extract_inner_fold_data(k_fold_path, inner)
                if feature_type == 'mfcc':
                    data = normalize_mfcc(data)
                data, labels, lengths = create_batches(data, labels, 'image', BATCH_SIZE)

                # run through all the epochs
                for epoch in range(EPOCHS):
                    print("epoch=", epoch)
                    train(data, labels, lengths, model)

                # collect the garbage
                del data, labels, lengths
                gc.collect()

                save_model(model, feature_type, n_feature, model_type, outer, inner)
    
    else:
        print(f'Models already exist for type:{model_type}_{n_feature}_{feature_type}. Skipping...')


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            for model in [Resnet18(), Resnet10(), Resnet6_2Deep(), Resnet6_4Deep]:
                create_inner_resnet(feature_type, n_feature, model, 'dev')
            

if __name__ == "__main__":
    main()