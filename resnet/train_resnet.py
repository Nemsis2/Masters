# libraries
import torch as th
import gc

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *
from get_best_features import *

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 128
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
EPOCHS = 16


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


def create_inner_resnet(feature_type, n_feature, model_type='dev'):
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
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        for inner in range(NUM_INNER_FOLDS):
            print("Inner fold=", inner)

            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_inner_fold_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

            for model in [Resnet18(), Resnet10(), Resnet6_2Deep(), Resnet6_4Deep()]:
                print(f'Creating {model.name}_{model_type} models for {n_feature}_{feature_type}')
                # run through all the epochs
                for epoch in range(EPOCHS):
                    print("epoch=", epoch)
                    train(data, labels, lengths, model)

                save_model(model, feature_type, n_feature, model_type, outer, inner)

            # collect the garbage
            del data, labels, lengths
            gc.collect()


def create_ts_resnet(feature_type, n_feature, model_type='ts'):
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)

        # create new models
        for model in [Resnet18(), Resnet10(), Resnet6_2Deep(), Resnet6_4Deep()]:
            print(f'Creating {model.name}_{model_type} models for {n_feature}_{feature_type}')
            criterion_kl = nn.KLDivLoss()
        
            models = []
            for inner in range(NUM_INNER_FOLDS):
                print("Inner fold=", inner)
                
                #grab all dev models
                model_path = f'../../models/tb/resnet/{model.name}/{feature_type}/{n_feature}_{feature_type}/dev/{model.name}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
                models.append(pickle.load(open(model_path, 'rb'))) # load in the model
                
            # grab outer fold training data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_outer_fold_data(k_fold_path)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

            # run through all epochs
            for epoch in range(EPOCHS):
                print("epoch=", epoch)
                train_ts(data, labels, model, models, criterion_kl, lengths)

            save_model(model, feature_type, n_feature, model_type, outer, inner)

        # collect the garbage
        del data, labels, lengths
        gc.collect()


def create_fss_resnet(feature_type, n_feature, fss_feature, model_type='fss'):
    for outer in range(NUM_OUTER_FOLDS):
        print("Outer fold=", outer)
        
        for inner in range(NUM_INNER_FOLDS):
            print("Inner fold=", inner)

            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_inner_fold_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

            # select only the relevant features
            feature_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
            if feature_type == 'mfcc':
                selected_features = dataset_fss(n_feature*3, fss_feature, feature_path)
            else:
                selected_features = dataset_fss(n_feature, fss_feature, feature_path)

            for batch in range(len(data)):
                chosen_features = []
                for feature in selected_features:
                    chosen_features.append(np.asarray(data[batch][:,:,feature]))
                data[batch] = th.as_tensor(np.stack(chosen_features, -1))


            for model in [Resnet18(), Resnet10(), Resnet6_2Deep(), Resnet6_4Deep()]:
                print(f'Creating {model.name}_{model_type} models for {n_feature}_{feature_type}')
                # run through all the epochs
                for epoch in range(EPOCHS):
                    print("epoch=", epoch)
                    train(data, labels, lengths, model)

                model_path = f'../../models/tb/resnet/{model.name}/{feature_type}/{n_feature}_{feature_type}/{model_type}'
                pickle.dump(model, open(f'{model_path}/{model.name}_{feature_type}_{n_feature}_fss_{fss_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model



def main():
    for feature_type in ['lfb', 'mfcc', 'melspec']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            #create_inner_resnet(feature_type, n_feature, 'dev')
            #create_ts_resnet(feature_type, n_feature, 'ts')

            for fraction_of_feature in [0.1, 0.2, 0.5]:
                if feature_type == 'mfcc':
                    create_fss_resnet(feature_type, n_feature, int(fraction_of_feature*n_feature*3), 'fss')
                else:
                    create_fss_resnet(feature_type, n_feature, int(fraction_of_feature*n_feature), 'fss')
    
            

if __name__ == "__main__":
    main()