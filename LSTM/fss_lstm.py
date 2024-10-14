# libraries
import torch as th
import gc

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from get_best_features import *
from lstm_model_scripts import *

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

def do_fss_lstm(feature_type, n_feature, hidden_dim, num_layers):
    """
    Description:
    ---------

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.
    """
    if feature_type == 'mfcc':
        total_features = n_feature*3
    else: total_features = n_feature
    for outer in range(NUM_OUTER_FOLDS):
        print("train_outer_fold=", outer)
        for inner in range(NUM_INNER_FOLDS):
            print("train_inner_fold=", inner)
            feature_priority, auc_priority = [], []
            features = np.arange(0,total_features)
            
            # load in training data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            train_data, train_labels = load_inner_data(k_fold_path, feature_type, inner)
            train_data, train_labels, train_lengths = create_batches(train_data, train_labels, "linear", BATCH_SIZE)
            
            # load in dev data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            dev_data, dev_labels, dev_names = load_dev_data(k_fold_path, feature_type, inner)
            dev_data, dev_labels, dev_lengths = create_batches(dev_data, dev_labels, "linear", BATCH_SIZE)

            while len(feature_priority) != total_features:
                performance = []

                # append all previously selected features
                if len(feature_priority) != 0:
                    chosen_features, chosen_features_dev = select_features(train_data, dev_data, feature_priority)
                else: chosen_features, chosen_features_dev = [], []

                # Pass through all unselected features
                for feature in features:
                    print(f'total features: {len(feature_priority)+1}')
                    print(f'Current feature eval: {feature}')
                    # append newest feature
                    copy_of_chosen_features = chosen_features.copy()
                    copy_of_chosen_features_dev = chosen_features_dev.copy()
                    latest_features, latest_features_dev = add_latest_feature(train_data, dev_data, copy_of_chosen_features, copy_of_chosen_features_dev, feature)

                    # create new model
                    model = bi_lstm_package(len(feature_priority)+1, hidden_dim[outer], num_layers[outer], outer, inner, EPOCHS, BATCH_SIZE, 'dev', n_feature, feature_type)

                    # train the model
                    model.train_on_select_features(latest_features, train_labels, train_lengths)

                    # assess performance
                    auc = model.dev_fss(latest_features_dev, dev_labels, dev_names, dev_lengths)
                    performance.append(auc)
                    print(f'auc:{auc}')

                    # force delete loaded in model
                    del model, copy_of_chosen_features, copy_of_chosen_features_dev
                    gc.collect()

                # select best performing feature from list
                best_feature = np.argmax(np.array(performance))
                print("Features array:", features)
                print("Best feature:", best_feature)
                print("Array selection", features[best_feature])
                
                feature_priority.append(str(features[best_feature]))
                auc_priority.append(str(performance[best_feature]))
                print("Best performing feature:", best_feature, "with an auc of:", performance[best_feature])

                # delete the chosen feature so it cannot be reselected
                features = np.delete(features, best_feature)
                
                #save current feature list
                file_name = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/fss/docs/features_outer_{outer}_inner_{inner}.txt'
                with open(file_name, 'w') as f:
                    for feature in feature_priority:
                        f.write("%s\n" % feature)

                # save current auc list
                file_name = f'../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/fss/docs/auc_outer_{outer}_inner_{inner}.txt'
                with open(file_name, 'w') as f:
                    for auc in auc_priority:
                        f.write("%s\n" % auc)


def main():
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
            features = [39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80] 
        
        for n_feature in features:
            do_fss_lstm(feature_type, n_feature, hyperparameters[feature_type][n_feature]['hidden_dim'], hyperparameters[feature_type][n_feature]['num_layers'])
        

if __name__ == "__main__":
    main()