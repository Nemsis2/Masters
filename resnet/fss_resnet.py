# libraries
import torch as th
import numpy as np
import gc

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *

NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
BATCH_SIZE = 64
EPOCHS = 16

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


def generate_feature_list_resnet_18(feature_type, n_features):
    """
    Description:
    ---------

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.
    """
    if feature_type == 'mfcc':
        total_features = n_features*3
    else: total_features = n_features

    for outer in range(NUM_OUTER_FOLDS):
        if outer == 2:
            print("train_outer_fold=", outer)
            for inner in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", inner)
                selected_feature_list, auc_list = [], []
                features = np.arange(0,total_features)

                # load in data
                k_fold_path = f'../../data/tb/combo/new/{n_features}_{feature_type}_fold_{outer}.pkl'
                train_data, train_labels, train_names = load_train_data(k_fold_path, inner, feature_type)
                dev_data, dev_labels, dev_names = load_dev_data(k_fold_path, inner, feature_type)
                train_data, train_labels, train_lengths, train_names = create_batches(train_data, train_labels, train_names, BATCH_SIZE)
                dev_data, dev_labels, dev_lengths, dev_names = create_batches(dev_data, dev_labels, dev_names, BATCH_SIZE)

                feature_selection_train_data = np.concatenate(train_data, 0)
                feature_selection_dev_data = np.concatenate(dev_data, 0)
                # iterate feature-1 times
                while len(selected_feature_list) != total_features:
                    auc_list = []
                    # Pass through all unselected features
                    for feature in features:
                        model = Resnet18()
                        current_features, current_features_dev = select_features(feature_selection_train_data, feature_selection_dev_data, selected_feature_list, feature)
                        current_features = create_data_batches(current_features, BATCH_SIZE)
                        current_features_dev = create_data_batches(current_features_dev, BATCH_SIZE)

                        # train the model
                        for epoch in range(EPOCHS):
                            print("epoch=", epoch)
                            train(current_features, train_labels, train_lengths, model)
                            
                        # get dev results
                        results = test(current_features_dev, model.model, dev_lengths)
                        results = np.vstack(results)
                        dev_labels = np.vstack(dev_labels)
                        results, new_labels = gather_results(results, dev_labels, dev_names)
                        auc = roc_auc_score(new_labels, results)
                        auc_list.append(auc)
                        print("new feature:", feature, "AUC:", auc)

                        # force delete loaded in model
                        del model
                        gc.collect()

                    # select best performing feature from list
                    best_feature = np.argmax(np.array(auc_list))
                    print("Features array:", features)
                    print("Best feature:", best_feature)
                    print("Array selection", features[best_feature])
                    
                    selected_feature_list.append(str(features[best_feature]))
                    auc_list.append(str(auc_list[best_feature]))
                    print("Best performing feature:", best_feature, "with an auc of:", auc_list[best_feature])

                    # delete the chosen feature so it cannot be reselected
                    features = np.delete(features, best_feature)
                
                    #save current feature list
                    file_name = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_features}_{feature_type}/fss/docs/features_outer_{outer}_inner_{inner}.txt'
                    with open(file_name, 'w') as f:
                        for feature in selected_feature_list:
                            f.write("%s\n" % feature)

                    # save current auc list
                    file_name = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_features}_{feature_type}/fss/docs/auc_outer_{outer}_inner_{inner}.txt'
                    with open(file_name, 'w') as f:
                        for auc in auc_list:
                            f.write("%s\n" % auc)


def main():
    for feature_type in ['lfb']:
        if feature_type == 'mfcc':
            features = [13]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80] 
        
        for n_feature in features:
            generate_feature_list_resnet_18(feature_type, n_feature)
        

if __name__ == "__main__":
    main()