# libraries
import torch as th
import os
import pickle
from imblearn.over_sampling import SMOTE
import gc

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

def extract_features_from_data(data, selected_features):
    # select only the relevant features
    chosen_features = []
    for i in range(len(selected_features)):
        chosen_features.append(np.asarray(data[:,selected_features[i]]))
    chosen_features = th.as_tensor(np.stack(chosen_features, -1))

    return chosen_features



for feature_type in ['melspec']:
    if feature_type == 'mfcc':
        n_feature = 13
    if feature_type == 'melspec' or feature_type == 'lfb':
        n_feature = 80

    for outer in range(3):
        test_auc_list = []

        # grab the test data for this model
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        test_data, test_labels, test_names = load_test_data(k_fold_path, feature_type)

        # grab all chosen features for this inner model
        fss_features = outer_fss(outer, n_feature, n_feature, f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/')

        # pass through the total number of features increasing the number to be trained on
        total_features = np.arange(1,n_feature)
        if feature_type == 'mfcc':
            total_features = np.arange(1,n_feature*3)

        # for num_selected_features in total_features:
        #     print(f'Currently selecing: {num_selected_features} features')
        test_models = []

        for inner in range(4):
            print("Inner fold=", inner)
            selected_features = fss_features
            print(f'selected features: {selected_features}')

            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            train_data, train_labels = load_inner_data(k_fold_path, feature_type, inner)

            current_train_data = extract_features_from_data(train_data.copy(), selected_features)

            model, params = grid_search_lr(current_train_data, train_labels)
            test_models.append(model)

        current_test_data = extract_features_from_data(test_data.copy(), selected_features)
        results = []
        for model in test_models:
            results.append(model.predict_proba(current_test_data)) # do a forward pass through the models

        del train_data, train_labels
        gc.collect()
        
        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], test_labels, test_names)
            output.append(new_results)

        results = sum(output)/len(test_models)
        test_auc_list.append(roc_auc_score(new_labels, results))

        del test_models
        gc.collect()

        print(f'{feature_type}, {n_feature} auc: {test_auc_list}')

            # # save current test auc list
            # file_name = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/test_auc_outer_{outer}.txt'
            # with open(file_name, 'w') as f:
            #     for auc in test_auc_list:
            #         f.write("%s\n" % auc)