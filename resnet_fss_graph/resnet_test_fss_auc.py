# libraries
import torch as th
import gc
from tqdm import tqdm

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *

EPOCHS = 16

# for 80 lfb, 80 melspec, 13 mfcc

# read in chosen features for this inner model

# select x features

# train model

# test on test set

# save result in test fss auc file


def outer_fss(outer, total_features, num_features, feature_path):
    """
    Uses previously generated SFS results to determine the highest "scoring" features
    as selected by 5 different models across an outer fold.


    Parameters:
    -----------
        outer(int): the outer fold to be considered. can only be 1, 2 or 3

        num_features(int) : the number of top features to be selected. Maximum of 180

    Returns:
    --------
        selected_features(list) : list of selected features with length corresponding to the value
        of num_features e.g. if num_features = 3, selected_features = [28, 64, 32]
    """
    
    if outer > 3:
        print("Outer fold", outer, "does not exist")
        return


    fold_feature = np.zeros(total_features)
    selected_features = []

    for inner in range(4):
        best_features = []
        file_name = f'{feature_path}features_outer_{outer}_inner_{inner}.txt'
        with open(file_name, 'r') as f:
            for line in f:
                best_features.append(line.split('\n')[0])

        for i in range(len(best_features)):
            fold_feature[int(best_features[i])] += i

    sorted_list = sorted(fold_feature)

    #find top num_features features
    count = 0
    for i in range(num_features):
        while sorted_list[i] != fold_feature[count]:
            count += 1
        selected_features.append(count)
        fold_feature[count] = 99999
        count = 0
    
    return selected_features



def extract_features_from_data(data, selected_features):
    for batch in range(len(data)):
        chosen_features = []
        for feature in selected_features:
            chosen_features.append(np.asarray(data[batch][:,:,feature]))
        data[batch] = th.as_tensor(np.stack(chosen_features, -1))

    return data


for feature_type in ['melspec']:
    if feature_type == 'mfcc':
        n_feature = 13
    if feature_type == 'melspec' or feature_type == 'lfb':
        n_feature = 80

    for outer in range(3):
        # grab the test data for this model
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        test_data, test_labels, test_names = extract_test_data(k_fold_path)
        if feature_type == 'mfcc':
            test_data = normalize_mfcc(test_data)
        test_data, test_labels, test_lengths, test_names = create_batches(test_data, test_labels, test_names, 64)

        test_auc_list = []
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
            
            # grab the inner training data for this model
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            train_data, train_labels, train_names = extract_inner_fold_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                train_data = normalize_mfcc(train_data)
            train_data, train_labels, train_lengths, train_names = create_batches(train_data, train_labels, train_names, 64)
            
            # select the features which will be trained on this time
            selected_features = fss_features
            print(f'selected features: {selected_features}')
            
            # select the chosen features from the training and test data
            print(train_data[0].shape)
            current_train_data = extract_features_from_data(train_data.copy(), selected_features)

            # train the current model
            for model in [Resnet18()]:
                print(f'Creating {model.name} models for {n_feature}_{feature_type}')
                # run through all the epochs
                for epoch in tqdm(range(EPOCHS)):
                    train(current_train_data, train_labels, train_lengths, model)
            
            test_models.append(model)

        del train_data, train_labels, train_lengths, train_names
        gc.collect()
        
        current_test_data = extract_features_from_data(test_data.copy(), selected_features)
        # test model on its test data and save auc
        results = []
        for model in test_models:
            results.append(test(current_test_data, model.model, test_lengths)) # do a forward pass through the models

        for i in range(len(results)):
            results[i] = np.vstack(results[i])

        labels = np.vstack(test_labels)
        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, test_names)
            output.append(new_results)

        results = sum(output)/len(test_models)
        test_auc_list.append(roc_auc_score(new_labels, results))

        print(f'{feature_type}, {n_feature} auc: {test_auc_list}')
        del test_models
        gc.collect()

        # # save current test auc list
        # file_name = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/test_auc_outer_{outer}.txt'
        # with open(file_name, 'w') as f:
        #     for auc in test_auc_list:
        #         f.write("%s\n" % auc)