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

# test on dev set

# save result in dev fss auc file

def read_in_features(feature_path):
    
    file_name = f'{feature_path}features_outer_{outer}_inner_{inner}.txt'
    features = []
    with open(file_name, 'r') as f:
        for line in f:
            features.append(int(line.split('\n')[0]))
    f.close()
    return features


def extract_features_from_data(data, selected_features):
    for batch in range(len(data)):
        chosen_features = []
        for feature in selected_features:
            chosen_features.append(np.asarray(data[batch][:,:,feature]))
        data[batch] = th.as_tensor(np.stack(chosen_features, -1))

    return data


for feature_type in ['mfcc', 'melspec', 'lfb']:
    if feature_type == 'mfcc':
        n_feature = 13
    if feature_type == 'melspec' or feature_type == 'lfb':
        n_feature = 80
    
    for outer in range(3):
        dev_auc_list = []

        for inner in range(4):
            print("Inner fold=", inner)
            
            # grab all chosen features for this inner model
            fss_features = read_in_features(f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/')

            # grab the inner training data for this model
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            train_data, train_labels, train_names = extract_inner_fold_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                train_data = normalize_mfcc(train_data)
            train_data, train_labels, train_lengths, train_names = create_batches(train_data, train_labels, train_names, 64)

            # grab the dev data for this model
            dev_data, dev_labels, dev_names = extract_dev_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                dev_data = normalize_mfcc(dev_data)
            dev_data, dev_labels, dev_lengths, dev_names = create_batches(dev_data, dev_labels, dev_names, 64)

            # pass through the total number of features increasing the number to be trained on
            total_features = np.arange(1,n_feature)
            if feature_type == 'mfcc':
                total_features = np.arange(1,n_feature*3)
            
            # for num_selected_features in total_features:
            #     print(f'Currently selecing: {num_selected_features} features')
                
            # select the features which will be trained on this time
            selected_features = fss_features
            print(f'selected features: {selected_features}')
            
            # select the chosen features from the training and dev data
            print(train_data[0].shape)
            current_train_data = extract_features_from_data(train_data.copy(), selected_features)
            current_dev_data = extract_features_from_data(dev_data.copy(), selected_features)

            for model in [Resnet18()]:
                print(f'Creating {model.name} models for {n_feature}_{feature_type}')
                # run through all the epochs
                for epoch in tqdm(range(EPOCHS)):
                    train(current_train_data, train_labels, train_lengths, model)
            
            # test model on its dev data and save auc
            results = test(dev_data, model.model, dev_lengths) # do a forward pass through the models
            results = np.vstack(results)
            dev_labels_ = np.vstack(dev_labels)
            results, dev_labels_ = gather_results(results, dev_labels_, dev_names)
            dev_auc_list.append(roc_auc_score(dev_labels_, results))


            print(f'{feature_type}, {n_feature} outer: {outer} inner: {inner} auc: {dev_auc_list}')

            # # save current dev auc list
            # file_name = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/auc_outer_{outer}_inner_{inner}.txt'
            # with open(file_name, 'w') as f:
            #     for auc in dev_auc_list:
            #         f.write("%s\n" % auc)


    

