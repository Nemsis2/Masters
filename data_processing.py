import pickle
from sklearn.utils import resample
import os
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf

"""
date: 15/02/2023 

author: Michael Knight

desc: Contains all functions pertaining to data processing

functions:

get_smote_balanced_dataset: takes in data and returns smote balanced data

split_data: takes in data and splits into training and evaluation sets with class distribution preserved

extract_data_info: applies smote to the data, splits into test and training and returns all
"""



def get_SMOTE_balanced_dataset(x_values, y_values):
    """
    #############################################################################
    #                This is a copy of Madhu's implementation.                  #
    #    Comments have been added and some lines have been removed or altered   #
    ##############################################################################
    
    inputs: x_values - Feature data
            y_values - Labels corresponding to the feature data

    output: none

    returns: smote balanced feature data and corresponding labels

    desc: reshapes the input data
          generates smote dataset and resamples
          reshapes new data to input shape and then returns

    """

    # get the dimensions of the feature data
    dim_1 = np.array(x_values).shape[0]
    dim_2 = np.array(x_values).shape[1]
    dim_3 = np.array(x_values).shape[2]
    
    # calculate the new secondary dimension
    new_dim = dim_1 * dim_2
    
    # reshape the feature data to be 2 dimensional
    new_x_values = np.array(x_values).reshape(new_dim, dim_3)
    
    # reshape the labels to match the new shape of the feature data
    new_y_values = []
    for i in range(len(y_values)):
        new_y_values.extend([y_values[i]]*dim_2)
    
    # convert from type list to type array
    new_y_values = np.array(new_y_values)
    
    # apply smote to the data set
    oversample = SMOTE()
    X_Values, Y_Values = oversample.fit_resample(new_x_values, new_y_values)
    
    # revert to original feature data shape
    x_values_SMOTE = X_Values.reshape(int(X_Values.shape[0]/dim_2), dim_2, dim_3)
    
    # revert to original label shape 
    y_values_SMOTE = []
    for i in range(int(X_Values.shape[0]/dim_2)):
        value_list = list(Y_Values.reshape(int(X_Values.shape[0]/dim_2), dim_2)[i])
        y_values_SMOTE.extend(list(set(value_list)))
        if len(set(value_list)) != 1:
            print('\n\n********* STOP: THERE IS SOMETHING WRONG IN TRAIN ******\n\n')
        
    return x_values_SMOTE, y_values_SMOTE



def split_data(file_name, k_folds):
    """
    inputs: file_name - name of the file to be processed (not the full path only the file name)
            k_folds - the number of folds to be created

    output: k_folds of input data

    returns: none

    desc: splits the data into k_folds where each fold preserves approximately the class balance
    """
    # create a path to the raw data
    raw_path = '../data/raw/' + str(file_name)
    
    # load the data from the input data path
    data = pickle.load(open(raw_path, 'rb'))
    
    for i in range(k_folds):
        # create the path to the new k_fold data
        k_folds_path = '../data/k_folds/' + str(file_name) + str("_k_fold_") + str((i+1))

        # if data has not already been processed
        if os.path.exists(k_folds_path) == False:

            # get data from dict and turn into two lists
            data_list_0 = []
            data_list_1 = []
            for key, values in dict(data).items():
                if len(values) != 0: # if list is not empty
                    if values[0][1] == 0:
                        data_list_0.append(key)
                    else:
                        data_list_1.append(key)

            # on the first iteration get the length of the full dataset
            if i == 0:
                n_samples_0 = int((1/k_folds)*len(data_list_0))
                n_samples_1 = int((1/k_folds)*len(data_list_1))

            # if we are not on the final fold
            if i != (k_folds-1):
                # sample from the data without replacement
                eval_0 = resample(data_list_0, replace=False, n_samples=n_samples_0)
                eval_1 = resample(data_list_1, replace=False, n_samples=n_samples_1)

                k_fold_data = {}
            
                # remove random samples from the main data and add random samples to the current k_fold
                for i in range(len(eval_0)):
                    k_fold_data[eval_0[i]] = data[eval_0[i]]
                    del data[eval_0[i]]

                for i in range(len(eval_1)):
                    k_fold_data[eval_1[i]] = data[eval_1[i]]
                    del data[eval_1[i]]

                # save the current k_fold
                with open(k_folds_path, 'wb') as fp:
                    pickle.dump(k_fold_data, fp)

                print("Fold:", str(k_folds_path), "saved going to next")
            
            # if we are on the final fold
            else:
                # save the final data as the final k_fold
                with open(k_folds_path, 'wb') as fp:
                    pickle.dump(data, fp)

                print('Fold:', str(k_folds_path),"saved going to next")

        else:
            print("Data already processed. If all folds were not completed delete previous and start again.")


def extract_data_info(data):
    """
    inputs: loaded feature data

    outputs: none

    returns: input_train_data - training data for the model to be fitted on
             input_train_labels - labels corresponding to the training data
             input_test_data - test data for the model to be tested on
             input_test_labels - labels corresponding to the test data
             n_row - # rows of the input data
             n_column - # columns of the input data


    desc: apply smote to the data and split into training and test set.
          reshape the data to fit into the resnet50 model and return all 
    """

    # get data from dict and turn into a list
    data_list = []
    for values in data.values():
        if len(values) != 0: # if list is not empty
            data_list.append(values[0])

    # split data into train and test with a 80/20 split and random seed 25
    train_data, test_data = train_test_split(data_list, test_size=0.2, random_state=25)

    # convert to np.array
    train_data = np.array(train_data, dtype=object)
    test_data = np.array(test_data, dtype=object)
    
    # split data into feature data and labels for both train and test set
    input_train_data = train_data[:,0]
    input_train_labels = train_data[:,1]
    input_test_data = test_data[:,0]
    input_test_labels = test_data[:,1]

    # get size of input data for reshaping   
    n_row = input_train_data[0].shape[0] # this is dependent on the 'seg' of the mel spectrogram
    n_column = input_train_data[0].shape[1] # this is dependent on the number of mel splits(N_mel) normally +3 for some reason

    # reshape the input data to fit into the resnet50 model
    input_train_data = np.array(input_train_data)
    input_test_data = np.array(input_test_data)
    input_train_data = np.array([x.reshape( (n_row, n_column) ) for x in input_train_data] )
    input_test_data = np.array([x.reshape( (n_row, n_column) ) for x in input_test_data] )

    # smote balance the split data
    input_train_data, input_train_labels = get_SMOTE_balanced_dataset(input_train_data, input_train_labels)
    input_test_data, input_test_labels = get_SMOTE_balanced_dataset(input_test_data, input_test_labels)

    # convert labels from type list to type array
    input_train_labels = np.array(input_train_labels)
    input_test_labels = np.array(input_test_labels)

    # expand labels to be one hot encoded as opposed to binary classified
    input_train_labels = tf.keras.utils.to_categorical(input_train_labels, 2)
    input_test_labels = tf.keras.utils.to_categorical(input_test_labels, 2)

    return input_train_data, input_train_labels, input_test_data, input_test_labels, n_row, n_column
