import os
import tensorflow as tf
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from data_preprocessing import *


#################################################
#                                               #
#               Functions Begin                 #
#                                               #
#################################################


def model_performance_analysis(data_path, resnet50_model):
    """
    inputs: data path - file path to the feature vectors of a mel spectrogram
            resnet50_model - file path to a pre-trained resnet50 model

    output: general roc assesment of model performance

    returns: accuracy of the model which calculated as the sum of the true_positive and true_negative predictions
             divided by the total number of predictions

    desc: reads in data from the data path.
          does predictions using the loaded model and assesses its performance
    """

    # load the data
    data = pickle.load(open(data_path, 'rb'))
    # extract the feature data and corresponding labels
    feature_data = []
    feature_labels = []
    for values in data.values():
        if len(values) != 0: # if list is not empty
            feature_data.append(values[0][0])
            feature_labels.append(values[0][1])

    # reshape the feature data and labels   
    n_row = feature_data[0].shape[0] # this is dependent on the 'seg' of the mel spectrogram
    n_column = feature_data[0].shape[1] # this is dependent on the number of mel splits(N_mel) normally +3 for some reason
    feature_data = np.array(feature_data)
    feature_data = np.array([x.reshape( (n_row, n_column) ) for x in feature_data] )
    feature_labels = tf.keras.utils.to_categorical(np.array(feature_labels), 2)

    # test model by performing predictions on the test set
    results = resnet50_model.predict(feature_data)

    # simple processing to give data on the four possible outcomes
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    for i in range(len(results)):
        print(results[i])
        if feature_labels[i][0] == 1:
            if results[i][0] > 0.5:
                true_positive += 1
            else:
                false_positive += 1
        else:
            if results[i][1] > 0.5:
                true_negative +=1
            else: 
                false_negative+=1

    print("LABELS")
    print(sum(feature_labels[:,0]))
    print(sum(feature_labels[:,1]))
    print("POSITIVES")
    print(true_positive)
    print(false_positive)
    print("NEGATIVES")
    print(true_negative)
    print(false_negative)
    print("DATA_PATH")
    print(data_path)
    
    if (true_positive+false_positive) != 0 and (true_negative+false_negative) != 0:
        AUC_score = (true_positive/(true_positive+false_positive) + true_negative/(true_negative+false_negative))*0.5
    else: 
        print("This fold has division by 0")
        AUC_score = 0

    return AUC_score


