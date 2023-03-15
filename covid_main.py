import tensorflow as tf
import pickle
import os, sys
import numpy as np
from sklearn.model_selection import train_test_split
from resnet_exp import ResNet50
from data_processing import *
from performance_analysis import *

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


"""
date: 15/02/2023 

author: Michael Knight

desc: main function which executes various functions to go from feature data to 
      trained models and them assess those models

functions:
"""


#############################################
#                                           #
#               MAIN FUNCTION               #
#                                           #
#############################################


# best average auc Resnet_50_feat_mel-26_frames-2048_segs-70 56% average over all folds 75% auc for k_fold_2 - covid data

# set desired outcomes
SPLIT_DATA = False
TRAIN_NEW_MODELS = False
ASSESS_MODEL_PERFORMANCE = True

# set global variables
k_folds = 5

# set paths
data_path = "../data/covid/raw/"
model_path = "../models/covid/"
k_fold_path = "../data/covid/k_folds/"


# split all the raw data 
if SPLIT_DATA == True:
    for raw_data_file in os.listdir(data_path):
        split_data(raw_data_file, k_folds)

# train models on the training data

"""
Add: some kind of hyperparameter optimization
"""

if TRAIN_NEW_MODELS == True:
    for data_filename in os.listdir(data_path):
        
        for test in range(k_folds):
            #set the model name
            model_name = "Resnet50_" + str(data_filename) + "_k_fold_" + str(test+1)

            if os.path.exists(model_path+model_name) == False:
                data = {}
                for train in range(k_folds):
                    # grab all none test data
                    if train != test:
                        data = data | pickle.load(open(k_fold_path + data_filename + "_k_fold_" + str(train+1), 'rb'))
                
                try:
                    # apply smote to the data and split into a training and test set
                    train_data, train_labels, test_data, test_labels, n_row, n_column = extract_data_info(data)

                    # create the model
                    model = ResNet50(input_shape=(n_row, n_column, 1), classes=2)
                    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

                    # train the model
                    output = model.fit(
                        x=train_data,
                        y=train_labels,
                        epochs=50,
                        batch_size=64,
                        validation_data= (test_data, test_labels))

                    # save the model
                    pickle.dump(model, open(model_path+model_name, 'wb'))

                    print("model:", model_name, "created")
                
                except:
                    print("Data set is empty move to next")

            else:
                    print("A model for this feature set already exists. Moving to next.")
            

if ASSESS_MODEL_PERFORMANCE == True:
    best_average_auc = 0
    best_average_model = "model"
    best_auc = 0
    best_model = "model"
    average_auc = 0
    
    for data_name in os.listdir(data_path):

        if os.path.exists(model_path + "Resnet50_" + data_name + "_k_fold_1"):
            
            for model_num in range(k_folds):
                # set the model name
                model_name = "Resnet50_"+ data_name + "_k_fold_" + str(model_num+1)
                
                # load the model and silence it from printing to the cmd line
                sys.stdout = open(os.devnull, "w")
                try:
                    model = pickle.load(open((model_path + model_name), 'rb'))
                    sys.stdout = sys.__stdout__

                    # get test data for this model
                    splt_word = 'Resnet50_'
                    res = str(model_name).split(splt_word, 1)
                    test_data_path = k_fold_path + res[1]

                    # evaluate the model performance
                    auc = model_performance_analysis(test_data_path, model)

                    average_auc += auc
                    
                    if auc > best_auc:
                        best_auc = auc
                        best_model = model_name
                except:
                    print("failed to load model")

        if average_auc > best_average_auc:
            best_average_auc = average_auc
            best_average_model = ("Resnet50_" + data_name)

        # reset the average    
        average_auc = 0

        print("current best auc:", best_auc)
        print("current best model:", best_model)
        print("current best average auc:", best_average_auc/k_folds)
        print("current best model:", best_average_model)


# load the model
#model_name = 'Resnet50_feat_mel-26_frames-1024_segs-120'
#model = pickle.load(open("../models/Resnet50_feat_mel-26_frames-1024_segs-120", 'rb'))

# get eval data for this model
#splt_word = 'Resnet50_'
#res = str(model_name).split(splt_word, 1)
#eval_data_path = eval_path + res[1]

# evaluate the model performance
#accuracy = model_performance_analysis(eval_data_path, model)