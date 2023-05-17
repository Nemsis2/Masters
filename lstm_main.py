import torch as th
import torchvision as thv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from sklearn.metrics import roc_auc_score
from torchvision import transforms
import gc
import os
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
from tb_main import extract_all_train_data, extract_train_data, add_val_data, save_model
from tb_main import create_new_folder, train, extract_test_data, test, performance_assess, to_categorical

"""
date: 21/02/2023 

author: Michael Knight
"""

# set paths
K_FOLD_PATH = "../data/tb/combo/multi_folds/"
MODEL_PATH = "../models/tb/"

# choose which melspec we will be working on
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"


# set hyperpaperameters
BATCH_SIZE = 128
NUM_EPOCHS = 50
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)


"""
Fetch the training data
"""
def fetch_training_data(train_outer_fold, train_inner_fold, final_model):
    # fetch the data
    if train_inner_fold == None: # if no inner fold is defined get the entire outer fold
        # get the entire outer fold
        batch_data, batch_labels = extract_all_train_data(K_FOLD_PATH + MELSPEC, train_outer_fold, final_model)
    else:
        # get the train fold
        batch_data, batch_labels = extract_train_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)

        # if this is a final model include the val set
        if final_model == 1:
            batch_data, batch_labels = add_val_data(batch_data, batch_labels, train_outer_fold, train_inner_fold)
                
    return batch_data, batch_labels



"""
Create a bi_lstm model
"""
class bi_lstm(nn.Module):
    def __init__(self):
        super(bi_lstm, self).__init__()
        
        self.drop = nn.Dropout(p=0.5)
        self.bi_lstm = nn.LSTM(input_size=180, hidden_size=64, num_layers=5, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        x = self.drop(x)
        self.bi_lstm.flatten_parameters()
        out, (h_n, c_n) = self.bi_lstm(x)
        result = self.fc(h_n[-1])
        return result

"""
Create a bi_lstm package including:
Model
Optimizer(Adam)
Model name
"""
class bi_lstm_package():
    def __init__(self):
        self.model = bi_lstm()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.model_name = "bi_lstm_"



"""
Break the data and corresponding labels into batches
"""
def create_batches(data, labels, batch_size):
    batched_data = []
    batched_labels = []
    
    # find the maximum size to interpolate all points to
    max = 0
    for i in range(len(data)):
        if(data[i].shape[0] > max):
            max = data[i].shape[0]

    # reshape the data
    for i in range(len(data)):
        # zero pad the data to be max x 180
        data[i] = (np.pad(data[i], [(0,(max - data[i].shape[0])), (0,0)], mode='constant', constant_values=0))
        data[i] = data[i].reshape(1,max,180)


    # use batches when loading to prevent memory overflow
    for i in range(int(np.ceil(len(data)/batch_size))):
        if(len(data) > (i+1)*batch_size):
            batched_data.append(data[i*batch_size:(i+1)*batch_size])# get the data batch
            batched_labels.append(labels[i*batch_size:(i+1)*batch_size]) # get the corresponding labels
        else: 
            batched_data.append(data[i*batch_size:]) # get the data batch
            batched_labels.append(labels[i*batch_size:]) # get the corresponding labels
            

    # vstack the data
    for i in range(len(batched_data)):
        batched_data[i] = np.vstack(batched_data[i])
        batched_labels[i] = np.vstack(to_categorical(batched_labels[i],2))


    return batched_data, batched_labels




"""
Train the model
"""
def train_model(train_outer_fold, train_inner_fold, model, working_folder, epochs, final_model=0):
      # run through all the epochs
      for epoch in range(epochs):
            print("epoch=", epoch)

            # fetch the data
            batch_data, batch_labels = fetch_training_data(train_outer_fold, train_inner_fold, final_model)

            # batch the data
            batch_data, batch_labels = create_batches(batch_data, batch_labels, BATCH_SIZE)

            # train the model
            train(batch_data, batch_labels, model)

            # collect the garbage
            del batch_data, batch_labels
            gc.collect()

      save_model(model, working_folder, train_outer_fold, train_inner_fold, final_model, NUM_EPOCHS)     



"""
test a singular lstm model on the corresponding test set.
"""
def test_model(model, test_fold):
    #read in the test set
    test_batch_data, test_batch_labels = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)


    total_true_positive = 0
    total_false_positive = 0
    total_true_negative = 0
    total_false_negative = 0

    # batch the data 
    test_batch_data, test_batch_labels = create_batches(test_batch_data, test_batch_labels, BATCH_SIZE)
        
    for i in range(len(test_batch_data)):
        # do a forward pass through the model
        results = test(test_batch_data, model)

        # assess the accuracy of the model
        auc, true_positive, false_positive, true_negative, false_negative  = performance_assess(test_batch_labels, results)

        total_true_positive += true_positive
        total_false_positive += false_positive
        total_true_negative += true_negative
        total_false_negative += false_negative

    sens = total_true_positive/(total_true_positive + total_false_negative)
    spec = total_true_negative/(total_true_negative + total_false_positive)

    # display and log the AUC for the test set
    print("AUC for test_fold",test_fold, "=", auc)
    logging.basicConfig(filename="log.txt", filemode='a', level=logging.INFO)
    logging_info = "Final performance for test fold:", str(test_fold), "AUC:", str(auc), "Sens", str(sens), "Spec", str(spec)
    logging.info(logging_info)

    # mark variable and then call the garbage collector to ensure memory is freed
    del test_batch_data, test_batch_labels
    gc.collect()




for i in range(1): #this will be used to determine how many models should be made
    working_folder = create_new_folder(str(MODEL_PATH + "inner/"))
    
    for train_outer_fold in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", train_outer_fold)
            
            for train_inner_fold in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", train_inner_fold)
                model = bi_lstm_package()
                train_model(train_outer_fold, train_inner_fold, model, working_folder, NUM_EPOCHS, final_model=1)  





print("Beginning Testing")
folder_names = os.listdir(MODEL_PATH + "inner/")
folder_names.sort()
for working_folder in folder_names:
    # pass through all the outer folds
    print(int(working_folder))
    if int(working_folder) == 46:
            for test_outer_fold in range(NUM_OUTER_FOLDS):
                print("test_outer_fold=", test_outer_fold)
                
                # for each outer fold pass through all the inner folds
                for test_inner_fold in range(NUM_INNER_FOLDS):
                        print("test_inner_fold=", test_inner_fold)
                        model = pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                        "_inner_fold_" + str(test_inner_fold) + "_final_model", 'rb')) # load in the model
                        test_model(model, test_outer_fold)