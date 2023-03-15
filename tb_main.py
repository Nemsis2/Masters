import torch as th
import torchvision as thv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import transforms
import gc
import pickle
import numpy as np
import logging
import matplotlib.pyplot as plt
from resnet import *


"""
date: 21/02/2023 

author: Michael Knight

desc: 

functions:
"""


"""
Train the teacher model:
    Implement ensemble models as this can be used for the student network

    
Create the student model:
      maybe a feed-forward nn
      try a cnn


Train the student model:
    Use KL loss and CE loss
    KL loss is loss between softmax of student vs softmax of teacher
    CE loss is loss between prediction of student vs actual label

    Also look into using the dev set as the training set for the student?
    Ask prof about the effects of using the training set for the teacher as the training set for the student as well
"""


#############################################
#                                           #
#               MAIN FUNCTION               #
#                                           #
#############################################

# dont forget to activate pytorch env otherwise no worky

device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)

# set paths
K_FOLD_PATH = "../data/tb/combo/multi_folds/"
MODEL_PATH = "../models/tb/"

# choose which melspec we will be working on
# 128 and 80 both seem to be worse
MELSPEC = "180_melspec_fold_"

# set hyperpaperameters
BATCH_SIZE = 128
LOAD_BATCH_SIZE = 1024
num_epochs = 25
num_outer_folds = 3
num_inner_folds = 4
pruning_percentage = 0.3

# training options for the models
train_model_flag = 1
prune_model_flag = 0
student_model_flag = 0

# testing options for the models
test_model_flag = 0
VAL_MODEL_TEST_FLAG = 1
test_student_model_flag = 0
generate_graph = 0



#https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
def to_categorical(y, num_classes):
      """ 1-hot encodes a tensor """
      return np.eye(num_classes, dtype='float')[np.array(y).astype(int)]



#https://github.com/sahandilshan/Simple-NN-Compression/blob/main/Simple_MNIST_Compression.ipynb
def get_pruned_parameters_countget_pruned_parameters_count(pruned_model):
    params = 0
    for param in pruned_model.parameters():
        if param is not None:
            params += th.nonzero(param).size(0)
    return params



#https://stackoverflow.com/questions/54846905/pytorch-get-all-layers-of-model
def nested_children(m: th.nn.Module):
    children = dict(m.named_children())
    output = {}
    if children == {}:
        # if module has no children; m is last child! :O
        return m
    else:
        # look for children from children... to the last child!
        for name, child in children.items():
            try:
                output[name] = nested_children(child)
            except TypeError:
                output[name] = nested_children(child)
    return output



def to_softmax(results):
      softmax = nn.Softmax(dim=1)
      results = softmax(results)

      return results


def prune_model(model, percentage):
      total_params_before = get_pruned_parameters_countget_pruned_parameters_count(model)
      l = nested_children(model)

      parameters_to_prune = []
      parameters = []
      for outer_key in l.keys():
            if "layer" in outer_key:
                  for inner_key in l[outer_key]:
                        for nets in l[outer_key][inner_key].keys():
                              if ("conv" or "bn") in nets:
                                    parameters_to_prune.append((l[outer_key][inner_key][nets], 'weight'))
                                    parameters.append(l[outer_key][inner_key][nets])

      prune.global_unstructured(
      parameters_to_prune,
      pruning_method=prune.L1Unstructured,
      amount= percentage, # Specifying the percentage
      )

      for layer in parameters:
            prune.remove(layer, 'weight')

      total_params_after = get_pruned_parameters_countget_pruned_parameters_count(model)

      return model, (total_params_after/total_params_before)



def extract_train_data(path, outer_fold, current_inner_fold):
      batch = []
      # read in the data located at the path 
      data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))

      # zip the information from the dictionary into a list of arrays
      for inner_fold in data.keys():
            if inner_fold == ("fold_"+str(current_inner_fold)):
                  for i, t in zip( data[inner_fold]['train']['inps'], data[inner_fold]['train']['tgts']):
                        batch.append([i, t])

      batch = np.array(batch, dtype=object)
      
      return batch



def extract_val_data(path, outer_fold, current_inner_fold):
      batch = []
      # read in the data located at the path 
      data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))

      # zip the information from the dictionary into a list of arrays
      for inner_fold in data.keys():
            if inner_fold == ("fold_"+str(current_inner_fold)):
                  for labels in data[inner_fold]['val'].keys():
                        for i,t in zip( data[inner_fold]['val'][labels]['inps'], data[inner_fold]['val'][labels]['tgts']):
                              batch.append([i,t])

      batch = np.array(batch, dtype=object)
      
      return batch



def extract_test_data(path, fold):
      batch = []
      # read in the data located at the path 
      data = pickle.load(open(path + str(fold) + ".pkl", 'rb'))

      # zip the information from the dictionary into a list of arrays
      for inner_fold in data.keys():
            for i, t in zip( data[inner_fold]['inps'], data[inner_fold]['tgts']):
                  batch.append([i, t])

      batch = np.array(batch, dtype=object)
      
      return batch



def reshape_data(data):
      # create the transform
      transform_spectra = transforms.Compose([  transforms.Resize((224,224)),
                                                ])
      
      del_indx = []
      for i in range(len(data)):
            if 224 > data[i].shape[0]:
                  # zero pad the data to be 224 x 180
                  data[i] = (np.pad(data[i], [(0,(224 - data[i].shape[0])), (0,0)], mode='constant', constant_values=0))
                  # bilinear interpolate the data to be 224 x 224
                  data[i] = transform_spectra(th.tensor(data[i]).unsqueeze(0).repeat(3,1,1)).unsqueeze(0)
            else:
                  # for now just bin the data
                  del_indx.append(i)

      # delete the large data
      data = np.delete(data, del_indx)
      # prep and return the data
      data = np.vstack(data)

      return data, del_indx



def process_data(x, y):
      x_return = []
      y_return = []
      # use batches when loading to prevent memory overflow
      for i in range(int(np.ceil(len(x)/BATCH_SIZE))):
            if(len(x) > (i+1)*BATCH_SIZE):
                  x_batch = x[i*BATCH_SIZE:(i+1)*BATCH_SIZE] # get the data batch
                  y_batch = y[i*BATCH_SIZE:(i+1)*BATCH_SIZE] # get the corresponding labels
                  x_processed, x_del_indx = reshape_data(x_batch) # process the data to be 3 x 224 x 224 and delete any data which cannot be made to fit this shape
                  y_batch = np.delete(y_batch, x_del_indx) # for any removed data delete the corresponding labels
                  y_batch = to_categorical(y_batch,2)
                  x_return.append(x_processed)
                  y_return.append(y_batch)
            else: 
                  x_batch = x[i*BATCH_SIZE:] # get the data batch
                  y_batch = y[i*BATCH_SIZE:] # get the corresponding labels
                  x_processed, x_del_indx = reshape_data(x_batch) # process the data to be 3 x 224 x 224 and delete any data which cannot be made to fit this shape
                  y_batch = np.delete(y_batch, x_del_indx) # for any removed data delete the corresponding labels
                  y_batch = to_categorical(y_batch,2)
                  x_return.append(x_processed)
                  y_return.append(y_batch)


      return x_return, y_return



def get_batch_of_data(data, labels, batch_size, i):
      # handles the case where load_batch_size number of samples are still available
      if i < int(np.ceil(len(data)/batch_size)):
            x = data[i*batch_size:(i+1)*batch_size]
            y = labels[i*batch_size:(i+1)*batch_size]
            x, y = process_data(x, y)
      # handles the case when there is less than load_batch_size number of samples left in the batch
      else:
            x = data[i*batch_size:]
            y = labels[i*batch_size:]
            x, y = process_data(x, y)

      return x,y



def train(x, y, model, optimizer, criterion):
      model = model.to(device)
      # use batches when loading to prevent memory overflow
      #optimizer.zero_grad() # set the optimizer grad to zero
      # loss = 0
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu
            y_batch = th.as_tensor(y[i]).to(device) # grab the label

            # run through the model
            results = model(x_batch) # get the model to make predictions
            loss = criterion(results, y_batch) # calculate the loss
            loss.backward() # use back prop
            optimizer.step() # update the model weights
            optimizer.zero_grad() # set the optimizer grad to zero



def test(x, models):
      all_model_results = []
      for i in range(len(models)):
            results = []
            with th.no_grad():
                  for j in range(len(x)):
                        results.append((models[i](th.tensor(x[j]).to(device))).cpu())
            
            all_model_results.append(results)

      return all_model_results



def performance_assess(y, results, test_type):
      true_positive = 0
      false_positive = 0
      true_negative = 0
      false_negative = 0

      if test_type == 'test':
            for i in range(len(results)):
                  if i == 0:
                        sum = np.array(results[i], dtype=object)
                  else:
                        sum += np.array(results[i], dtype=object)

            average_of_models = sum/4
      else:
            average_of_models = results

      
      for i in range(len(average_of_models)):
            for j in range(len(average_of_models[i])):
                  if y[i][j][0] == 1:
                        if average_of_models[i][j][0] > 0.5:
                              true_positive += 1
                        else:
                              false_positive += 1
                  else:
                        if average_of_models[i][j][1] > 0.5:
                              true_negative += 1
                        else: 
                              false_negative += 1

      AUC_score = (true_positive/(true_positive+false_positive) + true_negative/(true_negative+false_negative))*0.5

      return AUC_score, true_positive, false_positive, true_negative, false_negative



def train_model(train_outer_fold, train_inner_fold, model, criterion, optimizer):
      # run through all the epochs
      for epoch in range(num_epochs):

            print("epoch=", epoch)

            # get the train fold
            batch = extract_train_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)
            batch_data = batch[:,0] # get the data from the batch
            batch_labels = batch[:,1] # get the labels from the batch

            print("batch=", batch.shape)
            
            # break the data into segments load_batch_size long
            for i in range(int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))):
                  print("Seq-batch=", i)
                  x_train, y_train = get_batch_of_data(batch_data, batch_labels, LOAD_BATCH_SIZE, i)
                  
                  train(x_train, y_train, model, optimizer, criterion) # train the model on the current batch

                  del x_train, y_train 
                  gc.collect()


            del batch_data, batch_labels, batch
            gc.collect()

      pickle.dump(model, open(MODEL_PATH + "resnet18_" + MELSPEC + str(train_outer_fold) + "_inner_fold_" + str(train_inner_fold), 'wb')) # save the model



def validate_model(model, train_outer_fold, train_inner_fold):
      # get the train fold
      batch = extract_val_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)
      batch_data = batch[:,0] # get the data from the batch
      batch_labels = batch[:,1] # get the labels from the batch

      AUC_for_k_fold = 0
      # break the data into segments load_batch_size long
      for i in range(int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))):

            x_val, y_val = get_batch_of_data(batch_data, batch_labels, LOAD_BATCH_SIZE, i)


            results = []
            for i in range(len(x_val)):
                  with th.no_grad():
                        results.append(to_softmax((model(th.tensor(x_val[i]).to(device))).cpu()))

            AUC, true_pos, false_pos, true_neg, false_neg = performance_assess(y_val, results, test_type='val') # check the accuracy of the model

            AUC_for_k_fold += AUC


      AUC_for_k_fold = AUC_for_k_fold/int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))

      print("AUC for outer_fold", train_outer_fold, "and inner_fold", train_inner_fold, "=", AUC_for_k_fold )

      del batch, batch_labels, x_val, y_val
      gc.collect()



def test_models(models, test_fold, is_pruned):
      test_batch = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)
      test_batch_data = test_batch[:,0]
      test_batch_labels = test_batch[:,1]

      AUC_for_k_fold = 0
      # break the data into segments load_batch_size long
      for i in range(int(np.ceil(len(test_batch_data)/LOAD_BATCH_SIZE))):
            print("test", i)
            
            x_test, y_test = get_batch_of_data(test_batch_data, test_batch_labels, LOAD_BATCH_SIZE, i)

            results = test(x_test, models) # do a forward pass through the model

            for i in range(len(results)):
                  for j in range(len(results[i])):
                        results[i][j] = to_softmax(results[i][j])

            AUC, true_pos, false_pos, true_neg, false_neg = performance_assess(y_test, results, test_type='test') # check the accuracy of the model

            AUC_for_k_fold += AUC

            print("For test batch", i, "AUC_score:", AUC)

      AUC_for_k_fold = AUC_for_k_fold/int(np.ceil(len(test_batch_data)/LOAD_BATCH_SIZE))

      print("AUC for k_fold", AUC_for_k_fold )

      logging.basicConfig(filename="log.txt", filemode='a', level=logging.INFO)
      logging_info = "Final performance for test fold:", str(test_fold), "model is pruned to", is_pruned, "of original size" " - ", str(AUC_for_k_fold)
      logging.info(logging_info)


      del test_batch, test_batch_labels, x_test, y_test
      gc.collect()



#######################################################
#                                                     #
#                                                     #
#                 MAIN FUNCTION                       #
#                                                     #
#                                                     #
#######################################################



if train_model_flag == 1:
      print("Beginning Training")

      for train_outer_fold in range(num_outer_folds):
            print("train_outer_fold=", train_outer_fold)
            for train_inner_fold in range(num_inner_folds):
                  print("train_inner_fold=", train_inner_fold)
                  model = ResNet_2layer(ResidualBlock1, [1, 1, 1, 1], num_classes=2)
                  criterion = nn.CrossEntropyLoss()
                  optimizer = optim.Adam(model.parameters(), lr=0.0001)
                  train_model(train_outer_fold, train_inner_fold, model, criterion, optimizer)        



# wont work rn
if prune_model_flag == 1:
      print("Beginning Pruning")
      
      for prune_fold in range(num_outer_folds):
            print("prune_fold=", prune_fold)
            model = pickle.load(open((MODEL_PATH + "resnet18_" + MELSPEC + str(prune_fold)), 'rb')) # load in the model
            
            pruned_model, percentage_actually_pruned = prune_model(model, pruning_percentage)

            test_models(pruned_model, prune_fold, percentage_actually_pruned)

            pickle.dump(pruned_model, open(MODEL_PATH + "resnet18_" + MELSPEC + str(prune_fold) + "_pruned", 'wb')) # save the model



# wont work rn
if student_model_flag == 1:
      print("Creating student model")
      for train_fold in range(num_outer_folds):
            model = ResNet_2layer(ResidualBlock1, [1,1,1,1], num_classes=2).to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=0.0001)
            print("Student train_fold=", train_fold)
            train_model(train_outer_fold, train_inner_fold, model, criterion, optimizer)   



if VAL_MODEL_TEST_FLAG == 1:
      print("Beginning Model Validation")
      for train_outer_fold in range(num_outer_folds):
            print("test_outer_fold=", train_outer_fold)
            

            for train_inner_fold in range(num_inner_folds):
                  print("test_inner_fold=", train_inner_fold)
                  model = (pickle.load(open(MODEL_PATH + "resnet18_" + MELSPEC + str(train_outer_fold) + "_inner_fold_" + str(train_inner_fold), 'rb'))) # load in the model
                  validate_model(model, train_outer_fold, train_inner_fold)



if test_model_flag == 1:
      print("Beginning Testing")
      for test_outer_fold in range(num_outer_folds):
            print("test_outer_fold=", test_outer_fold)
            
            # get all the models
            models = []
            for test_inner_fold in range(num_inner_folds):
                  print("test_inner_fold=", test_inner_fold)
                  models.append(pickle.load(open(MODEL_PATH + "resnet18_" + MELSPEC + str(test_outer_fold) + "_inner_fold_" + str(test_inner_fold), 'rb'))) # load in the model
            
            test_models(models, test_outer_fold, 1)



# wont work rn
if test_student_model_flag == 1:
      print("Testing student model")
      for test_fold in range(num_outer_folds):
            print("test_fold=", test_fold)
            model = pickle.load(open((MODEL_PATH + "student_resnet_" + MELSPEC + str(test_fold)), 'rb')) # load in the model
            test_models(model, test_fold, 1)



# real clunky delete later
if generate_graph == 1:
      infile = "log.txt"

      important = []

      with open(infile) as f:
            f = f.readlines()

      fold = []
      size_of_original_model = []
      performance = []
      for line in f:
            seperated_line = line.split(", ")
            fold.append(seperated_line[1].split("'")[1])
            size_of_original_model.append(seperated_line[3])
            performance.append(seperated_line[5].split("'")[1])

      size_per_performance = []
      average_performance = []
      for i in range(int(len(fold)/3)):
            size_per_performance.append(np.array(size_of_original_model[i*3], dtype=np.float128)*100)
            average_performance.append((np.sum(np.array(performance[(i*3):(i*3+3)], dtype=np.float128))/3)*100)

      plt.plot(size_per_performance, average_performance)
      plt.xlabel('Percentage of original model')
      plt.ylabel('Percentage average performance across all folds')
      plt.show()