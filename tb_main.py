import torch as th
import torchvision as thv
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from torchvision import transforms
import gc
import os
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
Done: 
Re-implement inner fold models and then mess with hyperparameters until optimal is found for Resnet6.

Do a comparison between an ensemble of Resnet6(val_set included) vs a Resnet6 trained on the full train set(val included).

Create a ensemble model from the Resnet6 models.

To Do:
Create objects for resnet18, resnet18(with more skips) resnet12, resnet6(4 deep) and resnet6(2 deep).

Add random seed selection as well as proper storage.

Run tests.

Improve test managment and process all results.

Re-organise to improve readability (create test/train scripts etc.)

Possible extensions:
Implement pre-training and perform all tests again.

Look into pruning and see how this affects the models.

Just make a tiny model and see how it will perform.
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
MODEL_MELSPEC = "melspec_180"

# set hyperpaperameters
BATCH_SIZE = 128
LOAD_BATCH_SIZE = 1024
NUM_EPOCHS = 15
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
PRUNING_PERCENTAGE = 0.3

# training options for the models
TRAIN_INNER_MODEL_FLAG = 0
TRAIN_MODEL_OUTER_ONLY_FLAG = 1
TRAIN_ENSEMBLE_MODEL_FLAG = 0
PRUNE_MODEL_FLAG = 0

# testing options for the models
TEST_INNER_MODEL_FLAG = 0
TEST_INNER_ENSEMBLE_MODELS_FLAG = 0
TEST_OUTER_ONLY_MODEL_FLAG = 0
TEST_ENSEMBLE_MODEL_FLAG = 0
VAL_MODEL_TEST_FLAG = 0
GENERATE_GRAPH_FLAG = 0


class Resnet18():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [2, 2 ,2 ,2], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_18_"


class Resnet6_4Deep():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock1, [1, 1, 1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_6_4Deep_"


class Resnet6_2Deep():
      def __init__(self):
            self.model = ResNet_2layer(ResidualBlock2, [2, 2], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_6_2Deep_"


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



#https://stackoverflow.com/questions/52679734/how-to-create-new-folder-each-time-i-run-script
def create_new_folder(folder_path):
      folder_names = os.listdir(folder_path)
      if len(os.listdir(folder_path)) != 0:
            last_number = max([int(name) for name in folder_names if name.isnumeric()])
            new_name = str(last_number + 1).zfill(4)
      else:
            new_name = str("0001")
      
      os.mkdir((folder_path+new_name))

      return new_name


"""
save the model
"""
def save_model(model, working_folder, train_outer_fold, train_inner_fold, final_model, epochs):
      if train_inner_fold == None:
            if final_model == 0:
                  pickle.dump(model.model, open(MODEL_PATH + "outer/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_epochs_" + str(epochs), 'wb')) # save the model
            else:
                  pickle.dump(model.model, open(MODEL_PATH + "outer/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_final_model", 'wb')) # save the model
      else:
            if final_model == 0:
                  pickle.dump(model.model, open(MODEL_PATH + "inner/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_epochs_" + str(epochs), 'wb')) # save the model
            else:
                  pickle.dump(model.model, open(MODEL_PATH + "inner/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_final_model", 'wb')) # save the model


"""
add the validation set for a specific inner and outer fold
"""
def add_val_data(batch_data, batch_labels, train_outer_fold, train_inner_fold):
      val_batch_data, val_batch_labels = extract_val_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)
      batch_data = np.concatenate((batch_data, val_batch_data))
      batch_labels = np.concatenate((batch_labels, val_batch_labels))

      return batch_data, batch_labels


"""
normalizes output of the model using a softmax function
"""
def to_softmax(results):
      softmax = nn.Softmax(dim=1)
      results = softmax(results)
      return results


"""
log info from a test
"""
def log_test_info(test_fold, AUC_for_k_fold):
      logging.basicConfig(filename="log.txt", filemode='a', level=logging.INFO)
      logging_info = "Final performance for test fold", str(test_fold), ":", str(AUC_for_k_fold)
      logging.info(logging_info)


"""
perform pruning on a resnet model
"""
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


"""
get train data for a specific inner and outer fold
"""
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

      batch_data = batch[:,0] # get the data from the batch
      batch_labels = batch[:,1] # get the labels from the batch

      return batch_data, batch_labels


"""
get all data in an outer fold
extracts only train data if final_model =0 and includes validation data if final_model=1
"""
def extract_all_train_data(path, outer_fold, final_model=0):
      batch = []
      # read in the data located at the path 
      data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))

      # zip the information from the dictionary into a list of arrays
      for inner_fold in data.keys():
            # get the training data
            for i, t in zip( data[inner_fold]['train']['inps'], data[inner_fold]['train']['tgts']):
                  batch.append([i, t])

            if final_model == 1:
                  for labels in data[inner_fold]['val'].keys():
                        # get the validation data
                        for i,t in zip( data[inner_fold]['val'][labels]['inps'], data[inner_fold]['val'][labels]['tgts']):
                              batch.append([i,t])

      batch = np.array(batch, dtype=object)
      batch_data = batch[:,0] # get the data from the batch
      batch_labels = batch[:,1] # get the labels from the batch
      
      return batch_data, batch_labels


"""
get val data for a specific inner and outer fold
"""
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


      batch_data = batch[:,0] # get the data from the batch
      batch_labels = batch[:,1] # get the labels from the batch
      
      return batch_data, batch_labels


"""
get the test data
"""
def extract_test_data(path, fold):
      batch = []
      # read in the data located at the path 
      data = pickle.load(open(path + str(fold) + ".pkl", 'rb'))

      # zip the information from the dictionary into a list of arrays
      for inner_fold in data.keys():
            for i, t in zip( data[inner_fold]['inps'], data[inner_fold]['tgts']):
                  batch.append([i, t])

      batch = np.array(batch, dtype=object)

      batch_data = batch[:,0]
      batch_labels = batch[:,1]
      
      return batch_data, batch_labels


"""
reshape data to size 3 x 224 x 224
"""
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


"""
process a batch of data transforming it to size 3 x 224 x 224
"""
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


"""
create a batch of batch sized samples
"""
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


"""
complete a training step for a batch of data
"""
def train(x, y, model):
      model.model = model.model.to(device)
      # use batches when loading to prevent memory overflow
      #optimizer.zero_grad() # set the optimizer grad to zero
      # loss = 0
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu
            y_batch = th.as_tensor(y[i]).to(device) # grab the label

            # run through the model
            results = model.model(x_batch) # get the model to make predictions
            loss = model.criterion(results, y_batch) # calculate the loss
            loss.backward() # use back prop
            model.optimizer.step() # update the model weights
            model.optimizer.zero_grad() # set the optimizer grad to zero


"""
complete a training step for a batch of data
"""
def ensemble_train(x, y, model, inner_model, criterion_kl):
      model.model = model.model.to(device)
      inner_model = inner_model.to(device)
      
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu
            y_batch = th.as_tensor(y[i]).to(device) # grab the label

            # run through the model
            results = model.model(x_batch) # get the model to make predictions
            inner_results = get_predictions(x_batch, inner_model).to(device) #returns softmax predictions of the inner model
            ce_loss = model.criterion(results, y_batch) # calculate the loss
            ce_loss.backward(retain_graph=True) # use back prop
            results = F.log_softmax(results, dim=1) # gets the log softmax of the output of the ensemble model
            kl_loss = criterion_kl(results, inner_results) # calculate the loss
            kl_loss.backward(retain_graph=True) # use back prop
            model.optimizer.step() # update the model weights
            model.optimizer.zero_grad() # set the optimizer grad to zero

"""
complete a forward pass through the model without storing the gradient
"""
def test(x, model):
      with th.no_grad():
            results = []
            for i in range(len(x)):
                  results.append(to_softmax((model(th.tensor(x[i]).to(device))).cpu()))

      return results


"""
compare labels with predicted labels and get the AUC as well as true_p, flase_p, true_n and false_n
"""
def performance_assess(y, results):
      true_positive = 0
      false_positive = 0
      true_negative = 0
      false_negative = 0

      
      for i in range(len(results)):
            for j in range(len(results[i])):
                  if y[i][j][0] == 1:
                        if results[i][j][0] > 0.5:
                              true_positive += 1
                        else:
                              false_positive += 1
                  else:
                        if results[i][j][1] > 0.5:
                              true_negative += 1
                        else: 
                              false_negative += 1

      AUC_score = (true_positive/(true_positive+false_positive) + true_negative/(true_negative+false_negative))*0.5

      return AUC_score, true_positive, false_positive, true_negative, false_negative

"""
get predictions from the test of a model
"""
def get_predictions(x_batch, inner_model):
      with th.no_grad():
            results = (to_softmax((inner_model(th.tensor(x_batch).to(device))).cpu()))

      return results


"""
train a model on a specific inner fold within an outer fold.
can be set to include or exclude validation set as neccessary.
"""
def train_model(train_outer_fold, train_inner_fold, model, working_folder, final_model=0):
      # run through all the epochs
      for epoch in range(NUM_EPOCHS):
            print("epoch=", epoch)

            if train_inner_fold == None:
                  batch_data, batch_labels = extract_all_train_data(K_FOLD_PATH + MELSPEC, train_outer_fold, final_model)
            else:
                  # get the train fold
                  batch_data, batch_labels = extract_train_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)

                  # if this is a final model include the val set
                  if final_model == 1:
                        batch_data, batch_labels = add_val_data(batch_data, batch_labels, train_outer_fold, train_inner_fold)

            print("batch=", batch_data.shape)
            
            # break the data into segments load_batch_size long for processing
            for i in range(int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))):
                  print("Seq-batch=", i)
                  x_train, y_train = get_batch_of_data(batch_data, batch_labels, LOAD_BATCH_SIZE, i) # process a load_batch_size of data and break into batches
                  train(x_train, y_train, model) # train the model on the current load_batch_size working in batches
                  
                  # collect the garbage
                  del x_train, y_train 
                  gc.collect()

            # collect the garbage
            del batch_data, batch_labels
            gc.collect()

      save_model(model, working_folder, train_outer_fold, train_inner_fold, final_model, NUM_EPOCHS)


"""
trains the ensemble model on a specific outer and inner fold
"""
def train_ensemble_model(train_outer_fold, train_inner_fold, model, criterion_kl, epochs, final_model=0):
      # get the train fold
      batch_data, batch_labels = extract_train_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)

      if final_model == 1:
            batch_data, batch_labels = add_val_data(batch_data, batch_labels, train_outer_fold, train_inner_fold)

            inner_model = pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/resnet_18_" + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                           "_inner_fold_" + str(train_inner_fold) + "_final_model", 'rb')) # save the model
      else:
            inner_model = pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) +
                                           "_inner_fold_" + str(train_inner_fold) + "_epochs_" + epochs, 'rb')) # save the model

      print("batch=", batch_data.shape)
      
      # break the data into segments load_batch_size long
      for i in range(int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))):
            print("Seq-batch=", i)
            x_train, y_train = get_batch_of_data(batch_data, batch_labels, LOAD_BATCH_SIZE, i)
            
            ensemble_train(x_train, y_train, model, inner_model, criterion_kl) # train the model on the current batch

            del x_train, y_train 
            gc.collect()


      del batch_data, batch_labels
      gc.collect()


"""
validates a model by testing its performance on the corresponding validation fold.
"""
def validate_model(model, train_outer_fold, train_inner_fold):
      # get the train fold
      batch_data, batch_labels = extract_val_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)

      AUC_for_k_fold = 0
      # break the data into segments load_batch_size long
      for i in range(int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))):

            x_val, y_val = get_batch_of_data(batch_data, batch_labels, LOAD_BATCH_SIZE, i)            


            results = []
            for i in range(len(x_val)):
                  with th.no_grad():
                        results.append(to_softmax((model(th.tensor(x_val[i]).to(device))).cpu()))

            AUC, true_pos, false_pos, true_neg, false_neg = performance_assess(y_val, results) # check the accuracy of the model

            AUC_for_k_fold += AUC


      AUC_for_k_fold = AUC_for_k_fold/int(np.ceil(len(batch_data)/LOAD_BATCH_SIZE))

      print("AUC for outer_fold", train_outer_fold, "and inner_fold", train_inner_fold, "=", AUC_for_k_fold )

      logging.basicConfig(filename="log.txt", filemode='a', level=logging.INFO)
      logging_info = "Final performance for outer fold:", str(train_outer_fold), ":inner fold:",str(train_inner_fold), ":AUC:", str(AUC_for_k_fold)
      logging.info(logging_info)

      del batch_labels, x_val, y_val
      gc.collect()


"""
test a singular model on the corresponding test set.
"""
def test_model(model, test_fold):
      #read in the test set
      test_batch_data, test_batch_labels = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)

      AUC_for_k_fold = 0
      # break the data into processing segments load_batch_size long
      for i in range(int(np.ceil(len(test_batch_data)/LOAD_BATCH_SIZE))):
            print("test", i)
            
            # process the data and seperate into batches to be used in training
            x_test, y_test = get_batch_of_data(test_batch_data, test_batch_labels, LOAD_BATCH_SIZE, i)

            # do a forward pass through the model
            results = test(x_test, model)

            # assess the accuracy of the model
            AUC, true_pos, false_pos, true_neg, false_neg = performance_assess(y_test, results)
            AUC_for_k_fold += AUC

            # display the performance for this process batch
            print("For test batch", i, "AUC_score:", AUC)

      # calculate the total AUC for all process batches in the test set
      AUC_for_k_fold = AUC_for_k_fold/int(np.ceil(len(test_batch_data)/LOAD_BATCH_SIZE))

      # display and log the AUC for the test set
      print("AUC for k_fold", AUC_for_k_fold )
      log_test_info(test_fold, AUC_for_k_fold)

      # mark variable and then call the garbage collector to ensure memory is freed
      del test_batch_data, test_batch_labels, x_test, y_test
      gc.collect()


"""
test multiple models on the test set using the average decision among all models to make a prediction.
"""
def test_models(models, test_fold):
      test_batch_data, test_batch_labels = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)

      AUC_for_k_fold = 0
      # break the data into segments load_batch_size long
      for i in range(int(np.ceil(len(test_batch_data)/LOAD_BATCH_SIZE))):
            print("test", i)
            
            x_test, y_test = get_batch_of_data(test_batch_data, test_batch_labels, LOAD_BATCH_SIZE, i)

            results = []
            for model in models:
                  results.append(test(x_test, model)) # do a forward pass through the models

            # change all values to softmax
            for i in range(len(results)):
                  for j in range(len(results[i])):
                        results[i][j] = to_softmax(results[i][j])
                        
            final_results = results[0]
            for i in range(1,len(results)):
                  print("i=",i)
                  for j in range(len(results[i])):
                        final_results[j] += results[i][j]

            for i in range(len(final_results)):
                  final_results[i] = final_results[i]/len(results)

            AUC, true_pos, false_pos, true_neg, false_neg = performance_assess(y_test, final_results) # check the accuracy of the model

            AUC_for_k_fold += AUC

            print("For test batch", i, "AUC_score:", AUC)

      AUC_for_k_fold = AUC_for_k_fold/int(np.ceil(len(test_batch_data)/LOAD_BATCH_SIZE))

      print("AUC for k_fold", AUC_for_k_fold )

      logging.basicConfig(filename="log.txt", filemode='a', level=logging.INFO)
      logging_info = "Final performance for test fold:", str(test_fold), str(AUC_for_k_fold)
      logging.info(logging_info)


      del test_batch_labels, x_test, y_test
      gc.collect()



#######################################################
#                                                     #
#                                                     #
#                 MAIN FUNCTION                       #
#                                                     #
#                                                     #
#######################################################



"""
train a model for each inner fold within each outer fold resulting in inner_fold*outer_fold number of models.
trains only on training data.m
"""
if TRAIN_INNER_MODEL_FLAG == 1:
      for i in range(NUM_EPOCHS):
            working_folder = create_new_folder(str(MODEL_PATH + "inner/"))
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                  print("train_outer_fold=", train_outer_fold)
                  for train_inner_fold in range(NUM_INNER_FOLDS):
                        print("train_inner_fold=", train_inner_fold)
                        model = Resnet6_2Deep()
                        train_model(train_outer_fold, train_inner_fold, model, working_folder, final_model=1)        


"""
train a model for each outer_fold
can be set to either use both train and val data or just train data
results in outer_fold number of models.
"""
if TRAIN_MODEL_OUTER_ONLY_FLAG == 1:
      for i in range(NUM_EPOCHS):
            working_folder = create_new_folder(str(MODEL_PATH + "outer/"))
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                  print("train_outer_fold=", train_outer_fold)
                  model = Resnet6_2Deep()
                  train_model(train_outer_fold, None, model, working_folder, final_model=1)        


"""
trains an ensemble model using the inner models and the original data
"""
if TRAIN_ENSEMBLE_MODEL_FLAG == 1:
      for i in range(NUM_EPOCHS):
            print("Beginning Training")
            working_folder = create_new_folder(str(MODEL_PATH + "ensemble/"))
            
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                  print("train_outer_fold=", train_outer_fold)
                  model = Resnet6_4Deep()
                  criterion_kl = nn.KLDivLoss()
                  for epoch in range(NUM_EPOCHS):
                        for train_inner_fold in range(NUM_INNER_FOLDS):
                              print("train_inner_fold=", train_inner_fold)
                              train_ensemble_model(train_outer_fold, train_inner_fold, model, criterion_kl, NUM_EPOCHS, final_model=1)

                  pickle.dump(model.model, open((MODEL_PATH + "ensemble/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" 
                                                + str(train_outer_fold)), 'wb')) # save the model
                     


# wont work rn
if PRUNE_MODEL_FLAG == 1:
      print("Beginning Pruning")
      
      for prune_fold in range(NUM_OUTER_FOLDS):
            print("prune_fold=", prune_fold)
            model = pickle.load(open((MODEL_PATH + "resnet18_" + MELSPEC + str(prune_fold)), 'rb')) # load in the model
            
            pruned_model, percentage_actually_pruned = prune_model(model, PRUNING_PERCENTAGE)

            test_models(pruned_model, prune_fold, percentage_actually_pruned)

            pickle.dump(pruned_model, open(MODEL_PATH + "resnet_" + MELSPEC + str(prune_fold) + "_pruned", 'wb')) # save the model


"""
used for inner fold based models only.
validates each model by assessing its performance on its corresponding validation set.
"""
if VAL_MODEL_TEST_FLAG == 1:
      print("Beginning Validation")
      folder_names = os.listdir(MODEL_PATH + "inner/")
      folder_names.sort()
      print(folder_names)
      for i in range(len(folder_names)):
            # pass through all the outer folds
            for val_outer_fold in range(NUM_OUTER_FOLDS):
                  print("val_outer_fold=", val_outer_fold)
                  
                  # for each outer fold pass through all the inner folds
                  for val_inner_fold in range(NUM_INNER_FOLDS):
                        print("val_inner_fold=", val_inner_fold)
                        model = pickle.load(open(MODEL_PATH + "inner/" + folder_names[i] + "/resnet_6_4Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(val_outer_fold) + 
                                          "_inner_fold_" + str(val_inner_fold), 'rb')) # load in the model
                        validate_model(model, val_outer_fold, val_inner_fold)

"""
test inner fold models on the corresponding test set
"""
if TEST_INNER_MODEL_FLAG == 1:
      print("Beginning Testing")
      folder_names = os.listdir(MODEL_PATH + "inner/")
      folder_names.sort()
      for working_folder in folder_names:
            # pass through all the outer folds
            print(int(working_folder))
            if int(working_folder) > 30:
                  for test_outer_fold in range(NUM_OUTER_FOLDS):
                        print("test_outer_fold=", test_outer_fold)
                        
                        # for each outer fold pass through all the inner folds
                        for test_inner_fold in range(NUM_INNER_FOLDS):
                              print("test_inner_fold=", test_inner_fold)
                              model = pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                                "_inner_fold_" + str(test_inner_fold) + "_final_model", 'rb')) # load in the model
                              test_model(model, test_outer_fold)
            


"""
Use the average of all inner fold model predictions to make predictions.
"""
if TEST_INNER_ENSEMBLE_MODELS_FLAG == 1:
      print("Beginning Testing")
      folder_names = os.listdir(MODEL_PATH + "inner/")
      folder_names.sort()
      for working_folder in folder_names:
            # pass through all the outer folds
            print(int(working_folder))
            if int(working_folder) > 30:
                  for test_outer_fold in range(NUM_OUTER_FOLDS):
                        print("test_outer_fold_ensemble=", test_outer_fold)
                        
                        # for each outer fold pass through all the inner folds
                        models = []
                        for test_inner_fold in range(NUM_INNER_FOLDS):
                              models.append(pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                                "_inner_fold_" + str(test_inner_fold) + "_final_model", 'rb'))) # load in the model
                        
                        test_models(models, test_outer_fold)


"""
test the performance of all outer_fold based models
"""
if TEST_OUTER_ONLY_MODEL_FLAG == 1:
      print("Beginning Testing")
      folder_names = os.listdir(MODEL_PATH + "inner/")
      folder_names.sort()
      for working_folder in folder_names:
            if int(working_folder) > 15:
            # pass through all the outer folds
                  for test_outer_fold in range(NUM_OUTER_FOLDS):
                        print("test_outer_fold=", test_outer_fold)
                        model = pickle.load(open(MODEL_PATH + "outer/" + working_folder + "/resnet_18_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                                "_inner_fold_" + str(None) + "_final_model", 'rb')) # load in the model
                        test_model(model, test_outer_fold)

"""
test the performance of all ensemble models
"""
if TEST_ENSEMBLE_MODEL_FLAG == 1:
      folder_names = os.listdir(MODEL_PATH + "ensemble/")
      folder_names.sort()
      for working_folder in folder_names:
            if int(working_folder) > 15:
                  print("Beginning Testing")
                  for test_outer_fold in range(NUM_OUTER_FOLDS):
                        print("test_outer_fold=", test_outer_fold)
                        model = pickle.load(open((MODEL_PATH + "ensemble/" + working_folder + "/resnet_6_4Deep_" + MODEL_MELSPEC + "_outer_fold_" 
                                                            + str(test_outer_fold)), 'rb')) # load in the model
                        test_model(model, test_outer_fold)  