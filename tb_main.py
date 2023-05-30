# libraries
import torch as th
import torch.nn as nn
import torch.optim as optim
import gc
import os
import pickle


# custom scripts
from resnet import *
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from pruning import *
from model_scripts import *


"""
date: 21/02/2023 

author: Michael Knight
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
# 128 and 80 both seem to be worse more or less across the board
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"

# set hyperpaperameters
BATCH_SIZE = 128
NUM_EPOCHS = 15
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
PRUNING_PERCENTAGE = 0.3

# training options for the models
TRAIN_INNER_MODEL_FLAG = 0
TRAIN_OUTER_MODEL_FLAG = 0
TRAIN_ENSEMBLE_MODEL_FLAG = 0
PRUNE_MODEL_FLAG = 0

# testing options for the models
TEST_INNER_MODEL_FLAG = 0
TEST_INNER_ENSEMBLE_MODELS_FLAG = 0
TEST_OUTER_ONLY_MODEL_FLAG = 0
TEST_ENSEMBLE_MODEL_FLAG = 0
VAL_MODEL_TEST_FLAG = 0


class Resnet18():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [2, 2 ,2 ,2], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_18_"


class Resnet10():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [1, 1 ,1 ,1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_10_"


class Resnet6_4Deep():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock1, [1, 1, 1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_6_4Deep_"


class Resnet6_2Deep():
      def __init__(self):
            self.model = ResNet_2layer(ResidualBlock2, [1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.model_name = "resnet_6_2Deep_"


"""
trains the ensemble model on a specific outer and inner fold
"""
def train_ensemble_model(train_outer_fold, train_inner_fold, model, criterion_kl, epochs, current_model_num, final_model=0):
      # get the train fold
      data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold, final_model)
      
      # access the previously generated models.
      # this will not generalise well as it simply grabs folders for a range and does not check if they will work
      if model.model_name == "resnet_6_4Deep_":
            working_folder = f'{current_model_num + 1:04d}'
      elif model.model_name == "resnet_18_":
            working_folder = f'{current_model_num + 16:04d}'
      elif model.model_name == "resnet_6_2Deep_":
            working_folder = f'{current_model_num + 31:04d}'
      else:
            print("Working folder incorrectly set. Failing...")
            working_folder = 0

      # grab model
      if final_model == 1:
            inner_model = pickle.load(open(MODEL_PATH + "inner/" + str(working_folder) + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                           "_inner_fold_" + str(train_inner_fold) + "_final_model", 'rb')) # save the model
      else:
            inner_model = pickle.load(open(MODEL_PATH + "inner/" + str(working_folder) + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) +
                                           "_inner_fold_" + str(train_inner_fold) + "_epochs_" + epochs, 'rb')) # save the model

      print("batch=", data.shape)
      
      data, labels = create_batches(data, labels, "image", BATCH_SIZE)
            
      ensemble_train(data, labels, model, inner_model, criterion_kl) # train the model on the current batch

      del data, labels
      gc.collect()



#######################################################
#                                                     #
#                                                     #
#                 MAIN FUNCTION                       #
#                                                     #
#                                                     #
#######################################################

def main():
      ############################ Train Functions #############################
      

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
                              train_model(train_outer_fold, train_inner_fold, model, working_folder, NUM_EPOCHS, BATCH_SIZE, "image", MODEL_PATH, final_model=1)        


      """
      train a model for each outer_fold
      can be set to either use both train and val data or just train data
      results in outer_fold number of models.
      """
      if TRAIN_OUTER_MODEL_FLAG == 1:
            for i in range(NUM_EPOCHS):
                  working_folder = create_new_folder(str(MODEL_PATH + "outer/"))
                  for train_outer_fold in range(NUM_OUTER_FOLDS):
                        print("train_outer_fold=", train_outer_fold)
                        model = Resnet6_2Deep()
                        train_model(train_outer_fold, None, model, working_folder, NUM_EPOCHS, BATCH_SIZE, "image", MODEL_PATH, final_model=1)        


      """
      trains an ensemble model using the inner models and the original data
      """
      if TRAIN_ENSEMBLE_MODEL_FLAG == 1:
            for i in range(NUM_EPOCHS):
                  print("Beginning Training")
                  working_folder = create_new_folder(str(MODEL_PATH + "ensemble/"))
                  
                  for train_outer_fold in range(NUM_OUTER_FOLDS):
                        print("train_outer_fold=", train_outer_fold)
                        model = Resnet6_2Deep()
                        criterion_kl = nn.KLDivLoss()
                        for epoch in range(NUM_EPOCHS):
                              for train_inner_fold in range(NUM_INNER_FOLDS):
                                    print("train_inner_fold=", train_inner_fold)
                                    train_ensemble_model(train_outer_fold, train_inner_fold, model, criterion_kl, NUM_EPOCHS, i,final_model=1)

                        pickle.dump(model.model, open((MODEL_PATH + "ensemble/" + working_folder + "/" + model.model_name + MODEL_MELSPEC + "_outer_fold_" 
                                                      + str(train_outer_fold)), 'wb')) # save the model







      ############################ Prune Functions #############################                  


      # wont work rn
      if PRUNE_MODEL_FLAG == 1:
            print("Beginning Pruning")
            
            for prune_fold in range(NUM_OUTER_FOLDS):
                  print("prune_fold=", prune_fold)
                  model = pickle.load(open((MODEL_PATH + "resnet18_" + MELSPEC + str(prune_fold)), 'rb')) # load in the model
                  
                  pruned_model, percentage_actually_pruned = prune_model(model, PRUNING_PERCENTAGE)

                  test_models(pruned_model, prune_fold, percentage_actually_pruned)

                  pickle.dump(pruned_model, open(MODEL_PATH + "resnet_" + MELSPEC + str(prune_fold) + "_pruned", 'wb')) # save the model







      ############################ Test Functions #############################


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
                              validate_model(model, val_outer_fold, val_inner_fold, "image", BATCH_SIZE)


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
                  if int(working_folder) == 47:
                        for test_outer_fold in range(NUM_OUTER_FOLDS):
                              print("test_outer_fold=", test_outer_fold)
                              
                              # for each outer fold pass through all the inner folds
                              for test_inner_fold in range(NUM_INNER_FOLDS):
                                    print("test_inner_fold=", test_inner_fold)
                                    model = pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                                      "_inner_fold_" + str(test_inner_fold) + "_final_model", 'rb')) # load in the model
                                    test_model(model, test_outer_fold, "image", BATCH_SIZE)
                  

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
                              
                              test_models(models, test_outer_fold, "image", BATCH_SIZE)


      """
      test the performance of all outer_fold based models
      """
      if TEST_OUTER_ONLY_MODEL_FLAG == 1:
            print("Beginning Testing")
            folder_names = os.listdir(MODEL_PATH + "inner/")
            folder_names.sort()
            for working_folder in folder_names:
                  if int(working_folder) > 30:
                  # pass through all the outer folds
                        for test_outer_fold in range(NUM_OUTER_FOLDS):
                              print("test_outer_fold=", test_outer_fold)
                              model = pickle.load(open(MODEL_PATH + "outer/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                                      "_inner_fold_" + str(None) + "_final_model", 'rb')) # load in the model
                              test_model(model, test_outer_fold, "image", BATCH_SIZE)


      """
      test the performance of all ensemble models
      """
      if TEST_ENSEMBLE_MODEL_FLAG == 1:
            folder_names = os.listdir(MODEL_PATH + "ensemble/")
            folder_names.sort()
            for working_folder in folder_names:
                  if int(working_folder) > 60:
                        print("Beginning Testing")
                        for test_outer_fold in range(NUM_OUTER_FOLDS):
                              print("test_outer_fold=", test_outer_fold)
                              model = pickle.load(open((MODEL_PATH + "ensemble/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" 
                                                                  + str(test_outer_fold)), 'rb')) # load in the model
                              test_model(model, test_outer_fold, "image", BATCH_SIZE)  



if __name__ == "__main__":
    main()
