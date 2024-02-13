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
MODEL_PATH = "../models/tb/resnet6_4/"

MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"

# set hyperpaperameters
BATCH_SIZE = 128
NUM_EPOCHS = 16
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
NUM_FEATURES = 150

# training options for the models
TRAIN_INNER_MODEL_FLAG = 0
TRAIN_INNER_FSS_MODEL_FLAG = 1
TRAIN_OUTER_MODEL_FLAG = 0
TRAIN_ENSEMBLE_MODEL_FLAG = 0

# testing options for the models
TEST_GROUP_DECISION_FLAG = 0
TEST_GROUP_FSS__DECISION_FLAG = 1
TEST_OUTER_ONLY_MODEL_FLAG = 0
TEST_ENSEMBLE_MODEL_FLAG = 0
VAL_MODEL_TEST_FLAG = 0



""""
Resnet18: 4, 1, 1

Resnet10: 3, 3, 4

Resnet6_4: 5, 2, 4

Resnet6_2: 3, 4, 5
"""

"""
Steps taken:
create 5 validation models

take top performing for each outer as the GD models
Use thresholds as calculated on val set

Rename the folders to always be 1,2,3 (if the same model is used multiple times only a single folder is created)

Train the same number of ensemble models as folders in the GD folder.

Train 5 OM models
"""


class Resnet18():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [2, 2 ,2 ,2], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
            self.name = "resnet_18_"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=30)


class Resnet10():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [1, 1 ,1 ,1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_10_"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=30)


class Resnet6_4Deep():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock1, [1, 1, 1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_6_4Deep_"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=30)


class Resnet6_2Deep():
      def __init__(self):
            self.model = ResNet_2layer(ResidualBlock2, [1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_6_2Deep_"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=30)


"""
trains the ensemble model on a specific outer and inner fold
"""
def train_ensemble_model(train_outer_fold, model, criterion_kl, working_folder):
      # get the train fold
      data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold)

      # grab model
      models = []
      for test_inner_fold in range(NUM_INNER_FOLDS):
            models.append(pickle.load(open(MODEL_PATH + "GD/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(test_inner_fold), 'rb'))) # load in the model
            
      data, labels, lengths = create_batches(data, labels, "image", BATCH_SIZE)
            
      ensemble_train(data, labels, model, models, criterion_kl, lengths) # train the model on the current batch

      del data, labels, lengths
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
      trains only on training data
      """
      if TRAIN_INNER_MODEL_FLAG == 1:
            for i in range(5):
                  working_folder = create_new_folder(str(MODEL_PATH + "val/"))
                  for train_outer_fold in range(NUM_OUTER_FOLDS):
                        print("train_outer_fold=", train_outer_fold)
                        for train_inner_fold in range(NUM_INNER_FOLDS):
                              print("train_inner_fold=", train_inner_fold)
                              model = Resnet18()
                              train_model(train_outer_fold, train_inner_fold, model, working_folder, NUM_EPOCHS, BATCH_SIZE, "image", MODEL_PATH + "val/")  

      
      if TRAIN_INNER_FSS_MODEL_FLAG == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/"))
        print(working_folder)
        
        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", inner)

                model = Resnet6_4Deep()
                train_model_on_features(outer, inner, model, working_folder, NUM_EPOCHS, BATCH_SIZE, "image", NUM_FEATURES, MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/")  


      """
      train a model for each outer_fold
      can be set to either use both train and val data or just train data
      results in outer_fold number of models.
      """
      if TRAIN_OUTER_MODEL_FLAG == 1:
            for i in range(5):
                  working_folder = create_new_folder(str(MODEL_PATH + "OM/"))
                  for train_outer_fold in range(NUM_OUTER_FOLDS):
                        print("train_outer_fold=", train_outer_fold)
                        model = Resnet18()
                        train_model(train_outer_fold, None, model, working_folder, NUM_EPOCHS, BATCH_SIZE, "image", MODEL_PATH + "OM/")        


      """
      trains an ensemble model using the inner models and the original data
      """
      if TRAIN_ENSEMBLE_MODEL_FLAG == 1:
            folder_names = os.listdir(MODEL_PATH + "GD/")
            folder_names.sort()
            for i in range(len(folder_names)):
                  print("Beginning Training")
                  working_folder = create_new_folder(str(MODEL_PATH + "EM/"))
                  
                  for train_outer_fold in range(NUM_OUTER_FOLDS):
                        print("train_outer_fold=", train_outer_fold)
                        model = Resnet6_2Deep()
                        criterion_kl = nn.KLDivLoss()
                        
                        for epoch in range(NUM_EPOCHS):
                              
                              print("epoch=", epoch)

                              train_ensemble_model(train_outer_fold, model, criterion_kl, folder_names[i])

                        pickle.dump(model, open((MODEL_PATH + "EM/" + working_folder + "/" + model.name + MODEL_MELSPEC + "_outer_fold_" 
                                    + str(train_outer_fold)), 'wb')) # save the model





      ############################ Test Functions #############################


      """
      used for inner fold based models only.
      validates each model by assessing its performance on its corresponding validation set.
      """
      if VAL_MODEL_TEST_FLAG == 1:
            print("Beginning Validation for ", MODEL_PATH) 
            folder_names = os.listdir(MODEL_PATH + "val/")
            folder_names.sort()
            print(folder_names)
            best_fold0, best_fold1, best_fold2 = 0, 0, 0

            for i in range(len(folder_names)):
                  # pass through all the outer folds
                  for val_outer_fold in range(NUM_OUTER_FOLDS):
                        total_auc = 0
                        thresholds = []
                        print("val_outer_fold=", val_outer_fold)
                        
                        # for each outer fold pass through all the inner folds
                        for val_inner_fold in range(NUM_INNER_FOLDS):
                              model = pickle.load(open(MODEL_PATH + "val/" + folder_names[i] + "/resnet_6_4Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(val_outer_fold) + 
                                                "_inner_fold_" + str(val_inner_fold), 'rb')) # load in the model
                              auc, sens, spec, threshold = validate_model_patients(model.model, val_outer_fold, val_inner_fold, "image", BATCH_SIZE)
                              total_auc += auc
                              thresholds.append(threshold)
                        
                        threshold = np.mean(thresholds)
                        print("AUC:", total_auc/4)
                        print("Sens:", sens)
                        print("Spec:", spec)
                        print("threshold:", threshold)

                        if best_fold0 < total_auc/4 and val_outer_fold == 0:  
                              best_fold0 = total_auc/4
                              folder0 = folder_names[i]
                        
                        if best_fold1 < total_auc/4 and val_outer_fold == 1:
                              best_fold1 = total_auc/4
                              folder1 = folder_names[i]

                        if best_fold2 < total_auc/4 and val_outer_fold == 2:
                              best_fold2 = total_auc/4
                              folder2 = folder_names[i]
                        

                  print("Folder 0:", folder0,  "AUC:", best_fold0)
                  print("Folder 0:", folder1,  "AUC:", best_fold1)
                  print("Folder 0:", folder2,  "AUC:", best_fold2)
                  

      """
      Use the average of all inner fold model predictions to make predictions.
      """
      if TEST_GROUP_DECISION_FLAG == 1:
            print("Beginning Testing Group Decision")
            folder_names = os.listdir(MODEL_PATH + "GD/")
            folder_names.sort()
            #resnet_6_2Deep thresholds = [0.6148279251323806, 0.5092164112751074, 0.46910479619566886]
            #resnet_6_4Deep thresholds = [0.6957424222451528, 0.618322073306822,0.43312809038823064]
            #resnet10 thresholds = [0.5691165582675124, 0.617273574978484, 0.5697435768196426]
            #resnet18 thresholds = [0.5535756670131251, 0.5219284197470794, 0.30767399159036685]
            auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
            for working_folder in folder_names:
                  # pass through all the outer folds
                  print(int(working_folder))
                  for outer in range(NUM_OUTER_FOLDS):
                        print("test group decision=", outer)
                        
                        # for each outer fold pass through all the inner folds
                        models = []
                        for test_inner_fold in range(NUM_INNER_FOLDS):
                              models.append(pickle.load(open(MODEL_PATH + "GD/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                          "_inner_fold_" + str(test_inner_fold), 'rb'))) # load in the model

                        auc[outer], sens[outer], spec[outer] = test_models_patients(models, outer, "image", BATCH_SIZE, thresholds[outer])

                  print("AUC:", np.mean(auc), "var:", np.var(auc))
                  print("sens:", np.mean(sens), "var:", np.var(sens))
                  print("spec:", np.mean(spec), "var:", np.var(spec))

      
      """
      Use the average of all inner fold model predictions to make predictions.
      """
      if TEST_GROUP_FSS__DECISION_FLAG == 1:
            print("Beginning Testing Group Decision")
            folder_names = os.listdir(MODEL_PATH +"GD_" + str(NUM_FEATURES) + "/")
            folder_names.sort()
            #resnet_6_2Deep thresholds = [0.6148279251323806, 0.5092164112751074, 0.46910479619566886]
            #resnet_6_4Deep thresholds = [0.6957424222451528, 0.618322073306822,0.43312809038823064]
            thresholds = [0.5691165582675124, 0.617273574978484, 0.5697435768196426]
            #resnet18 thresholds = [0.5535756670131251, 0.5219284197470794, 0.30767399159036685]
            auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
            for working_folder in folder_names:
                  # pass through all the outer folds
                  print(int(working_folder))
                  for outer in range(NUM_OUTER_FOLDS):
                        print("test group decision=", outer)
                        
                        # for each outer fold pass through all the inner folds
                        models = []
                        for test_inner_fold in range(NUM_INNER_FOLDS):
                              models.append(pickle.load(open(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/" + working_folder + "/resnet_6_4Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                          "_inner_fold_" + str(test_inner_fold), 'rb'))) # load in the model

                        auc[outer], sens[outer], spec[outer] = test_models_patients_on_select(models, NUM_FEATURES, outer, "image", BATCH_SIZE, thresholds[outer])

                  print("AUC:", np.mean(auc), "var:", np.var(auc))
                  print("sens:", np.mean(sens), "var:", np.var(sens))
                  print("spec:", np.mean(spec), "var:", np.var(spec))


      """
      test the performance of all outer_fold based models
      """
      if TEST_OUTER_ONLY_MODEL_FLAG == 1:
            print("Beginning Testing Outer Model")
            folder_names = os.listdir(MODEL_PATH + "OM/")
            folder_names.sort()
            #resnet_6_2Deep thresholds = [0.6148279251323806, 0.5092164112751074, 0.46910479619566886]
            #resnet_6_4Deep thresholds = [0.6957424222451528, 0.618322073306822,0.43312809038823064]
            #resnet10 thresholds = [0.5691165582675124, 0.617273574978484, 0.5697435768196426]
            #resnet18 thresholds = [0.5535756670131251, 0.5219284197470794, 0.30767399159036685]
            auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
            total_auc, total_sens, total_spec = 0,0,0
            var_auc, var_sens, var_spec = 0,0,0
            for working_folder in folder_names:
                  # pass through all the outer folds
                  for outer in range(NUM_OUTER_FOLDS):
                        print("test_outer_fold=", outer)
                        model = pickle.load(open(MODEL_PATH + "OM/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                                "_inner_fold_" + str(None), 'rb')) # load in the model
                        auc[outer], sens[outer], spec[outer] = test_patients(model.model, outer, "image", BATCH_SIZE, thresholds[outer])
                  total_auc += np.mean(auc)
                  total_sens += np.mean(sens)
                  total_spec += np.mean(spec)
                  var_auc += np.var(auc)
                  var_sens += np.var(sens)
                  var_spec += np.var(spec)
            
            print("AUC:", total_auc/5, "var:", var_auc/5)
            print("sens:", total_sens/5, "var:", var_sens/5)
            print("spec:", total_spec/5, "var:", var_spec/5)


      """
      test the performance of all ensemble models
      """
      if TEST_ENSEMBLE_MODEL_FLAG == 1:
            folder_names = os.listdir(MODEL_PATH + "EM/")
            folder_names.sort()
            #resnet_6_2Deep thresholds = [0.6148279251323806, 0.5092164112751074, 0.46910479619566886]
            #resnet_6_4Deep thresholds = [0.6957424222451528, 0.618322073306822,0.43312809038823064]
            #resnet10 thresholds = [0.5691165582675124, 0.617273574978484, 0.5697435768196426]
            #resnet18 thresholds = [0.5535756670131251, 0.5219284197470794, 0.30767399159036685]
            auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
            for working_folder in folder_names:
                  print("Beginning Testing Ensemble Model")
                  for outer in range(NUM_OUTER_FOLDS):
                        print("test_ensemble_fold=", outer)
                        model = pickle.load(open((MODEL_PATH + "EM/" + working_folder + "/resnet_6_2Deep_" + MODEL_MELSPEC + "_outer_fold_" 
                                                            + str(outer)), 'rb')) # load in the model
                        auc[outer], sens[outer], spec[outer] = test_patients(model.model, outer, "image", BATCH_SIZE, thresholds[outer])             
                        
                  print("AUC:", np.mean(auc), "var:", np.var(auc))
                  print("sens:", np.mean(sens), "var:", np.var(sens))
                  print("spec:", np.mean(spec), "var:", np.var(spec))



if __name__ == "__main__":
    main()
