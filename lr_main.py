# libraries
import torch as th
import torch.nn as nn
import torch.optim as optim
import os
import pickle

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from pruning import *
from model_scripts import *


"""
date: 21/02/2023 

author: Michael Knight
"""

# declare global variables

# set paths
K_FOLD_PATH = "../data/tb/combo/multi_folds/"
MODEL_PATH = "../models/tb/bi_lstm/"

# choose which melspec we will be working on
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"


# set hyperpaperameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# training options for the models
TRAIN_INNER_MODEL_FLAG = 1
TRAIN_OUTER_MODEL_FLAG = 0
TRAIN_ENSEMBLE_MODEL_FLAG = 0

# testing options for the models
TEST_INNER_MODEL_FLAG = 0
TEST_GROUP_DECISION_FLAG = 0
TEST_OUTER_ONLY_MODEL_FLAG = 0
TEST_ENSEMBLE_MODEL_FLAG = 0
VAL_MODEL_TEST_FLAG = 0


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)


#https://towardsdatascience.com/logistic-regression-with-pytorch-3c8bbea594be
class LogisticRegression(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)     
        
    def forward(self, x):
        outputs = th.sigmoid(self.linear(x))
        return outputs
    
    
def main():
    """
    train a model for each inner fold within each outer fold resulting in inner_fold*outer_fold number of models.
    trains only on training data.m
    """
    if TRAIN_INNER_MODEL_FLAG == 1:
        for i in range(15): #this will be used to determine how many models should be made
            working_folder = create_new_folder(str(MODEL_PATH + "inner/"))
            
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                    print("train_outer_fold=", train_outer_fold)
                    
                    for train_inner_fold in range(NUM_INNER_FOLDS):
                        print("train_inner_fold=", train_inner_fold)
                        model = LogisticRegression()
                        train_model(train_outer_fold, train_inner_fold, model, working_folder, NUM_EPOCHS, BATCH_SIZE, "linear", MODEL_PATH, final_model=1)  
     

if __name__ == "__main__":
    main()
