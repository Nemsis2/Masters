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
BATCH_SIZE = 128
NUM_EPOCHS = 60
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# training options for the models
TRAIN_INNER_MODEL_FLAG = 0
TRAIN_OUTER_MODEL_FLAG = 0
TRAIN_ENSEMBLE_MODEL_FLAG = 1

# testing options for the models
TEST_INNER_MODEL_FLAG = 0
TEST_GROUP_DECISION_FLAG = 0
TEST_OUTER_ONLY_MODEL_FLAG = 0
TEST_ENSEMBLE_MODEL_FLAG = 0
VAL_MODEL_TEST_FLAG = 0
VAL_MODEL_OUTER_TEST_FLAG = 0


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("exiting since cuda not enabled")
      exit(1)

HIDDEN_LAYERS = [32,64]
LAYERS = [1,2,3]

"""
Validation model:   1-15    -hidden_layers=32, num_layers=1
                    16-30   -hidden_layers=32, num_layers=2
                    31-45   -hidden_layers=32, num_layers=3
                    46-60   -hidden_layers=64, num_layers=1
                    61-75   -hidden_layers=64, num_layers=2
                    76-90   -hidden_layers=64, num_layers=3

"""




"""
Create a bi_lstm model
"""
class bi_lstm(nn.Module):
    def __init__(self, hidden_dim, layers):
        super(bi_lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.bi_lstm = nn.LSTM(input_size=180, hidden_size=hidden_dim, num_layers=layers, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(hidden_dim*2, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim,2)

    def forward(self, x):
        self.bi_lstm.flatten_parameters()
        out, (h_n, c_n) = self.bi_lstm(x)
        out_forward = out[range(len(out)), (x.shape[1] - 1), :self.hidden_dim]
        out_reverse = out[range(len(out)), 0, self.hidden_dim:]
        out_reduced = th.cat((out_forward, out_reverse), dim=1)
        result = self.drop(out_reduced)
        result = self.fc1(result)
        result = self.relu(result)
        result = self.fc2(result)
        return result



"""
Create a bi_lstm package including:
Model
Optimizer(Adam)
Model name
"""
class bi_lstm_package():
    def __init__(self, hidden_dim, layers):
        self.model = bi_lstm(hidden_dim, layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
        self.model_name = "bi_lstm_"
 


"""
trains the ensemble model on a specific outer and inner fold
"""
def train_ensemble_model(train_outer_fold, train_inner_fold, model, criterion_kl, epochs, working_folder, final_model=0):
    # get the train fold
    data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold, final_model)
    
    # grab model
    inner_model = pickle.load(open(MODEL_PATH + "inner/" + str(working_folder) + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(train_outer_fold) + 
                                        "_inner_fold_" + str(train_inner_fold) + "_final_model", 'rb')) # save the model

    print("batch=", data.shape)
    
    data, labels = create_batches(data, labels, "linear", BATCH_SIZE)
        
    ensemble_train(data, labels, model, inner_model, criterion_kl) # train the model on the current batch

    del data, labels
    gc.collect()


#************************MAIN*********************#

def main():
    ######################## Train Functions ###############################
    
    
    """
    train a model for each inner fold within each outer fold resulting in inner_fold*outer_fold number of models.
    trains only on training data.m
    """
    if TRAIN_INNER_MODEL_FLAG == 1:
        use_val = 1
        for i in range(15): #this will be used to determine how many models should be made
            if use_val == 0:
                working_folder = create_new_folder(str(MODEL_PATH + "val/"))
            else:
                working_folder = create_new_folder(str(MODEL_PATH + "inner/"))
            
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                    print("train_outer_fold=", train_outer_fold)
                    
                    for train_inner_fold in range(NUM_INNER_FOLDS):
                        print("train_inner_fold=", train_inner_fold)
                        #print(HIDDEN_LAYERS[hidden], LAYERS[layer])
                        lstm = bi_lstm_package(32,2)
                        train_model(train_outer_fold, train_inner_fold, lstm, working_folder, NUM_EPOCHS, BATCH_SIZE, "linear", MODEL_PATH, final_model=use_val)


    """
    train a model for each outer_fold
    can be set to either use both train and val data or just train data
    results in outer_fold number of models.
    """
    if TRAIN_OUTER_MODEL_FLAG == 1:
        use_val = 1
        for i in range(15): #this will be used to determine how many models should be made
            if use_val == 0:
                working_folder = create_new_folder(str(MODEL_PATH + "val/"))
            else:
                working_folder = create_new_folder(str(MODEL_PATH + "outer/"))
            
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                    print("train_outer_fold=", train_outer_fold)
                    lstm = bi_lstm_package(32, 2)
                    train_model(train_outer_fold, None, lstm, working_folder, NUM_EPOCHS, BATCH_SIZE, "linear", MODEL_PATH, final_model=use_val)


    """
    trains an ensemble model using the inner models and the original data
    """
    if TRAIN_ENSEMBLE_MODEL_FLAG == 1:
        use_val = 1
        for i in range(15):
            print("Beginning Training")
            if use_val == 0:
                working_folder = create_new_folder(str(MODEL_PATH + "val/"))
            else:
                working_folder = create_new_folder(str(MODEL_PATH + "ensemble/"))
                
            for train_outer_fold in range(NUM_OUTER_FOLDS):
                print("train_outer_fold=", train_outer_fold)
                lstm = bi_lstm_package(32,2)
                criterion_kl = nn.KLDivLoss()
                
                for epoch in range(NUM_EPOCHS):
                    for train_inner_fold in range(NUM_INNER_FOLDS):
                        print("train_inner_fold=", train_inner_fold)
                        train_ensemble_model(train_outer_fold, train_inner_fold, lstm, criterion_kl, NUM_EPOCHS, working_folder,final_model=use_val)

                    pickle.dump(lstm.model, open((MODEL_PATH + "ensemble/" + working_folder + "/" + lstm.model_name + MODEL_MELSPEC + "_outer_fold_" 
                                                    + str(train_outer_fold)), 'wb')) # save the model

   
   
    ########################## VAL FUNCTIONS ##############################
      
      
    """
    used for inner fold based models only.
    validates each model by assessing its performance on its corresponding validation set.
    """
    if VAL_MODEL_TEST_FLAG == 1:
        print("Beginning Validation")
        folder_names = os.listdir(MODEL_PATH + "val/")
        folder_names.sort()
        print(folder_names)

        average_auc_0, average_auc_1, average_auc_2 = 0,0,0
        average_sens_0, average_sens_1, average_sens_2 = 0,0,0
        average_spec_0, average_spec_1, average_spec_2 = 0,0,0
        count = 0
        auc_fold_0, auc_fold_1, auc_fold_2 = 0,0,0

        for i in range(len(folder_names)):
                # pass through all the outer folds
                for val_outer_fold in range(NUM_OUTER_FOLDS):
                    #print("val_outer_fold=", val_outer_fold)
                    
                    # for each outer fold pass through all the inner folds
                    for val_inner_fold in range(NUM_INNER_FOLDS):
                            #print("val_inner_fold=", val_inner_fold)
                            model = pickle.load(open(MODEL_PATH + "val/" + folder_names[i] + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(val_outer_fold) + 
                                            "_inner_fold_" + str(val_inner_fold) + "_epochs_60", 'rb')) # load in the model
                            auc, sens, spec = validate_model(model, val_outer_fold, val_inner_fold, "linear", BATCH_SIZE)
                            
                            if val_outer_fold == 0:
                                average_auc_0 += auc
                                average_sens_0 += sens
                                average_spec_0 += spec
                            elif(val_outer_fold == 1):
                                average_auc_1 += auc
                                average_sens_1 += sens
                                average_spec_1 += spec
                            elif(val_outer_fold == 2):
                                average_auc_2 += auc
                                average_sens_2 += sens
                                average_spec_2 += spec



                count +=1
                if count % 15 == 0:
                    print(count)
                    if auc_fold_0 < average_auc_0/(15*4):
                        auc_fold_0 = average_auc_0/(15*4)

                    if auc_fold_1 < average_auc_1/(15*4):
                        auc_fold_1 = average_auc_1/(15*4)

                    if auc_fold_2 < average_auc_2/(15*4):
                        auc_fold_2 = average_auc_2/(15*4)
                    print("Average auc fold 0:", average_auc_0/(15*4))
                    #print("Average sens fold 0:", average_sens_0/(15*4))
                    #print("Average spec fold 0:", average_spec_0/(15*4))
                    print("Average auc fold 1:", average_auc_1/(15*4))
                    #print("Average sens fold 1:", average_sens_1/(15*4))
                    #print("Average spec fold 1:", average_spec_1/(15*4))
                    print("Average auc fold 2:", average_auc_2/(15*4))
                    #print("Average sens fold 2:", average_sens_2/(15*4))
                    #print("Average spec fold 2:", average_spec_2/(15*4))

                    print("        ")

                    average_auc_0, average_auc_1, average_auc_2 = 0,0,0
                    average_sens_0, average_sens_1, average_sens_2 = 0,0,0
                    average_spec_0, average_spec_1, average_spec_2 = 0,0,0

        print("Max for fold 0", auc_fold_0)
        print("Max for fold 1", auc_fold_1)
        print("Max for fold 2", auc_fold_2)



    """
    used for inner fold based models only.
    validates each model by assessing its performance on its corresponding validation set.
    """
    if VAL_MODEL_OUTER_TEST_FLAG == 1:
        print("Beginning Validation")
        folder_names = os.listdir(MODEL_PATH + "val/")
        folder_names.sort()
        print(folder_names)

        average_auc_0, average_auc_1, average_auc_2 = 0,0,0
        average_sens_0, average_sens_1, average_sens_2 = 0,0,0
        average_spec_0, average_spec_1, average_spec_2 = 0,0,0
        count = 0

        for i in range(len(folder_names)):
                # pass through all the outer folds
                for val_outer_fold in range(NUM_OUTER_FOLDS):
                    #print("val_outer_fold=", val_outer_fold)
                    model = pickle.load(open(MODEL_PATH + "val/" + folder_names[i] + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(val_outer_fold) + 
                                    "_inner_fold_" + str(None) + "_epochs_15", 'rb')) # load in the model
                    auc, sens, spec = validate_model(model, val_outer_fold, None, "linear", BATCH_SIZE)
                    
                    if val_outer_fold == 0:
                        average_auc_0 += auc
                        average_sens_0 += sens
                        average_spec_0 += spec
                    elif(val_outer_fold == 1):
                        average_auc_1 += auc
                        average_sens_1 += sens
                        average_spec_1 += spec
                    elif(val_outer_fold == 2):
                        average_auc_2 += auc
                        average_sens_2 += sens
                        average_spec_2 += spec



                count +=1
                if count % 15 == 0:
                    print(count)
                    auc_fold_0, auc_fold_1, auc_fold_2 = 0,0,0
                    if auc_fold_0 < average_auc_0/(15):
                        auc_fold_0 = average_auc_0/(15)

                    if auc_fold_1 < average_auc_1/(15):
                        auc_fold_1 = average_auc_1/(15)

                    if auc_fold_2 < average_auc_2/(15):
                        auc_fold_2 = average_auc_2/(15)
                    print("Average auc fold 0:", average_auc_0/(15))
                    #print("Average sens fold 0:", average_sens_0/(15*4))
                    #print("Average spec fold 0:", average_spec_0/(15*4))
                    print("Average auc fold 1:", average_auc_1/(15))
                    #print("Average sens fold 1:", average_sens_1/(15*4))
                    #print("Average spec fold 1:", average_spec_1/(15*4))
                    print("Average auc fold 2:", average_auc_2/(15))
                    #print("Average sens fold 2:", average_sens_2/(15*4))
                    #print("Average spec fold 2:", average_spec_2/(15*4))

                    print("        ")

                    average_auc_0, average_auc_1, average_auc_2 = 0,0,0
                    average_sens_0, average_sens_1, average_sens_2 = 0,0,0
                    average_spec_0, average_spec_1, average_spec_2 = 0,0,0

        print("Max for fold 0", auc_fold_0)
        print("Max for fold 1", auc_fold_1)
        print("Max for fold 2", auc_fold_2)



    ########################## TEST FUNCTIONS ##############################


    """
    test inner fold models on the corresponding test set
    """
    if TEST_INNER_MODEL_FLAG == 1:
        print("Beginning Testing")
        folder_names = os.listdir(MODEL_PATH + "inner/")
        folder_names.sort()
        for working_folder in folder_names:
            # pass through all the outer folds
            for test_outer_fold in range(NUM_OUTER_FOLDS):
                    print("test_outer_fold=", test_outer_fold)
                    
                    # for each outer fold pass through all the inner folds
                    for test_inner_fold in range(NUM_INNER_FOLDS):
                        print("test_inner_fold=", test_inner_fold)
                        model = pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                            "_inner_fold_" + str(test_inner_fold) + "_final_model", 'rb')) # load in the model
                        test_model(model, test_outer_fold, "linear", BATCH_SIZE)


    """
    test the performance of all outer_fold based models
    """
    if TEST_OUTER_ONLY_MODEL_FLAG == 1:
        print("Beginning Testing")
        folder_names = os.listdir(MODEL_PATH + "outer/")
        folder_names.sort()
        for working_folder in folder_names:
            # pass through all the outer folds
            for test_outer_fold in range(NUM_OUTER_FOLDS):
                print("test_outer_fold=", test_outer_fold)
                model = pickle.load(open(MODEL_PATH + "outer/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold)
                                            + "_inner_fold_None"    + "_final_model", 'rb')) # load in the model
                test_model(model, test_outer_fold, "linear", BATCH_SIZE)


    """
    Use the average of all inner fold model predictions to make predictions.
    """
    if TEST_GROUP_DECISION_FLAG == 1:
        print("Beginning Testing")
        folder_names = os.listdir(MODEL_PATH + "inner/")
        folder_names.sort()
        for working_folder in folder_names:
                # pass through all the outer folds
                print(int(working_folder))
                for test_outer_fold in range(NUM_OUTER_FOLDS):
                        print("test_outer_fold_ensemble=", test_outer_fold)
                        
                        # for each outer fold pass through all the inner folds
                        models = []
                        for test_inner_fold in range(NUM_INNER_FOLDS):
                            models.append(pickle.load(open(MODEL_PATH + "inner/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(test_outer_fold) + 
                                                "_inner_fold_" + str(test_inner_fold) + "_final_model", 'rb'))) # load in the model
                        
                        test_models(models, test_outer_fold, "linear", BATCH_SIZE)



if __name__ == "__main__":
    main()
