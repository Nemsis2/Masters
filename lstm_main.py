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
from get_best_features import *


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

#select the number of features
NUM_FEATURES = 150

# set hyperpaperameters
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# training options for the models
TRAIN_INNER_MODEL_FLAG = 0
TRAIN_INNER_FSS_MODEL_FLAG = 0
TRAIN_OUTER_MODEL_FLAG = 0
TRAIN_ENSEMBLE_MODEL_FLAG = 0

# testing options for the models
TEST_GROUP_DECISION_FLAG = 1
TEST_OUTER_ONLY_MODEL_FLAG = 0
TEST_ENSEMBLE_MODEL_FLAG = 0
TEST_GROUP_FSS_DECISION_FLAG = 0
TEST_OUTER_FSS_ONLY_MODEL_FLAG = 0
VAL_MODEL_TEST_FLAG = 0
VAL_FSS_MODEL_TEST_FLAG = 0
TRAIN_OUTER_FSS_MODEL_FLAG = 0


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("Cuda not enabled. Exiting...")
      exit(1)

HIDDEN_LAYERS = [32, 64, 128]
LAYERS = [1, 2, 3]
BEST_HIDDEN_LAYERS = [128, 64, 32]
BEST_LAYERS = [3, 2, 2]


def get_oracle_thresholds(results, labels, threshold):
    sens_threshold, spec_threshold = np.zeros(len(threshold)), np.zeros(len(threshold))
    for i in range(len(threshold)):
        thresholded_results = (np.array(results)>threshold[i]).astype(np.int8)
        sens, spec = calculate_sens_spec(labels, thresholded_results)
        sens_threshold[i] = np.abs(sens-0.9)
        spec_threshold[i] = np.abs(spec-0.7)

    print(sens_threshold)
    sens = np.nanargmin(sens_threshold)
    spec = np.nanargmin(spec_threshold)
    print("sens", sens)
    return threshold[sens], threshold[spec]


"""
Create a bi_lstm model
"""
class bi_lstm(nn.Module):
    def __init__(self, input_dim, hidden_dim, layers):
        super(bi_lstm, self).__init__()
        self.hidden_dim = hidden_dim
        self.layers = layers        
        if layers < 1:
            self.bi_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, dropout=0.5, bidirectional=True)
        else:
            self.bi_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, bidirectional=True)

        self.drop1 = nn.Dropout(p=0.5)
        self.batchnorm = nn.BatchNorm1d(self.hidden_dim * 2)
        self.fc1 = nn.Linear(hidden_dim*2, 32)
        self.mish = nn.Mish()
        self.fc2 = nn.Linear(32,2)

    def forward(self, x, lengths):
        total_length = x.shape[1]
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        self.bi_lstm.flatten_parameters()
        out, (h_n, c_n) = self.bi_lstm(x)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)
        
        out_forward = out[range(len(out)), (np.array(lengths) - 1), :self.hidden_dim]
        out_reverse = out[range(len(out)), 0, self.hidden_dim:]
        out_reduced = th.cat((out_forward, out_reverse), dim=1)
        
        result = self.drop1(out_reduced)
        result = self.batchnorm(result)
        result = self.fc1(result)
        result = self.mish(result)
        result = self.fc2(result)
        return result 


"""
Create a bi_lstm package including
Model
Optimizer(RMSprop)
Model name
"""
class bi_lstm_package():
    def __init__(self, input_size, hidden_dim, layers, outer, inner, folder, epochs, batch_size, model_type):
        self.name = "bi_lstm_"
        self.seed = th.seed()
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_type = model_type
        self.outer = outer
        self.inner = inner
        self.folder = folder
        
        self.model = bi_lstm(input_size, hidden_dim, layers)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
        self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=180, steps_per_epoch=30)

    def train(self):
        if self.inner == None:
            data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, self.outer)
        else:
            #data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, self.outer, self.inner)
            data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, self.outer)

        data, labels, lengths = create_batches(data, labels, "linear", self.batch_size)
        
        # run through all the epochs
        for epoch in range(self.epochs):
            print("epoch=", epoch)
            train(data, labels, lengths, self)

        # collect the garbage
        del data, labels, lengths
        gc.collect()

    def train_on_select(self, num_features):
        if self.inner == None:
                data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, self.outer)
                features = dataset_fss(num_features)
        else:
                data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, self.outer)
                features = dataset_fss(num_features)

        data, labels, lengths = create_batches(data, labels, "linear", self.batch_size)

        for batch in range(len(data)):
            chosen_features = []
            for feature in features:
                chosen_features.append(np.asarray(data[batch][:,:,feature]))
            data[batch] = th.as_tensor(np.stack(chosen_features, -1))
        
        # run through all the epochs
        for epoch in range(self.epochs):
            print("epoch=", epoch)
            train(data, labels, lengths, self)

        # collect the garbage
        del data, labels, lengths
        gc.collect()

    def save(self):
        pickle.dump(self, open("../models/tb/bi_lstm/" + self.model_type + "/" + self.folder + "/" + self.name + "melspec_180_outer_fold_" + str(self.outer) + 
                    "_inner_fold_" + str(self.inner), 'wb')) # save the model
        
    def val_on_select(self, num_features):
        data, labels, names = extract_val_data(K_FOLD_PATH + MELSPEC, self.outer, self.inner)

        # preprocess data and get batches
        data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)
        features = inner_fss(self.inner, self.outer, num_features)

        for batch in range(len(data)):
            chosen_features = []
            for feature in features:
                chosen_features.append(np.asarray(data[batch][:,:,feature]))
            data[batch] = th.as_tensor(np.stack(chosen_features, -1))

        results = []
        for i in range(len(data)):
            with th.no_grad():
                results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

        results = np.vstack(results)
        labels = np.vstack(labels)

        unq,ids,count = np.unique(names,return_inverse=True,return_counts=True)
        out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,labels[:,0])/count))
        results = out[:,1]
        labels = out[:,2]

        fpr, tpr, threshold = roc_curve(labels, results, pos_label=1)
        threshold = threshold[np.nanargmin(np.absolute(([1 - tpr] - fpr)))]

        return results, labels, threshold

    def val(self):
        data, labels, names = extract_val_data(K_FOLD_PATH + MELSPEC, self.outer, self.inner)

        # preprocess data and get batches
        data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)

        results = []
        for i in range(len(data)):
            with th.no_grad():
                results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

        results = np.vstack(results)
        labels = np.vstack(labels)

        unq,ids,count = np.unique(names,return_inverse=True,return_counts=True)
        out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,labels[:,0])/count))
        results = out[:,1]
        labels = out[:,2]

        #fpr, tpr, threshold = roc_curve(labels, results, pos_label=1)
        #threshold = threshold[np.nanargmin(np.absolute(([1 - tpr] - fpr)))]
        
        fpr, tpr, thresholds = roc_curve(labels, results, pos_label=1)
        sens_threshold, spec_threshold = get_oracle_thresholds(results, labels, thresholds)
        threshold = sens_threshold

        return results, labels, threshold
    
    def test(self):
        # read in the test set
        test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", self.outer)

        # preprocess data and get batches
        test_data, test_labels, test_names, lengths = create_test_batches(test_data, test_labels, test_names, "linear", self.batch_size)

        results = []
        for i in range(len(test_data)):
            with th.no_grad():
                results.append(to_softmax((self.model((test_data[i]).to(device), lengths[i])).cpu()))

        results = np.vstack(results)
        test_labels = np.vstack(test_labels)

        unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
        out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,test_labels[:,0])/count))
        results = out[:,1]
        test_labels = out[:,2]
        
        return results, test_labels
    
    def test_on_select(self, num_features):
        # read in the test set
        test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", self.outer)

        # preprocess data and get batches
        test_data, test_labels, test_names, lengths = create_test_batches(test_data, test_labels, test_names, "linear", self.batch_size)
        features = dataset_fss(num_features)
        
        for batch in range(len(test_data)):
            chosen_features = []
            for feature in features:
                chosen_features.append(np.asarray(test_data[batch][:,:,feature]))
            test_data[batch] = th.tensor(np.stack(chosen_features, -1))

        results = []
        for i in range(len(test_data)):
            with th.no_grad():
                results.append(to_softmax((self.model((test_data[i]).to(device), lengths[i])).cpu()))

        results = np.vstack(results)
        test_labels = np.vstack(test_labels)

        unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
        out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,test_labels[:,0])/count))
        results = out[:,1]
        test_labels = out[:,2]
        
        return results, test_labels


"""
trains the ensemble model on a specific outer and inner fold
"""
def train_ensemble_model(outer, model, criterion_kl, working_folder):
    # get the train fold
    data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, outer)

    # grab model
    models = []
    for test_inner_fold in range(NUM_INNER_FOLDS):
        models.append(pickle.load(open(MODEL_PATH + "GD/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                "_inner_fold_" + str(test_inner_fold), 'rb'))) # load in the model
        
    data, labels, lengths = create_batches(data, labels, "linear", 128)
        
    ensemble_train(data, labels, model, models, criterion_kl, lengths) # train the model on the current batch

    del data, labels, lengths
    gc.collect()



#************************MAIN*********************#

def main():
    ######################## Train Functions ###############################
    
    
    """
    train a model for each inner fold within each outer fold resulting in inner_fold*outer_fold number of models.
    trains only on training data
    """
    if TRAIN_INNER_MODEL_FLAG == 1:
        #for hidden in HIDDEN_LAYERS:
            #for layer in LAYERS:
        working_folder = create_new_folder(str(MODEL_PATH + "GD/"))
        print(working_folder)
        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", inner)

                model = bi_lstm_package(180, BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, inner, working_folder, epochs=16, batch_size=128, model_type="GD")
                model.train()
                model.save()


    
    if TRAIN_INNER_FSS_MODEL_FLAG == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/"))
        print(working_folder)
        
        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)
            
            for inner in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", inner)

                model = bi_lstm_package(NUM_FEATURES, BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, inner, working_folder, epochs=16, batch_size=128, model_type=str("GD_" + str(NUM_FEATURES)))
                model.train_on_select(NUM_FEATURES)
                model.save()  
    
    
    """
    train a model for each outer_fold
    """
    if TRAIN_OUTER_MODEL_FLAG == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "OM/"))
        print(working_folder)
        
        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)

            model = bi_lstm_package(180, BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, None, working_folder, epochs=16, batch_size=128, model_type="OM")
            model.train()
            model.save()



    """
    train a model for each outer_fold
    """
    if TRAIN_OUTER_FSS_MODEL_FLAG == 1:
        working_folder = create_new_folder(str(MODEL_PATH + "OM_" + str(NUM_FEATURES) + "/"))
        print(working_folder)
        
        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)

            model = bi_lstm_package(NUM_FEATURES, BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, None, working_folder, epochs=16, batch_size=128, model_type=str("OM_" + str(NUM_FEATURES)))
            model.train_on_select(NUM_FEATURES)
            model.save()

    """
    trains an ensemble model using the inner models and the original data
    """
    if TRAIN_ENSEMBLE_MODEL_FLAG == 1:
        folder_names = os.listdir(MODEL_PATH + "GD/")
        folder_names.sort()
        for i in range(len(folder_names)):
            print("Beginning Training")
            working_folder = create_new_folder(str(MODEL_PATH + "EM/"))
                
            for outer in range(NUM_OUTER_FOLDS):
                print("train_outer_fold=", outer)
                lstm = bi_lstm_package(180, BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, None, working_folder, epochs=16, batch_size=128, model_type="EM")
                criterion_kl = nn.KLDivLoss()
                
                for epoch in range(16):
                    print("epoch=", epoch)

                    train_ensemble_model(outer, lstm, criterion_kl, folder_names[i])

                pickle.dump(lstm, open((MODEL_PATH + "EM/" + working_folder + "/" + lstm.name + MODEL_MELSPEC + "_outer_fold_" 
                                                    + str(outer)), 'wb')) # save the model






    #########train_on_select################# VAL FUNCTIONS ##############################
      
      
    """
    validates each model by assessing its performance on its corresponding validation set.
    """
    if VAL_MODEL_TEST_FLAG == 1:
        print("Beginning Validation")
        folder_names = os.listdir(MODEL_PATH + "val/")
        folder_names.sort()
        print(folder_names)
        best_fold0, best_fold1, best_fold2 = 0, 0, 0

        for folder in folder_names: # pass through all the outer folds
            auc = 0
            for outer in range(NUM_OUTER_FOLDS):
                print("val_outer_fold=", outer)

                for inner in range(NUM_INNER_FOLDS): # for each outer fold pass through all the inner folds
                    model = pickle.load(open(MODEL_PATH + "val/" + folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                    "_inner_fold_" + str(inner), 'rb')) # load in the model
                    th.manual_seed(model.seed) #set the seed to be the same as the one the model was generated on

                    results, test_labels, threshold = model.val()
                    auc += roc_auc_score(test_labels, results)

                if best_fold0 < auc/4 and outer == 0:
                    best_fold0 = auc/4
                    folder0 = folder
                
                if best_fold1 < auc/4 and outer == 1:
                    best_fold1 = auc/4
                    folder1 = folder

                if best_fold2 < auc/4 and outer == 2:
                    best_fold2 = auc/4
                    folder2 = folder

                auc = 0  

            print("Folder 0:", folder0, "AUC:", best_fold0)
            print("Folder 0:", folder1, "AUC:", best_fold1)
            print("Folder 0:", folder2, "AUC:", best_fold2)
            

    if VAL_FSS_MODEL_TEST_FLAG == 1:
        print("Beginning Validation")
        folder_names = os.listdir(MODEL_PATH + "val_150/")
        folder_names.sort()
        print(folder_names)
        best_fold0, best_fold1, best_fold2 = 0, 0, 0

        for folder in folder_names: # pass through all the outer folds
            auc = 0
            for outer in range(NUM_OUTER_FOLDS):
                print("val_outer_fold=", outer)

                for inner in range(NUM_INNER_FOLDS): # for each outer fold pass through all the inner folds
                    model = pickle.load(open(MODEL_PATH + "val_150/" + folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                    "_inner_fold_" + str(inner), 'rb')) # load in the model
                    th.manual_seed(model.seed) #set the seed to be the same as the one the model was generated on

                    results, test_labels, threshold = model.val_on_select(150)
                    auc += roc_auc_score(test_labels, results)

                if best_fold0 < auc/4 and outer == 0:
                    best_fold0 = auc/4
                    folder0 = folder
                
                if best_fold1 < auc/4 and outer == 1:
                    best_fold1 = auc/4
                    folder1 = folder

                if best_fold2 < auc/4 and outer == 2:
                    best_fold2 = auc/4
                    folder2 = folder

                auc = 0  

            print("Folder 0:", folder0, "AUC:", best_fold0)
            print("Folder 0:", folder1, "AUC:", best_fold1)
            print("Folder 0:", folder2, "AUC:", best_fold2)


    ########################## TEST FUNCTIONS ##############################

    
    """
    test the performance of all outer_fold based models
    """
    if TEST_OUTER_ONLY_MODEL_FLAG == 1:
        print("Beginning Outer Model Testing")
        folder_names = os.listdir(MODEL_PATH + "OM/")
        folder_names.sort()
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        threshold = [0.6090625404465824, 0.490797293445992, 0.5695457850370491]
        for working_folder in folder_names:
                # pass through all the outer folds
                for outer in range(NUM_OUTER_FOLDS):
                    print("test_outer_fold=", outer)
                    model = pickle.load(open(MODEL_PATH + "OM/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer)
                                                + "_inner_fold_None", 'rb')) # load in the model
                    th.manual_seed(model.seed)
                    result, label = model.test() # test the model

                    print(threshold)
                    auc[outer] = roc_auc_score(label, result)
                    results = (np.array(result)>threshold[outer]).astype(np.int8)
                    sens[outer], spec[outer] = calculate_sens_spec(label, results)

                print("AUC:", np.mean(auc), "var:", np.var(auc))
                print("sens:", np.mean(sens), "var:", np.var(sens))
                print("spec:", np.mean(spec), "var:", np.var(spec))



    """
    test the performance of all outer_fold based models
    """
    if TEST_OUTER_FSS_ONLY_MODEL_FLAG == 1:
        print("Beginning Outer Model Testing")
        folder_names = os.listdir(MODEL_PATH + "OM_" + str(NUM_FEATURES) + "/")
        folder_names.sort()
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        threshold = [0.6090625404465824, 0.490797293445992, 0.5695457850370491]
        for working_folder in folder_names:
                # pass through all the outer folds
                for outer in range(NUM_OUTER_FOLDS):
                    print("test_outer_fold=", outer)
                    model = pickle.load(open(MODEL_PATH + "OM_" + str(NUM_FEATURES) +"/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer)
                                                + "_inner_fold_None", 'rb')) # load in the model
                    th.manual_seed(model.seed)
                    result, label = model.test_on_select(NUM_FEATURES) # test the model

                    print(threshold)
                    auc[outer] = roc_auc_score(label, result)
                    results = (np.array(result)>threshold[outer]).astype(np.int8)
                    sens[outer], spec[outer] = calculate_sens_spec(label, results)

                print("AUC:", np.mean(auc), "var:", np.var(auc))
                print("sens:", np.mean(sens), "var:", np.var(sens))
                print("spec:", np.mean(spec), "var:", np.var(spec))
    """
    Use the average of all inner fold model predictions to make predictions.
    """
    if TEST_GROUP_DECISION_FLAG == 1:
        print("Beginning Group Decision Model Testing")
        folder_names = os.listdir(MODEL_PATH + "GD/")
        folder_names.sort()
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        for working_folder in folder_names:
            # pass through all the outer folds
            print(working_folder)
            
            for outer in range(NUM_OUTER_FOLDS):
                print("test_outer_fold_ensemble=", outer)
                
                # for each outer fold pass through all the inner folds
                results, thresholds = [], []
                for test_inner_fold in range(NUM_INNER_FOLDS):
                    model = pickle.load(open(MODEL_PATH + "threshold/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                        "_inner_fold_" + str(test_inner_fold), 'rb')) # load in the model
                    th.manual_seed(model.seed) # set the seed to be the same as the one the model was generated on
                    _, _, threshold = model.val() # get the threshold

                    model = pickle.load(open(MODEL_PATH + "GD/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                        "_inner_fold_" + str(test_inner_fold), 'rb')) # load in the model
                    th.manual_seed(model.seed) # set the seed to be the same as the one the model was generated on
                    result, label = model.test() # test the model
                    
                    results.append(result)
                    thresholds.append(threshold)

                results = np.mean(np.vstack(np.array(results)), axis=0)
                threshold = np.mean(thresholds)
                print("threshold:",threshold)
                auc[outer] = roc_auc_score(label, results)
                results = (np.array(results)>threshold).astype(np.int8)
                sens[outer], spec[outer] = calculate_sens_spec(label, results)

            print("AUC:", np.mean(auc), "var:", np.var(auc))
            print("sens:", np.mean(sens), "var:", np.var(sens))
            print("spec:", np.mean(spec), "var:", np.var(spec))
    

    """
    test the performance of all ensemble models
    """
    if TEST_ENSEMBLE_MODEL_FLAG == 1:
        print("Beginning Ensemble Model Testing")
        folder_names = os.listdir(MODEL_PATH + "EM/")
        folder_names.sort()
        threshold = [0.6090625404465824, 0.490797293445992, 0.5695457850370491]
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        for working_folder in folder_names:
                # pass through all the outer folds
                for outer in range(NUM_OUTER_FOLDS):
                    print("test_outer_fold=", outer)
                    model = pickle.load(open(MODEL_PATH + "EM/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer), 'rb')) # load in the model
                    th.manual_seed(model.seed)
                    result, label = model.test() # test the model
                    print(threshold)
                    auc[outer] = roc_auc_score(label, result)
                    results = (np.array(result)>threshold[outer]).astype(np.int8)
                    sens[outer], spec[outer] = calculate_sens_spec(label, results)

        print("AUC:", np.mean(auc), "var:", np.var(auc))
        print("sens:", np.mean(sens), "var:", np.var(sens))
        print("spec:", np.mean(spec), "var:", np.var(spec))


    if TEST_GROUP_FSS_DECISION_FLAG == 1:
        print("Beginning Group Decision Model Testing")
        folder_names = os.listdir(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/")
        folder_names.sort()
        auc, sens, spec = np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64), np.array([1,2,3], dtype=np.float64)
        for working_folder in folder_names:
            # pass through all the outer folds
            print(working_folder)
            
            for outer in range(NUM_OUTER_FOLDS):
                print("test_outer_fold_ensemble=", outer)
                
                # for each outer fold pass through all the inner folds
                results, thresholds = [], []
                for test_inner_fold in range(NUM_INNER_FOLDS):
                    model = pickle.load(open(MODEL_PATH + "GD_" + str(NUM_FEATURES) + "/" + working_folder + "/bi_lstm_" + MODEL_MELSPEC + "_outer_fold_" + str(outer) + 
                                        "_inner_fold_" + str(test_inner_fold), 'rb')) # load in the model
                    th.manual_seed(model.seed) # set the seed to be the same as the one the model was generated on
                    
                    result, label = model.test_on_select(NUM_FEATURES) # test the model
                    results.append(result)

                results = np.mean(np.vstack(np.array(results)), axis=0)
                auc[outer] = roc_auc_score(label, results)

            print("AUC:", np.mean(auc), "var:", np.var(auc))

if __name__ == "__main__":
    main()