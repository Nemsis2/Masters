# libraries
import torch as th
import os
import pickle
from sklearn.metrics import roc_auc_score

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from pruning import *
from model_scripts import *
from lstm_main import bi_lstm, bi_lstm_package

"""
date: 10/10/2023 

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
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
      print("Cuda not enabled. Exiting...")
      exit(1)

BEST_HIDDEN_LAYERS = [128, 64, 32]
BEST_LAYERS = [3, 2, 2]


def val_fss(data, labels, lengths, names, model):
    results = []
    for i in range(len(data)):
        with th.no_grad():
            results.append(to_softmax((model.model((data[i]).to(device), lengths[i])).cpu()))

    results = np.vstack(results)
    labels = np.vstack(labels)

    unq,ids,count = np.unique(names,return_inverse=True,return_counts=True)
    out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,labels[:,0])/count))
    results = out[:,1]
    labels = out[:,2]

    fpr, tpr, threshold = roc_curve(labels, results, pos_label=1)
    threshold = threshold[np.nanargmin(np.absolute(([1 - tpr] - fpr)))]

    return results, labels, threshold


def main():
    for number_of_models in range(4):
        working_folder = create_new_folder(str(MODEL_PATH + "FSS/"))
        print(working_folder)
        for outer in range(NUM_OUTER_FOLDS):
            print("train_outer_fold=", outer)
            for inner in range(NUM_INNER_FOLDS):
                print("train_inner_fold=", inner)
                feature_priority, auc_priority = [], []
                features = np.arange(0,180)
                
                # Load in train data
                train_data, train_labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, outer, inner)
                train_data, train_labels, train_lengths = create_batches(train_data, train_labels, "linear", 128)
                
                # Load in val data
                val_data, val_labels, val_names = extract_val_data(K_FOLD_PATH + MELSPEC, outer, inner)
                val_data, val_labels, val_lengths = create_batches(val_data, val_labels, "linear", 128)
                
                # iterate feature-1 times
                while len(feature_priority) != 180:
                    performance = []
                    
                    base_features = []
                    for batch in range(len(train_data)):
                        base_features_batch = []
                        for prev_select_feature in feature_priority:
                            base_features_batch.append(np.asarray(train_data[batch][:,:,int(prev_select_feature)]))
                        if len(feature_priority) > 0:
                            base_features.append(th.as_tensor(np.stack(base_features_batch, -1)))

                    base_val_features = []
                    for batch in range(len(val_data)):
                        base_val_features_batch = []
                        for prev_select_feature in feature_priority:
                            base_val_features_batch.append(np.asarray(val_data[batch][:,:,int(prev_select_feature)]))
                        if len(feature_priority) > 0:
                            base_val_features.append(th.as_tensor(np.stack(base_val_features_batch, -1)))
                    
                    # Pass through all unselected features
                    for feature in features:
                        # create new model
                        model = bi_lstm_package(len(feature_priority)+1, BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, inner, working_folder, epochs=16, batch_size=128, model_type="FSS")

                        current_features = base_features.copy()
                        new_feature = []
                        for batch in range(len(train_data)):
                            new_feature.append(th.unsqueeze(th.as_tensor(np.asarray(train_data[batch][:,:,feature])), -1))
                        #concat current_features and new_feature
                        if (len(current_features) !=0):
                            for i in range(len(new_feature)):
                                current_features[i] = th.cat((current_features[i] ,new_feature[i]), -1)
                            print("current_feature:",current_features[0].shape)
                        else:
                            current_features = new_feature

                        current_val_features = base_val_features.copy()
                        new_val_feature = []
                        for batch in range(len(val_data)):
                            new_val_feature.append(th.unsqueeze(th.as_tensor(np.asarray(val_data[batch][:,:,feature])),-1))
                        if (len(current_val_features) !=0):
                            for i in range(len(new_val_feature)):
                                current_val_features[i] = th.cat((current_val_features[i] ,new_val_feature[i]), -1)
                            print("current_feature:",current_val_features[0].shape)
                        else:
                            current_val_features = new_val_feature

                
                        # train model on current features
                        train(current_features, train_labels, train_lengths, model)
                            
                        # Assess on the corresponding val data and store the performance in a list
                        th.manual_seed(model.seed) #set the seed to be the same as the one the model was generated on
                        results, test_labels, threshold = val_fss(current_val_features, val_labels, val_lengths, val_names, model)
                        auc = roc_auc_score(test_labels, results)
                        performance.append(auc)
                        print("Feature:", feature, "AUC:", auc)

                        # force delete loaded in model
                        del model
                        gc.collect()
                
                    # select best performing feature from list
                    best_feature = np.argmax(np.array(performance))
                    print("Features array:", features)
                    print("Best feature:", best_feature)
                    print("Array selection", features[best_feature])
                    feature_priority.append(str(features[best_feature]))
                    auc_priority.append(str(performance[best_feature]))
                    print("Best performing feature:", best_feature, "with an auc of:", performance[best_feature])

                    # train model on best performing feature and save
                    # create new model
                    model = bi_lstm_package(len(feature_priority),BEST_HIDDEN_LAYERS[outer], BEST_LAYERS[outer], outer, inner, working_folder, epochs=16, batch_size=128, model_type="FSS")
                        
                    # train model on best feature and save
                    train(current_features, train_labels, train_lengths, model)
                    pickle.dump(model, open("../models/tb/bi_lstm/" + model.model_type + "/" + model.folder + "/" + model.name + "melspec_180_outer_fold_" + str(model.outer) + 
                        "_inner_fold_" + str(model.inner) + "_features_" + str(len(feature_priority)), 'wb')) # save the model
                    
                    # delete the previous model
                    previous_model_path = str("../models/tb/bi_lstm/" + model.model_type + "/" + model.folder + "/" + model.name + "melspec_180_outer_fold_" + str(model.outer) + 
                        "_inner_fold_" + str(model.inner) + "_features_" + str(len(feature_priority)-1))
                    
                    if os.path.exists(previous_model_path):
                        os.remove(previous_model_path)
                        
                    features = np.delete(features, [best_feature])
                    
                    # force delete loaded in model
                    del model
                    gc.collect()

                    #save current feature list
                    file_name = MODEL_PATH + "FSS/" + working_folder + "/features_outer_" + str(outer) + "_inner_" + str(inner) + ".txt"
                    with open(file_name, 'w') as f:
                        for feature in feature_priority:
                            f.write("%s\n" % feature)

                    # save current auc list
                    file_name = MODEL_PATH + "FSS/" + working_folder + "/auc_outer_" + str(outer) + "_inner_" + str(inner) + ".txt"
                    with open(file_name, 'w') as f:
                        for auc in auc_priority:
                            f.write("%s\n" % auc)
        


           

if __name__ == "__main__":
    main()