import torch as th
import torch.nn as nn
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import roc_auc_score


#https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
def to_categorical(y, num_classes):
      """ 1-hot encodes a tensor """
      return np.eye(num_classes, dtype='float')[np.array(y).astype(int)]


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
def save_model(model, working_folder, train_outer_fold, train_inner_fold, final_model, epochs, model_path, model_melspec):
      if train_inner_fold == None:
            if final_model == 0:
                  pickle.dump(model.model, open(model_path + "val/" + working_folder + "/" + model.model_name + model_melspec + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_epochs_" + str(epochs), 'wb')) # save the model
            else:
                  pickle.dump(model.model, open(model_path + "outer/" + working_folder + "/" + model.model_name + model_melspec + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_final_model", 'wb')) # save the model
      else:
            if final_model == 0:
                  pickle.dump(model.model, open(model_path + "val/" + working_folder + "/" + model.model_name + model_melspec + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_epochs_" + str(epochs), 'wb')) # save the model
            else:
                  pickle.dump(model.model, open(model_path + "inner/" + working_folder + "/" + model.model_name + model_melspec + "_outer_fold_" + str(train_outer_fold) + 
                                    "_inner_fold_" + str(train_inner_fold) + "_final_model", 'wb')) # save the model
                  

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
def log_test_info(test_fold, auc, sens, spec):
    logging.basicConfig(filename="log.txt", filemode='a', level=logging.INFO)
    logging_info = "Final performance for test fold:", str(test_fold), "AUC:", str(auc), "Sens", str(sens), "Spec", str(spec)
    logging.info(logging_info)


"""
compare labels with predicted labels and get the AUC as well as true_p, flase_p, true_n and false_n
"""
def performance_assess(y, results):
    true_positive = 0
    false_positive = 0
    true_negative = 0
    false_negative = 0

    temp_results = []
    temp_y = []
    for i in range(len(y)):
        for j in range(len(y[i])):
                temp_y.append(y[i][j][0])
                
                if y[i][j][0] == 1: #if the result is positive
                    if results[i][j][0] > 0.5: #if we have predicted positive
                            true_positive += 1
                            temp_results.append(1)
                    else: #if we have predicted negative
                            false_negative += 1
                            temp_results.append(0)
                else: #if the result is negative
                    if results[i][j][1] > 0.5: #if we have predicted negative
                            true_negative += 1
                            temp_results.append(0)
                    else: #if we have predicted positive
                            false_positive += 1
                            temp_results.append(1)
    
    auc = roc_auc_score(temp_y, temp_results)
    sens = true_positive/(true_positive + false_negative)
    spec = true_negative/(true_negative + false_positive)

    return auc, sens, spec