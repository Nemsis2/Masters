import torch as th
import torch.nn as nn
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import roc_auc_score, roc_curve



#from lr_main import AUC_metric
from sklearn.metrics import auc


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
sort the patient ids such that each is unqiue
"""
def sort_patient_id(patients):
      # create a new list for patients. this code is horrific but kinda works
      patients_return = []
      current_patient = 0
      new_patient_id = 0
      for i in range(patients.shape[0]):
            if current_patient == patients[i]:
                  patients_return.append(new_patient_id)
            else:
                  current_patient = patients[i]
                  new_patient_id += 1
                  patients_return.append(new_patient_id)

      return patients_return


"""
total the predictions over all the models
"""
def total_predictions(results):
      final_results = results[0]
      for i in range(1,len(results)):
            for j in range(len(results[i])):
                  final_results[j] += results[i][j]

      for i in range(len(final_results)):
            final_results[i] = final_results[i]/len(results)

      return final_results


"""
Turn the results and test labels to binary
"""
def binarise_results(results, test_labels):
      y = []
      binary_results = []
      for i in range(len(results)):
            for j in range(len(results[i])):
                  y.append(test_labels[i][j][0])
                  binary_results.append(float(np.array(results[i][j][0])))

      return y, binary_results

"""
binarise each prediction based off wheter it is greater or smaller than 0.5
"""
def binarise_predictions(predicted_y, results):
      true_positive, false_positive, true_negative, false_negative = 0, 0, 0, 0 
      y_return, results_return = [], []

      for i in range(len(predicted_y)):
            for j in range(len(predicted_y[i])):
                  y_return.append(predicted_y[i][j][0])
                  
                  if predicted_y[i][j][0] == 1: #if the result is positive
                        if results[i][j][0] > 0.5: #if we have predicted positive
                              true_positive += 1
                              results_return.append(1)
                        else: #if we have predicted negative
                              false_negative += 1
                              results_return.append(0)
                  else: #if the result is negative
                        if results[i][j][1] > 0.5: #if we have predicted negative
                              true_negative += 1
                              results_return.append(0)
                        else: #if we have predicted positive
                              false_positive += 1
                              results_return.append(1)

      sens = true_positive/(true_positive + false_negative)
      spec = true_negative/(true_negative + false_positive)

      return y_return, results_return, sens, spec


"""
make predictions per patient
"""
def make_patient_predictions(y, results, patients, optimal_threshold):
      patient_predictions, patient_true_predictions  = [], []
      c, current_prediction, current_true_prediction, current_patient = 0, 0, 0, 0

      for i in range(len(patients)):
            if current_patient == patients[i]:
                  c += 1
                  current_prediction += results[i]
                  current_true_prediction += y[i]
            else:
                  patient_predictions.append(current_prediction/c)
                  patient_true_predictions.append(current_true_prediction/c)
                  current_patient += 1
                  c = 1
                  current_prediction = results[i]
                  current_true_prediction = y[i]

      for i in range(len(patient_predictions)):
            if patient_predictions[i] > optimal_threshold:
                  patient_predictions[i] = 1
            else:
                  patient_predictions[i] = 0

      return patient_predictions, patient_true_predictions



def calculate_sens_spec(patient_true_predictions, patient_predictions):
      """
      calculates the sensitivity and specificity given true labels and predicited labels
      """
      sens, spec, p, n = 0, 0, 0, 0
      for i in range(len(patient_true_predictions)):
            if patient_true_predictions[i] == 1:
                  p += 1
                  if patient_predictions[i] == patient_true_predictions[i]:
                        sens += 1
            elif patient_true_predictions[i] == 0:
                  n += 1
                  if patient_predictions[i] == patient_true_predictions[i]:
                        spec += 1

      return sens/p, spec/n


"""
save the model
"""
def save_model(model, working_folder, train_outer_fold, train_inner_fold, epochs, model_path, model_melspec):
      if train_inner_fold == None:
            pickle.dump(model, open(model_path + working_folder + "/" + model.name + model_melspec + "_outer_fold_" + str(train_outer_fold) + 
                        "_inner_fold_" + str(train_inner_fold), 'wb')) # save the model
      else:
            pickle.dump(model, open(model_path + working_folder + "/" + model.name + model_melspec + "_outer_fold_" + str(train_outer_fold) + 
                        "_inner_fold_" + str(train_inner_fold), 'wb')) # save the model

                  
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
def cough_performance_assess(y, results):
      y, results, sens, spec = binarise_predictions(y, results)
      auc = roc_auc_score(y, results)
      return auc, sens, spec


"""
compare labels with predicted labels and get the AUC as well as true_p, flase_p, true_n and false_n per patient
"""
def patient_performance_assess(y, results, patients, optimal_threshold):
      patients = sort_patient_id(patients)
      patient_predictions, patient_true_predictions = make_patient_predictions(y, results, patients, optimal_threshold)
      auc = roc_auc_score(patient_true_predictions, patient_predictions)
      sens, spec = calculate_sens_spec(patient_true_predictions, patient_predictions)

      return auc, sens, spec


"""
get the optimal decision threshold for the corresponding validation set
"""
def get_optimal_threshold(y, results):
      fpr, tpr, threshold = roc_curve(y, results, pos_label=1)
      fnr = 1 - tpr
      optimal_threshold = threshold[np.nanargmin(np.absolute((fnr - fpr)))]
      
      return optimal_threshold