import torch as th
import torch.nn as nn
import numpy as np
import logging
import os
import pickle
from sklearn.metrics import roc_auc_score, roc_curve


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
def save_model(model, feature_type, n_feature, model_type, outer, inner):
      model_path = f'../../models/tb/resnet/{model.name}/{feature_type}/{n_feature}_{feature_type}/{model_type}'
      if model_type == 'dev':
            pickle.dump(model, open(f'{model_path}/{model.name}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}', 'wb')) # save the model
      if model_type == 'ts' or model_type == 'ts_2':
            pickle.dump(model, open(f'{model_path}/{model.name}_{feature_type}_{n_feature}_outer_fold_{outer}', 'wb')) # save the model


                  
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
compare labels with predicted labels and get the AUC as well as true_p, flase_p, true_n and false_n per patient
"""
def patient_performance_assess(y, results, patients, optimal_threshold):
      patients = sort_patient_id(patients)
      patient_predictions, patient_true_predictions = make_patient_predictions(y, results, patients, optimal_threshold)
      auc = roc_auc_score(patient_true_predictions, patient_predictions)
      sens, spec = calculate_sens_spec(patient_true_predictions, patient_predictions)

      return auc, sens, spec


"""
get the EER decision threshold for the corresponding validation set
"""
def get_EER_threshold(y, results):
      fpr, tpr, threshold = roc_curve(y, results, pos_label=1)
      fnr = 1 - tpr
      index = np.nanargmin(np.absolute((fnr - fpr)))
      if index == 0:
            optimal_threshold = 1
      else:
            optimal_threshold = threshold[index]
      
      return optimal_threshold



def get_oracle_thresholds(labels, results):
      fpr, tpr, threshold = roc_curve(labels, results, pos_label=1)
      tpr = np.delete(tpr,0)
      fpr = np.delete(fpr,0)
      threshold = np.delete(threshold,0)
      
      sens_threshold, spec_threshold = np.zeros(len(threshold)), np.zeros(len(threshold))
      for i in range(len(threshold)):
            thresholded_results = (np.array(results)>threshold[i]).astype(np.int8)
            sens, spec = calculate_sens_spec(labels, thresholded_results)
            sens_threshold[i] = np.abs(sens-0.9)
            spec_threshold[i] = np.abs(spec-0.7)
    
      sens = np.nanargmin(sens_threshold)
      spec = np.nanargmin(spec_threshold)
      
      return threshold[sens], threshold[spec]

def gather_results(results, labels, names):
      """
      Description:
      ---------

      Inputs:
      ---------
            results: multiple model prob predictions for each value in the data with shape num_models x num_data_samples

            labels: list or array which contains a label for each value in the data

            names: list or array of patient_id associated with each value in the data

      Outputs:
      --------
            out[:,1]: averaged model prob predictions for each unique patient_id in names

            out[:,2]: label associated with each value in out[:,1]
      """

      unq,ids,count = np.unique(names,return_inverse=True,return_counts=True)
      out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,labels[:,0])/count))
      return out[:,1], out[:,2]


def calculate_metrics(labels, results):
      auc = roc_auc_score(labels, results)
        
      # get eer, and oracle thresholds
      eer_threshold = get_EER_threshold(labels, results)
      sens_threshold, spec_threshold = get_oracle_thresholds(labels, results)

      # eer sens and spec
      eer_results = (np.array(results)>eer_threshold).astype(np.int8)
      inner_eer_sens, inner_eer_spec = calculate_sens_spec(labels, eer_results)

      # oracle sens and spec
      # using locked sens = 0.9
      sens_results = (np.array(results)>sens_threshold).astype(np.int8)
      _, inner_oracle_spec = calculate_sens_spec(labels, sens_results)

      # using locked spec = 0.7
      spec_results = (np.array(results)>spec_threshold).astype(np.int8)
      inner_oracle_sens, _ = calculate_sens_spec(labels, spec_results)

      return auc, inner_eer_sens, inner_eer_spec, inner_oracle_sens, inner_oracle_spec