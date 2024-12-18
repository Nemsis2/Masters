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


def labels_per_frame(data, labels):
      per_frame_labels = []
      for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                  per_frame_labels.append(labels[i])
      
      return np.array(per_frame_labels)
     

def cough_labels_per_frame(data):
      cough_labels = []
      for cough in range(data.shape[0]):
            for frame in range(data[cough].shape[0]):
                  cough_labels.append(cough)
      
      return cough_labels

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
      tpr = np.delete(tpr,0)
      fpr = np.delete(fpr,0)
      threshold = np.delete(threshold,0)
      fnr = 1 - tpr

      index = np.nanargmin(np.absolute((fnr - fpr)))
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

def load_model(model_path):
      model = pickle.load(open(model_path, 'rb')) # load in the model
      return model

def select_features(train_data, dev_data, feature_priority, feature):
      chosen_features, chosen_features_dev = [], []
      for prev_select_feature in feature_priority:
            chosen_features.append(np.asarray(train_data[:,int(prev_select_feature)]))
            chosen_features_dev.append(np.asarray(dev_data[:,int(prev_select_feature)]))
      chosen_features.append(np.asarray(train_data[:,feature]))
      chosen_features_dev.append(np.asarray(dev_data[:,feature]))
      chosen_features = th.as_tensor(np.stack(chosen_features, -1))
      chosen_features_dev = th.as_tensor(np.stack(chosen_features_dev, -1))

      return chosen_features, chosen_features_dev
