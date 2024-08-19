import torch as th
import torch.nn as nn
import numpy as np
import logging
import pickle
from sklearn.metrics import roc_curve, roc_auc_score


#https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
def to_categorical(y, num_classes):
      """ 1-hot encodes a tensor """
      return np.eye(num_classes, dtype='float')[np.array(y).astype(int)]


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

def normalize_mfcc(data):
      for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                  if np.all(data[i][j]) != 0:
                        data[i][j] = (data[i][j]-np.max(data[i][j]))/(np.max(data[i][j])-np.min(data[i][j]))

      return data

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

def load_model(model_path):
      model = pickle.load(open(model_path, 'rb')) # load in the model
      th.manual_seed(model.seed) # set the seed to be the same as the one the model was generated on
      return model


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


def select_features(train_data, dev_data, feature_priority):
      base_features = []
      for batch in range(len(train_data)):
            base_features_batch = []
            for prev_select_feature in feature_priority:
                  base_features_batch.append(np.asarray(train_data[batch][:,:,int(prev_select_feature)]))
            
            base_features.append((np.stack(base_features_batch, -1)))

      base_dev_features = []
      for batch in range(len(dev_data)):
            base_dev_features_batch = []
            for prev_select_feature in feature_priority:
                  base_dev_features_batch.append(np.asarray(dev_data[batch][:,:,int(prev_select_feature)]))
            
            base_dev_features.append((np.stack(base_dev_features_batch, -1)))

      return base_features, base_dev_features


def add_latest_feature(train_data, dev_data, chosen_features, chosen_features_dev, feature):
      
      if len(chosen_features) != 0:
            new_feature = []
            for batch in range(len(chosen_features)):
                  new_feature.append(th.unsqueeze(th.as_tensor(np.asarray(train_data[batch][:,:,feature])), -1))
            for i in range(len(new_feature)):
                  chosen_features[i] = th.cat((chosen_features[i] ,new_feature[i]), -1)


            new_feature = []
            for batch in range(len(chosen_features_dev)):
                  new_feature.append(th.unsqueeze(th.as_tensor(np.asarray(dev_data[batch][:,:,feature])), -1))
            for i in range(len(new_feature)):
                  chosen_features_dev[i] = th.cat((chosen_features_dev[i] ,new_feature[i]), -1)

      else:
            for batch in range(len(train_data)):
                  base_features_batch = (np.asarray((train_data[batch][:,:,int(feature),np.newaxis])))
                  chosen_features.append(th.as_tensor(base_features_batch))

            for batch in range(len(dev_data)):
                  base_dev_features_batch = (np.asarray(dev_data[batch][:,:,int(feature),np.newaxis]))
                  chosen_features_dev.append(th.as_tensor(base_dev_features_batch))

      return chosen_features, chosen_features_dev