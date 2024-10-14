# libraries
import torch as th
import pickle
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 64


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)



def normalize_mfcc(data):
      for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                  if np.all(data[i][j]) != 0:
                        data[i][j] = (data[i][j]-np.max(data[i][j]))/(np.max(data[i][j])-np.min(data[i][j]))

      return data



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


def load_test_data(k_fold_path):
    data = pickle.load(open(k_fold_path, 'rb'))
    data = np.array(data, dtype=object)
    names = data[:,0]
    data_ = data[:,1]
    labels = data[:,2]

    if feature_type=="mfcc":
        data_ = normalize_mfcc(data_)

    data_ = np.array([np.mean(x, axis=0) for x in data_])

    return data_, labels.astype("int"), names


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
    labels = np.array(labels, dtype=np.int64)
    out = np.column_stack((unq,np.bincount(ids,results[:,1])/count, np.bincount(ids,labels)/count))
    return out[:,1], out[:,2].astype('int')


def test_lr(feature_type, n_feature):
    data_path = f'../../data/tb/CAGE_QC/{feature_type}/{n_feature}/'
    model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/'
    auc, sens, spec, oracle_sens, oracle_spec = 0, 0, 0, 0, 0
    for outer in range(1,11):
        # grab all models to be tested for that outer fold
        models = []
        for model_outer in range(3):
            for inner in range(4):
                inner_model_path = f'{model_path}lr_{feature_type}_{n_feature}_outer_fold_{model_outer}_inner_fold_{inner}'
                models.append(pickle.load(open(inner_model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'{data_path}fold_{outer}.pkl' 
        data, labels, names = load_test_data(k_fold_path)

        # do a forward pass through the models
        results = []
        for model in models:
            results.append(model.predict_proba(data))

        # gather the results over all models and average by patient
        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        # average over models and calculate auc
        results = sum(output)/4
        auc = roc_auc_score(new_labels, results)
        print(auc)

        # get eer, and oracle thresholds
        eer_threshold = get_EER_threshold(new_labels, results)
        sens_threshold, spec_threshold = get_oracle_thresholds(new_labels, results)

        # eer sens and spec
        eer_results = (np.array(results)>eer_threshold).astype(np.int8)
        inner_eer_sens, inner_eer_spec = calculate_sens_spec(new_labels, eer_results)

        # oracle sens and spec
        # using locked sens = 0.9
        sens_results = (np.array(results)>sens_threshold).astype(np.int8)
        _, inner_oracle_spec = calculate_sens_spec(new_labels, sens_results)

        # using locked spec = 0.7
        spec_results = (np.array(results)>spec_threshold).astype(np.int8)
        inner_oracle_sens, _ = calculate_sens_spec(new_labels, spec_results)

        # eer sens and spec
        sens += inner_eer_sens
        spec += inner_eer_spec
        
        # oracle sens and spec
        oracle_sens += inner_oracle_sens
        oracle_spec += inner_oracle_spec

    return auc/10, sens/10, spec/10, oracle_sens/10, oracle_spec/10



for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180]

        for n_feature in features:
            print(f'testing {feature_type} with {n_feature}:')
            auc = test_lr(feature_type, n_feature)
            print(auc)