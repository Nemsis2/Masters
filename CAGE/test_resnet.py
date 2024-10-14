# libraries
import torch as th
import pickle
import numpy as np
import torch.nn as nn
from sklearn.metrics import roc_auc_score


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


# declare global variables
# set hyperpaperameters
BATCH_SIZE = 128

def normalize_mfcc(data):
      for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                  if np.all(data[i][j]) != 0:
                        data[i][j] = (data[i][j]-np.max(data[i][j]))/(np.max(data[i][j])-np.min(data[i][j]))

      return data


#https://discuss.pytorch.org/t/is-there-something-like-keras-utils-to-categorical-in-pytorch/5960
def to_categorical(y, num_classes):
      """ 1-hot encodes a tensor """
      return np.eye(num_classes, dtype='float')[np.array(y).astype(int)]


def reshape_data(data):
    """
    reshape data to size 1 x feature x 224
    """
    del_indx = []
    for i in range(len(data)):
        data[i] = np.transpose(data[i])
        if 224 > data[i].shape[-1]:
                # zero pad the data to be feature x 224
                data[i] = th.tensor((np.pad(data[i], [(0,0), (0,(224 - data[i].shape[-1]))], mode='constant', constant_values=0))).unsqueeze(0).unsqueeze(0)
        else:
                del_indx.append(i)

    # delete the large data
    data = np.delete(data, del_indx)

    return data, del_indx


def create_batches(data, labels, names, batch_size):
    """
    creates a data batch of size batch_size for only given data, labels and patient names
    """
    batched_data = []
    batched_labels = []
    data_lengths = []
    batched_lengths = []

    for i in range(len(data)):
        data_lengths.append(data[i].shape[0])
    
    data, del_indx = reshape_data(data)
    labels = np.delete(labels, del_indx)
    names = np.delete(names, del_indx)

    for i in range(int(np.ceil(len(data)/batch_size))):
        if(len(data) > (i+1)*batch_size):
            batched_data.append(data[i*batch_size:(i+1)*batch_size])
            batched_labels.append(labels[i*batch_size:(i+1)*batch_size])
            batched_lengths.append(data_lengths[i*batch_size:(i+1)*batch_size])
        else: 
            batched_data.append(data[i*batch_size:]) 
            batched_labels.append(labels[i*batch_size:]) 
            batched_lengths.append(data_lengths[i*batch_size:])

    for i in range(len(batched_data)):
        batched_data[i] = th.as_tensor(np.vstack(batched_data[i])).float()
        batched_labels[i] = th.as_tensor(np.vstack(to_categorical(batched_labels[i],2)))

    return batched_data, batched_labels, batched_lengths, names

def to_softmax(results):
      softmax = nn.Softmax(dim=1)
      results = softmax(results)
      return results


def get_predictions(x_batch, model, lengths):
      with th.no_grad():
            results = (to_softmax((model((x_batch).to(device), lengths)).cpu()))
      return results


def test(x, model, lengths):
      results = []
      for i in range(len(x)):
            results.append(get_predictions(x[i], model, lengths[i]))
      return results


def load_test_data(k_fold_path):
    data = pickle.load(open(k_fold_path, 'rb'))
    data = np.array(data, dtype=object)
    names = data[:,0]
    data_ = data[:,1]
    labels = data[:,2]
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


def test_resnet(feature_type, n_feature, model_type):
    data_path = f'../../../data/tb/CAGE_QC/{feature_type}/{n_feature}/'
    for outer in range(1,11):
        # grab all models to be tested for that outer fold
        models = []
        
        for model_outer in range(3):
            for inner in range(3):
                # get the testing models
                model_path = f'../../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/dev/{model_type}_{feature_type}_{n_feature}_outer_fold_{model_outer}_inner_fold_{inner}'
                models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'{data_path}fold_{outer}.pkl'
        data, labels, names = load_test_data(k_fold_path)
        if feature_type == 'mfcc':
            data = normalize_mfcc(data)
        data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)


        results = []
        for model in models:
            results.append(test(data, model.model, lengths)) # do a forward pass through the models

        for i in range(len(results)):
            results[i] = np.vstack(results[i])
        
        labels = np.vstack(labels)
        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        # average over models and calculate auc
        results = sum(output)/len(output)
        auc = roc_auc_score(new_labels, results)
        print(auc)

for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180]

        for n_feature in features:
            for model in ['resnet_18', 'resnet_10']:
                print(f'testing {feature_type} with {n_feature}:')
                auc = test_resnet(feature_type, n_feature, model)