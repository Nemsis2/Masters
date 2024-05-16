import numpy as np
import pickle
from helper_scripts import sort_patient_id, labels_per_frame, cough_labels_per_frame



def normalize_mfcc(data):
      for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                  if np.all(data[i][j]) != 0:
                        data[i][j] = (data[i][j]-np.max(data[i][j]))/(np.max(data[i][j])-np.min(data[i][j]))

      return data

def extract_inner_fold_data(path, inner_fold):
    """
    Description:
    ---------
    Extracts training data for the relevant inner fold given the path to the data.
    
    Inputs:
    ---------
    path: (string) path to the data from the working directory

    inner_fold: (int) inner fold to be selected from the data

    Outputs:
    --------
    batch_data: (np.array) array of all extracted data

    batch_labels: (np.array) array of labels associated with each extracted data point
    """

    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path, 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold_key in data.keys():
        if inner_fold_key == ("fold_" + str(inner_fold)):
            # grab the training data
            for i, t in zip( data[inner_fold_key]['train']['inps'], data[inner_fold_key]['train']['tgts']):
                batch.append([i, t])

    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch

    return batch_data, batch_labels


def extract_outer_fold_data(path):
    """
    Description:
    ---------
    Extracts training data for the relevant outer fold given the path to the data.
    
    Inputs:
    ---------
    path: (string) path to the data from the working directory

    Outputs:
    --------
    batch_data: (np.array) array of all extracted data

    batch_labels: (np.array) array of labels associated with each extracted data point
    """
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path, 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold in data.keys():
        if inner_fold == ("fold_" + str(0)):
            # get the training data
            for i,t,p in zip( data[inner_fold]['train']['inps'], data[inner_fold]['train']['tgts'], data[inner_fold]['train']['p']):
                batch.append([i,t,p])

            for labels in data[inner_fold]['val'].keys():
                # get the validation data
                for i,t,p in zip( data[inner_fold]['val'][labels]['inps'], data[inner_fold]['val'][labels]['tgts'], data[inner_fold]['val'][labels]['p']):
                        batch.append([i,t,p])

    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch
    batch_names = batch[:,2]
    batch_names = np.array(sort_patient_id(batch_names))
    data, labels
    return batch_data, batch_labels, batch_names


def extract_dev_data(path, inner_fold):
    """
    Description:
    ---------
    Extracts only the dev data for the relevant outer fold given the path to the data.
    
    Inputs:
    ---------
    path: (string) path to the data from the working directory

    inner_fold: (int) inner fold to be selected from the data

    Outputs:
    --------
    batch_data: (np.array) array of all extracted data

    batch_labels: (np.array) array of labels associated with each extracted data point

    batch_names: (np.array) array of ids associated with patients whose coughs are included within the extracte data
    """
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path, 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold_key in data.keys():
        if inner_fold_key == ("fold_"+str(inner_fold)):
            for labels in data[inner_fold_key]['val'].keys():
                for i,t,p in zip(data[inner_fold_key]['val'][labels]['inps'], data[inner_fold_key]['val'][labels]['tgts'], data[inner_fold_key]['val'][labels]['p']):
                    batch.append([i,t,p])
    
    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0]
    batch_labels = batch[:,1]
    batch_names = batch[:,2]
    batch_names = np.array(sort_patient_id(batch_names))
    
    return batch_data, batch_labels, batch_names



def extract_test_data(path):
    """
    Description:
    ---------
    Extracts only the test data for the relevant outer fold given the path to the data.
    
    Inputs:
    ---------
    path: (string) path to the data from the working directory

    Outputs:
    --------
    batch_data: (np.array) array of all extracted data

    batch_labels: (np.array) array of labels associated with each extracted data point

    batch_names: (np.array) array of ids associated with patients whose coughs are included within the extracte data
    """
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path, 'rb'))

    # zip the information from the dictionary into a list of arrays
    for patient_id in data.keys():
        for i, t, p in zip( data[patient_id]['inps'], data[patient_id]['tgts'], data[patient_id]['p']):
            batch.append([i, t, p])

    batch = np.array(batch, dtype=object)
    
    batch_data = batch[:,0]
    batch_labels = batch[:,1]
    batch_names = batch[:,2]
    batch_names = np.array(sort_patient_id(batch_names))

    return batch_data, batch_labels, batch_names



def load_inner_data(k_fold_path, feature_type, inner):
    data, labels = extract_inner_fold_data(k_fold_path, inner)

    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    data = np.array([np.mean(x, axis=0) for x in data])

    return data, labels.astype("int")


def load_inner_per_frame_data(k_fold_path, feature_type, inner):
    data, labels = extract_inner_fold_data(k_fold_path, inner)

    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    labels = labels_per_frame(data, labels)
    data = np.vstack(data)
    return data, labels.astype("int")


def load_dev_data(k_fold_path, feature_type, inner):
    data, labels, names = extract_dev_data(k_fold_path, inner)

    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    data = np.array([np.mean(x, axis=0) for x in data])

    return data, labels.astype("int"), names


def load_dev_per_frame_data(k_fold_path, feature_type, inner):
    data, labels, names = extract_dev_data(k_fold_path, inner)
    
    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    cough_labels = cough_labels_per_frame(data)
    labels = labels_per_frame(data, labels)
    data = np.vstack(data)

    return data, labels.astype("int"), names, cough_labels

def load_test_data(k_fold_path, feature_type):
    data, labels, names = extract_test_data(k_fold_path)

    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    data = np.array([np.mean(x, axis=0) for x in data])

    return data, labels.astype("int"), names


def load_test_per_frame_data(k_fold_path, feature_type):
    data, labels, names = extract_test_data(k_fold_path)
    
    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    cough_labels = cough_labels_per_frame(data)
    labels = labels_per_frame(data, labels)
    data = np.vstack(data)

    return data, labels.astype("int"), names, cough_labels

def load_test_per_frame_data_tbi2(k_fold_path, feature_type):
    data, labels, names = extract_test_data(k_fold_path)
    if feature_type=="mfcc":
        data = normalize_mfcc(data)

    labels = labels_per_frame(data, labels)
    names = labels_per_frame(data, names)
    data = np.vstack(data)

    return data, labels.astype("int"), names, 