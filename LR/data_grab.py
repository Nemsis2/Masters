import numpy as np
import pickle
from helper_scripts import sort_patient_id
K_FOLD_PATH = "../data/tb/combo/multi_folds/"


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
            for i, t in zip( data[inner_fold]['train']['inps'], data[inner_fold]['train']['tgts']):
                batch.append([i, t])

            for labels in data[inner_fold]['val'].keys():
                # get the validation data
                for i,t in zip( data[inner_fold]['val'][labels]['inps'], data[inner_fold]['val'][labels]['tgts']):
                        batch.append([i,t])

    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch
    
    return batch_data, batch_labels


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