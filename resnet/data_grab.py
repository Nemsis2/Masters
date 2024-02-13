import numpy as np
import pickle
from helper_scripts import sort_patient_id


def normalize_mfcc(data):
      for i in range(data.shape[0]):
            for j in range(data[i].shape[0]):
                  if np.all(data[i][j]) != 0:
                        data[i][j] = (data[i][j]-np.max(data[i][j]))/(np.max(data[i][j])-np.min(data[i][j]))

      return data



"""
get train data for a specific inner and outer fold
"""
def extract_inner_fold_data(path, inner_fold):
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


"""
get all data in an outer fold
"""
def extract_outer_fold_data(path):
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


"""
get dev data for a specific inner and outer fold
"""
def extract_dev_data(path, inner_fold):
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


"""
get the test data
"""
def extract_test_data(path):
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