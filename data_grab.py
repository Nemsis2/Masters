import numpy as np
import pickle
K_FOLD_PATH = "../data/tb/combo/multi_folds/"



"""
get train data for a specific inner and outer fold
"""
def extract_inner_fold_data(path, outer_fold, inner_fold, final_model):
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))


    # zip the information from the dictionary into a list of arrays
    for inner_fold_key in data.keys():
        if inner_fold_key == ("fold_" + str(inner_fold)):
            # grab the training data
            for i, t in zip( data[inner_fold_key]['train']['inps'], data[inner_fold_key]['train']['tgts']):
                batch.append([i, t])

            if final_model == 1: # if this is a final model
                for labels in data[inner_fold_key]['val'].keys():
                    # get the validation data
                    for i,t in zip( data[inner_fold_key]['val'][labels]['inps'], data[inner_fold_key]['val'][labels]['tgts']):
                        batch.append([i, t])

    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch

    return batch_data, batch_labels



"""
get all data in an outer fold
extracts only train data if final_model =0 and includes validation data if final_model=1
"""
def extract_outer_fold_data(path, outer_fold, final_model):
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold in data.keys():
        # get the training data
        for i, t in zip( data[inner_fold]['train']['inps'], data[inner_fold]['train']['tgts']):
            batch.append([i, t])

        if final_model == 1:
            for labels in data[inner_fold]['val'].keys():
                # get the validation data
                for i,t in zip( data[inner_fold]['val'][labels]['inps'], data[inner_fold]['val'][labels]['tgts']):
                        batch.append([i,t])

    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch
    
    return batch_data, batch_labels



"""
get val data for a specific inner and outer fold
"""
def extract_val_data(path, outer_fold, inner_fold):
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold_key in data.keys():
        if inner_fold_key == ("fold_"+str(inner_fold)):
                for labels in data[inner_fold_key]['val'].keys():
                    for i,t in zip( data[inner_fold_key]['val'][labels]['inps'], data[inner_fold_key]['val'][labels]['tgts']):
                            batch.append([i,t])
    
    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch
    
    return batch_data, batch_labels


def extract_outer_val_data(path, outer_fold):
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path + str(outer_fold) + ".pkl", 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold_key in data.keys():
        for labels in data[inner_fold_key]['val'].keys():
            for i,t in zip(data[inner_fold_key]['val'][labels]['inps'], data[inner_fold_key]['val'][labels]['tgts']):
                batch.append([i,t])
    
    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch
    
    return batch_data, batch_labels


"""
get the test data
"""
def extract_test_data(path, fold):
    batch = []
    # read in the data located at the path 
    data = pickle.load(open(path + str(fold) + ".pkl", 'rb'))

    # zip the information from the dictionary into a list of arrays
    for inner_fold in data.keys():
        for i, t in zip( data[inner_fold]['inps'], data[inner_fold]['tgts']):
                batch.append([i, t])

    batch = np.array(batch, dtype=object)

    batch_data = batch[:,0]
    batch_labels = batch[:,1]
    
    return batch_data, batch_labels