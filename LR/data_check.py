import numpy as np
import pickle

n_feature = 180
feature_type = 'melspec'



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
            for i, t, p in zip( data[inner_fold_key]['train']['inps'], data[inner_fold_key]['train']['tgts'], data[inner_fold_key]['train']['p']):
                batch.append([i, t, p])

    batch = np.array(batch, dtype=object)
    batch_data = batch[:,0] # get the data from the batch
    batch_labels = batch[:,1] # get the labels from the batch
    batch_patients = batch[:,2] # get the number of patients

    return batch_data, batch_labels, batch_patients


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


for outer in range(3):
    for inner in range(4):
        print(f'For Outer Fold {outer} and Inner Fold {inner}')
        k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'

        ############ Train set information ############
        # Import train data
        data, labels, names = extract_inner_fold_data(k_fold_path, inner)
        
        # get the number of patients and the number of coughs per patient
        values, counts = np.unique(names, return_counts=True)
        print(f'Training Data information')
        print(f'# of positive coughs: {sum(labels)}')
        print(f'# of negative coughs: {len(labels)-sum(labels)}')
        print(f'# of coughs: {len(labels)}')
        print(f'# of Unique patients: {len(values)}')
        print(f'mean coughs per patient: {(np.mean(counts))}')

        ############ Dev set information ############
        # Import dev data
        data, labels, names = extract_dev_data(k_fold_path, inner)

        # get the number of patients and the number of coughs per patient
        values, counts = np.unique(names, return_counts=True)
        print(f'Development Data information')
        print(f'# of positive coughs: {sum(labels)}')
        print(f'# of negative coughs: {len(labels)-sum(labels)}')
        print(f'# of coughs: {len(labels)}')
        print(f'# of Unique patients: {len(values)}')
        print(f'mean coughs per patient: {(np.mean(counts))}')

    ############ Test set information ############
    # Import test data
    print(f'Test set information for Outer Fold {outer}')
    k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl'
    data, labels, names = extract_test_data(k_fold_path)

    # get the number of patients and the number of coughs per patient
    values, counts = np.unique(names, return_counts=True)
    print(f'# of positive coughs: {sum(labels)}')
    print(f'# of negative coughs: {len(labels)-sum(labels)}')
    print(f'# of coughs: {len(labels)}')
    print(f'# of Unique patients: {len(values)}')
    print(f'mean coughs per patient: {(np.mean(counts))}')