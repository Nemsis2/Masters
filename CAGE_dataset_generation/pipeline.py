import librosa
import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm

#grab list
list_dir = '../../data/tb/CAGE_03_04_2023/lists/10_fold_cross_validation/'
patient_dir = '../../data/tb/CAGE_03_04_2023/audio/'


def normalize(mspec, min_log_value = -80):
    
    _normer = -(min_log_value)/2
    
    return (mspec + _normer)/_normer

# grab the patient data csv
patient_labels = pd.read_csv('../../data/tb/CAGE_03_04_2023/labels.csv')

for i in range(1,11):
    # read in the patient list
    filename = f'{list_dir}{i}.lst'
    print(filename)
    with open(filename) as input_data:
        patient_list = input_data.read().splitlines()

    # go through all patients in the list
    # grab all coughs for the patient
    cough_list = []
    for patient in patient_list:
        for cough in os.listdir(f'{patient_dir}{patient}'):
            # import the audio
            audio = librosa.load(f'{patient_dir}{patient}/{cough}', sr=None)

            # convert audio to melspec
            melspec_ = librosa.feature.melspectrogram(y=audio[0], 
                                    sr=audio[1], 
                                    n_mels=180, 
                                    n_fft=2048, 
                                    hop_length=512)
    
            # conver melspec to db and normalize to between -1 and 1
            melspec_ = normalize(librosa.power_to_db(melspec_, ref=np.max))

            # append to a list the: [patient_id, melspec, Tb status]
            cough_list.append([patient, melspec_, patient_labels.loc[patient_labels['Patient_ID'] == patient].to_numpy()[0][-1]])

    #save this list of data as a pkl
    melspec_dir = f'../../data/tb/CAGE_03_04_2023/melspecs/fold_{i}'
    print()
    with open(melspec_dir, 'wb') as handle:
        pickle.dump(cough_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
