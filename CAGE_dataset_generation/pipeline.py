import librosa
import os
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm

#grab list
list_dir = '../../data/tb/CAGE_QC/lists/10_fold_cross_validation/'
patient_dir = '../../data/tb/CAGE_QC/segmented_audio/'


def normalize(mspec, min_log_value = -80):
    
    _normer = -(min_log_value)/2
    
    return (mspec + _normer)/_normer


def melspec(audio, n_mels):
        # convert audio to melspec
        melspec_ = librosa.feature.melspectrogram(y=audio[0], sr=audio[1], n_mels=n_mels, n_fft=2048, hop_length=512)

        # convert melspec to db and normalize to between -1 and 1
        return normalize(librosa.power_to_db(melspec_, ref=np.max))


def mfcc(audio, n_mfcc):
    mfcc_ = librosa.feature.mfcc(y=audio[0], sr=audio[1], n_mfcc=n_mfcc, n_fft=2048, hop_length=512)

    if mfcc_.shape[-1] > 9:
        mfcc_delta = librosa.feature.delta(mfcc_)
        mfcc_delta_delta = librosa.feature.delta(mfcc_, order=2)
    else: 
        mfcc_delta = np.zeros_like(mfcc_)
        mfcc_delta_delta = np.zeros_like(mfcc_)
    
    # Mean and varince norm
    mfccs_ = np.concatenate((mfcc_, mfcc_delta, mfcc_delta_delta), axis=0)
    mfccs_mean = (mfccs_.T - np.mean(mfccs_, axis=1)).T
    var = np.std(mfccs_, axis=1)
    var = np.power(var, 2)
    if not np.any(var==0): return (mfccs_mean.T/var).T
    else: return mfccs_mean


def lfb(n_bins=140, n_fft=2048, sr=44100):
    weights = np.zeros((n_bins, int(1 + n_fft // 2)))
    # Center freqs of each FFT bin
    fftfreqs = np.arange(1 + n_fft // 2)*sr / n_fft
    lfbfreqs = np.arange(start=0, stop=fftfreqs[-1], step=fftfreqs[-1]/(n_bins+2))

    ramps = np.subtract.outer(lfbfreqs, fftfreqs)

    fdiff = np.diff(lfbfreqs)
    for i in range(n_bins):
        # lower and upper slopes for all bins
        lower = -ramps[i] / fdiff[i]
        upper = ramps[i + 2] / fdiff[i + 1]
        weights[i] = np.maximum(0, np.minimum(lower, upper))

    enorm = 2.0 / (lfbfreqs[2 : n_bins + 2] - lfbfreqs[:n_bins])
    weights *= enorm[:, np.newaxis]

    return weights


def lfb_energy(S, n_bins, n_fft, sr):
    fb = lfb(n_bins, n_fft, sr)
    return normalize(librosa.power_to_db(np.dot(fb, S), ref=np.max))


def create_melspecs(patient_list):
    for n_mels in [80, 128, 180]:
        # go through all patients in the list
        # grab all coughs for the patient
        cough_list = []
        for patient in patient_list:
            for cough in os.listdir(f'{patient_dir}{patient}'):
                # import the audio
                audio = librosa.load(f'{patient_dir}{patient}/{cough}', sr=None)

                melspec_ = melspec(audio, n_mels).T

                # append to a list the: [patient_id, melspec, Tb status]
                cough_list.append([patient, melspec_, patient_labels.loc[patient_labels['Patient_ID'] == patient].to_numpy()[0][-1]])

        #save this list of data as a pkl
        melspec_dir = f'../../data/tb/CAGE_QC/mfb/{n_mels}/fold_{i}.pkl'
        with open(melspec_dir, 'wb') as handle:
            pickle.dump(cough_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_linear_filterbanks(patient_list):
    for n_lfbs in [80, 128, 180]:
        # go through all patients in the list
        # grab all coughs for the patient
        cough_list = []
        for patient in patient_list:
            for cough in os.listdir(f'{patient_dir}{patient}'):
                # import the audio
                audio = librosa.load(f'{patient_dir}{patient}/{cough}', sr=None)
                S, n_fft = librosa.core.spectrum._spectrogram(y=audio[0], n_fft=2048, hop_length=512, power=2, win_length=None, window="hann", center=True, pad_mode="reflect")

                lfb_ = (lfb_energy(S, n_lfbs, n_fft, audio[1]).T)

                # append to a list the: [patient_id, melspec, Tb status]
                cough_list.append([patient, lfb_, patient_labels.loc[patient_labels['Patient_ID'] == patient].to_numpy()[0][-1]])


            #save this list of data as a pkl
        lfb_dir = f'../../data/tb/CAGE_QC/lfb/{n_lfbs}/fold_{i}.pkl'
        with open(lfb_dir, 'wb') as handle:
            pickle.dump(cough_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_mfccs(patient_list):
    for n_mfccs in [13, 26,39]:
        # go through all patients in the list
        # grab all coughs for the patient
        cough_list = []
        for patient in patient_list:
            for cough in os.listdir(f'{patient_dir}{patient}'):
                # import the audio
                audio = librosa.load(f'{patient_dir}{patient}/{cough}', sr=None)

                mfcc_ = mfcc(audio, n_mfccs).T

                # append to a list the: [patient_id, melspec, Tb status]
                cough_list.append([patient, mfcc_, patient_labels.loc[patient_labels['Patient_ID'] == patient].to_numpy()[0][-1]])

        #save this list of data as a pkl
        mfcc_dir = f'../../data/tb/CAGE_QC/mfcc/{n_mfccs}/fold_{i}.pkl'
        with open(mfcc_dir, 'wb') as handle:
            pickle.dump(cough_list, handle, protocol=pickle.HIGHEST_PROTOCOL)


# grab the patient data csv
patient_labels = pd.read_csv('../../data/tb/CAGE_QC/labels.csv')

for i in range(1,11):
    # read in the patient list
    filename = f'{list_dir}{i}.lst'
    print(filename)
    with open(filename) as input_data:
        patient_list = input_data.read().splitlines()

    create_melspecs(patient_list)

    create_linear_filterbanks(patient_list)

    create_mfccs(patient_list)