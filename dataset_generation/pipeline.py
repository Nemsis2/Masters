import librosa
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.model_selection import StratifiedKFold

noisy_ps = ['Wu0381', 'Wu0392', 'Wu0399', 'Wu0403', 'Wu0405', 'Wu0413',
           'Wu0414', 'Wu0417', 'Wu0430', 'Wu0450', 'Wu0471', 'Wu0488',
           'Wu0494']

def normalize(mspec, min_log_value = -80):
    
    _normer = -(min_log_value)/2
    
    return (mspec + _normer)/_normer


def melspec(audio, sr, n_mels, n_fft, hop_length):

    melspec_ = librosa.feature.melspectrogram(y=audio, 
                                    sr=sr, 
                                    n_mels=n_mels, 
                                    n_fft=n_fft, 
                                    hop_length=hop_length)
    
    return normalize(librosa.power_to_db(melspec_, ref=np.max))

def downsample(data, sr=16000):
    for p in tqdm(data):
        for c in data[p]: 
            data[p][c] = (librosa.resample(data[p][c][0], data[p][c][1], sr, res_type='kaiser_fast'), sr)
    return data

def mfcc(audio, sr, n_mfcc, n_fft, hop_length):

    mfcc_ = librosa.feature.mfcc(y=audio, 
                                    sr=sr, 
                                    n_mfcc=n_mfcc, 
                                    n_fft=n_fft, 
                                    hop_length=hop_length)

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

def mfccs(data, n_mfcc=13, n_fft=2048, hop_length=512):
    
    data_cpy = {}

    for p in tqdm(data):
        data_cpy[p] = []

        for c in data[p]:
            audio = data[p][c][0]
            sr = data[p][c][1]
            data_cpy[p].append(mfcc(audio, sr, n_mfcc, n_fft, hop_length).T)
    
    return data_cpy

def melspecs(data, n_mels=128, n_fft=2048, hop_length=512):
    
    data_cpy = {}

    for p in tqdm(data):
        data_cpy[p] = []

        for c in data[p]:
            audio = data[p][c][0]
            sr = data[p][c][1]
            # Geoff work
            sample = melspec(audio, sr, n_mels, n_fft, hop_length).T
            data_cpy[p].append(sample)
    
    return data_cpy

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

def lfb_energies(data, n_bins=140, hop_length=512, n_fft=2048, sr=44100):
    data_cpy = {}

    for p in tqdm(data):
        data_cpy[p] = []

        for c in data[p]:
            audio = data[p][c][0]
            sr = data[p][c][1]
            S, n_fft = librosa.core.spectrum._spectrogram(
                y=audio,
                n_fft=n_fft,
                hop_length=hop_length,
                power=2,
                win_length=None,
                window="hann",
                center=True,
                pad_mode="reflect")

            data_cpy[p].append(lfb_energy(S, n_bins, n_fft, sr).T)
    
    return data_cpy

def get_melspec_dataset(data, n_mels=128, n_fft=2048, hop_length=512, remove_noisy=True, train=True):

    # Construct train/test dataset
    if train:

        if remove_noisy:
            data, labels = remove_noisy_ps(data, get_labels())
        
        else: 
            labels = get_labels()

        dataset = {}
        data = melspecs(data, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length)   

        for i, p in enumerate(data):
            fold = f'fold_{i}'
            dataset[fold] = {}
            
            test_data = {p: data[p]}
            test_label = {p: labels[p]}

            train_data = data.copy()
            train_labels = labels.copy()
            
            del train_data[p]
            del train_labels[p]

            dataset_test = create_dataset(test_data, test_label)
            dataset_train = create_dataset(train_data, train_labels)

            dataset[fold]['train'] = dataset_train
            dataset[fold]['test'] = dataset_test

    # Construct dev dataset 
    else:
        labels = {p: 0 if p.split('_')[0] == 'CONX' else 1 for p in data}
        data = melspecs(data, 
            n_mels=n_mels, 
            n_fft=n_fft, 
            hop_length=hop_length)

        dataset = create_dataset(data, labels, train=True)

    return dataset

def get_splits(data, config):
    
    if config['feature']=='melspec':
        datasets = get_melspec_dataset(data, 
                            n_mels=config['n_mels'], 
                            n_fft=config['n_fft'], 
                            hop_length=config['hop_length'], 
                            remove_noisy=True, 
                            train=True)
    return datasets

def create_dataset(data, labels, train = True, start=0):

    ps = list(data.keys())
    dataset = {}

    if train:
        dataset['inps'] = []
        dataset['tgts'] = []
        dataset['p'] = []
        for i, p in enumerate(ps):
            i = i + start
            for cough in data[p]:
                dataset['inps'].append(cough)
                #dataset['inps'].append(normalize(cough.detach().numpy()))
                dataset['tgts'].append(labels[p])
                dataset['p'].append(i)
                
    else:
        for i, p in enumerate(ps):
            dataset[p] = {}
            dataset[p]['inps'] = []
            dataset[p]['tgts'] = []
            dataset[p]['p'] = []
            for cough in data[p]:
                dataset[p]['inps'].append(cough)
                dataset[p]['tgts'].append(labels[p])
                dataset[p]['p'].append(i)
    return dataset


def remove_noisy_ps(data, labels):

    data_cpy = dict(data)
    labels_cpy = dict(labels)

    for p in noisy_ps: 
        del data_cpy[p]
        del labels_cpy[p]

    return data_cpy, labels_cpy

def get_labels(path = "Cough_patient_metadata.xlsx"):
    
    df = pd.read_excel(path)
    tb = 'Final_TB_Result (1 = TB, 0 = No-TB)'

    labels = {}
    #Wu0392_1
    for index, row in df.iterrows():
        labels[row['Subject ID']] = row['Final_TB_Result (1 = TB, 0 = No-TB)']

    # These coughs don't have recordings
    keys = ['Wu0455', 'Wu0478', 'Wu0451']
    for key in keys: del labels[key]

    return labels

def kfold_combined_dataset(clinic, renier, feature='melspec', n_splits=5, win_length=2048, hop_length=512, n_mels=80, n_mfccs=13, n_bins=80):

    splits = {}

    all_labels = get_labels()
    c_labels = {p: all_labels[p] for p in clinic}
    clinic, c_labels = remove_noisy_ps(clinic, c_labels)
    
    if feature == 'melspec':
        c_data = melspecs(clinic, n_mels=n_mels, n_fft=win_length, hop_length=hop_length)
        r_data = melspecs(renier, n_mels=n_mels, n_fft=win_length, hop_length=hop_length)
    if feature == 'lfb':
        c_data = lfb_energies(clinic, n_bins=n_bins)
        r_data = lfb_energies(renier, n_bins=n_bins)
    if feature == 'mfcc':
        c_data = mfccs(clinic, n_mfcc=n_mfccs)
        r_data = mfccs(renier, n_mfcc=n_mfccs)
    
    dataset = {}


    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    c_data_ps = np.array(list(c_labels.keys()))
    c_data_lb = np.array(list(c_labels.values()))

    
    for i in range(n_splits): 
        splits[f'fold_{i}'] = {}
        splits[f'fold_{i}']['train'] = []
        splits[f'fold_{i}']['val'] = []

    i=0
    for train_index, val_index in skf.split(c_data_ps, c_data_lb):
        
        fold = f'fold_{i}'
        dataset[fold] = {}

        val_data = c_data.copy()
        val_labels = c_labels.copy()

        for p in c_data_ps[train_index]: del val_data[p]
        for p in c_data_ps[train_index]: del val_labels[p]

        c_train_p = [{p:c_labels[p]} for p in c_data_ps[train_index]]
        splits[fold]['train'] += c_train_p

        train_data = c_data.copy()
        train_labels = c_labels.copy()
        
        for p in c_data_ps[val_index]: del train_data[p]
        for p in c_data_ps[val_index]: del train_labels[p]

        c_val_p = [{p:c_labels[p]} for p in c_data_ps[val_index]]
        splits[fold]['val'] += c_val_p

        dataset_val = create_dataset(val_data, val_labels, train=False)
        dataset_train = create_dataset(train_data, train_labels)

        dataset[fold]['train'] = dataset_train
        dataset[fold]['val'] = dataset_val

        i+=1
    #return dataset
    r_labels = {p: 0 if p.split('_')[0] == 'CONX' else 1 for p in r_data}

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    
    r_data_ps = np.array(list(r_labels.keys()))
    r_data_lb = np.array(list(r_labels.values()))

    i=0
    for train_index, val_index in skf.split(r_data_ps, r_data_lb):
        fold = f'fold_{i}'
        start = dataset[fold]['train']['p'][-1] + 1
        
        
        val_data = r_data.copy()
        val_labels = r_labels.copy()

        for p in r_data_ps[train_index]: del val_data[p]
        for p in r_data_ps[train_index]: del val_labels[p]

        r_train_p = [{p:r_labels[p]} for p in r_data_ps[train_index]]
        splits[fold]['train'] += r_train_p

        train_data = r_data.copy()
        train_labels = r_labels.copy()
        
        for p in r_data_ps[val_index]: del train_data[p]
        for p in r_data_ps[val_index]: del train_labels[p]

        r_val_p = [{p:r_labels[p]} for p in r_data_ps[val_index]]
        splits[fold]['val'] += r_val_p

        dataset_val = create_dataset(val_data, val_labels, train=False)
        dataset_train = create_dataset(train_data, train_labels, start=start)

        dataset[fold]['train']['inps'] += dataset_train['inps']
        dataset[fold]['train']['tgts'] += dataset_train['tgts']
        dataset[fold]['train']['p'] += dataset_train['p'] 

        for p in dataset_val: dataset[fold]['val'][p] = dataset_val[p]
        i+=1


    return dataset, splits

def combined_test_dataset(clinic, renier, feature='melspec', win_length=2048, hop_length=512, n_mels=80, n_mfccs=13, n_bins=80):

    all_labels = get_labels()
    c_labels = {p: all_labels[p] for p in clinic}
    
    if feature == 'melspec':
        c_data = melspecs(clinic, n_mels=n_mels, n_fft=win_length, hop_length=hop_length)
        r_data = melspecs(renier, n_mels=n_mels, n_fft=win_length, hop_length=hop_length)
    if feature == 'lfb':
        c_data = lfb_energies(clinic, n_bins=n_bins)
        r_data = lfb_energies(renier, n_bins=n_bins)
    if feature == 'mfcc':
        c_data = mfccs(clinic, n_mfcc=n_mfccs)
        r_data = mfccs(renier, n_mfcc=n_mfccs)

    r_labels = {p: 0 if p.split('_')[0] == 'CONX' else 1 for p in r_data}
    dataset = {}

    dataset_test_c = create_dataset(c_data, c_labels, train=False)
    dataset_test_r = create_dataset(r_data, r_labels, train=False)

    return {**dataset_test_c, **dataset_test_r}