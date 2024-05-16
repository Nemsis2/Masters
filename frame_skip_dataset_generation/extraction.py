import pickle
import pipeline
import numpy as np
import copy



"""
Code originally created by Geoffrey Frost
Edited by Michael Knight
Creation Date: 18 Jan 2024
"""


if __name__ == '__main__':
    
    # Load in data
    path = '3way_renier_data.pkl'
    with open(path, 'rb') as f:
        r_data_base = pickle.load(f)

    path = '3way_data.pkl'
    with open(path, 'rb') as f:
        c_data_base = pickle.load(f)

    # Do feature extraction for each outer fold
    for i in range(3):
        for feature_type in ['melspec', 'lfb', 'mfcc']:
            # Train dataset
            # Remove test patients before preprocessing
            path = f'/tardis_copies/masters/data/tb/frame_skip/test_patients_fold_{i}.txt'
            test_patients = np.loadtxt(path, delimiter=',', dtype=str)

            c_data = copy.deepcopy(c_data_base)
            r_data = copy.deepcopy(r_data_base)

            win_length = 2048
            hop_length = 512

            for p in test_patients:
                if p[0:2] =='Wu': del c_data[p]
                else: del r_data[p]

            n_mels = [80, 128, 180]
            n_mfccs = [13, 26, 39]
            n_lfb = [80, 128, 180]
            for j in range(3):
                print(f'Creating dataset... \n'
                    f'Fold: {i} \n'  
                    f'type: {feature_type} \n'
                    )
                dataset, splits = pipeline.kfold_combined_dataset(c_data, r_data, n_splits=4, feature=feature_type, win_length=win_length, hop_length=hop_length, n_mels=n_mels[j], n_mfccs=n_mfccs[j], n_bins=n_lfb[j])
                
                if feature_type == 'melspec':
                    fname = f'{n_mels[j]}_{feature_type}_fold_{i}.pkl'
                elif feature_type == 'mfcc':
                    fname = f'{n_mfccs[j]}_{feature_type}_fold_{i}.pkl'
                elif feature_type == 'lfb':
                    fname = f'{n_lfb[j]}_{feature_type}_fold_{i}.pkl'

                save_path = f'/tardis_copies/masters/data/tb/frame_skip/'
                with open(save_path+fname, 'wb') as handle:
                    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
                
                with open(f'{save_path}splits_fold_{i}.pkl', 'wb') as handle:
                    pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)

                # Test dataset
                # Remove test patients before preprocessing
                test_patients = np.loadtxt(path, delimiter=',', dtype=str)
                c_data_cpy = {}
                r_data_cpy = {}

                for p in test_patients:
                    if p[0:2] =='Wu': c_data_cpy[p] = c_data_base[p]
                    else: r_data_cpy[p] =  r_data_base[p]

                # remove 3way speed pertubations
                for p in list(c_data_cpy.keys()):
                    for c in list(c_data_cpy[p].keys()):
                        if c.split('_')[-1] != 'base': del c_data_cpy[p][c]

                for p in list(r_data_cpy.keys()):
                    for c in list(r_data_cpy[p].keys()):
                        if c.split('_')[-1] != 'base': del r_data_cpy[p][c]

                test_dataset = pipeline.combined_test_dataset(c_data_cpy, r_data_cpy, feature=feature_type, n_mels=n_mels[j], n_mfccs=n_mfccs[j], n_bins=n_lfb[j])
                
                fname = f'test_dataset_{feature_type}_{n_mels[j]}_fold_{i}.pkl'

                if feature_type == 'melspec':
                    fname = f'test_dataset_{feature_type}_{n_mels[j]}_fold_{i}.pkl'
                elif feature_type == 'mfcc':
                    fname = f'test_dataset_{feature_type}_{n_mfccs[j]}_fold_{i}.pkl'
                elif feature_type == 'lfb':
                    fname = f'test_dataset_{feature_type}_{n_lfb[j]}_fold_{i}.pkl'

                save_path = f'/tardis_copies/masters/data/tb/frame_skip/test/'
                with open(save_path+fname, 'wb') as handle:
                    pickle.dump(test_dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)