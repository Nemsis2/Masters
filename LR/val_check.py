import numpy as np
from data_grab import extract_inner_fold_data, extract_dev_data

for outer in range(3):
    for inner in range(4):
        copy = 0
        k_fold_path = f'../../data/tb/combo/new/180_melspec_fold_{outer}.pkl'
        inner_data, inner_labels = extract_inner_fold_data(k_fold_path, inner)

        dev_data, dev_labels, dev_names = extract_dev_data(k_fold_path, inner)

        for sample in inner_data:
            for dev_sample in dev_data:
                if np.array_equal(sample, dev_sample):
                    copy += 1

        print(f'Number of copies found between inner train {inner} and dev data {inner} = {copy}')