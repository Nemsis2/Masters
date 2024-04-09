# libraries
import torch as th
import os
import pickle

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from get_best_features import *
from lr_model_scripts import *

n_feature = 180
feature_type = 'melspec'

for outer in range(3):
    for inner in range(4):
        outer_labels = []
        k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
        data, labels = load_inner_data(k_fold_path, feature_type, inner)
        data, dev_labels, names = load_dev_data(k_fold_path, feature_type, inner)

        total = len(labels) + len(dev_labels)
        total_positive = sum(labels) + sum(dev_labels)
        total_negative = total - total_positive
        print(f'{total_positive} & {total_negative} & {total}')