# libraries
import os


# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import * 
from model_scripts import *

"""
date: 23/10/2023 

author: Michael Knight
"""

# declare global variables

# set paths
K_FOLD_PATH = "../data/tb/combo/multi_folds/"
MODEL_PATH = "../models/tb/lr/"




def inner_fss(inner, outer, num_features):
    """
    Uses previously generated SFS results to determine the highest "scoring" features
    as selected by 5 different models.


    Parameters:
    -----------
        inner(int) : the inner fold to be considered.

        outer(int): the outer fold to be considered.

        num_features(int) : the number of top features to be selected. Maximum of 180

    Returns:
    --------
        selected_features(list) : list of selected features with length corresponding to the value
        of num_features e.g. if num_features = 3, selected_features = [28, 64, 32]

    """
    # Check input params are valid
    if num_features > 180:
        print("Number of requested features exceds total number of features")
        return
    
    if inner > 4:
        print("Inner fold", inner, "does not exist")
        return
    
    if outer > 3:
        print("Outer fold", outer, "does not exist")
        return
    
    #find files
    folder_names = os.listdir(MODEL_PATH + "FSS/")
    folder_names.sort()
    fold_feature = np.zeros(180)
    selected_features = []

    for folder_name in folder_names:
        best_features = []
        file_name = MODEL_PATH + "FSS/" + folder_name + "/features_outer_" + str(outer) + "_inner_" + str(inner) + ".txt"
        with open(file_name, 'r') as f:
            for line in f:
                best_features.append(line.split('\n')[0])

        for i in range(len(best_features)):
            fold_feature[int(best_features[i])] += i

    sorted_list = sorted(fold_feature)

    #find top 50
    count = 0
    for i in range(num_features):
        while sorted_list[i] != fold_feature[count]:
            count += 1
        selected_features.append(count)
        fold_feature[count] = 9999
        count = 0
    
    return selected_features



def outer_fss(outer, num_features):
    """
    Uses previously generated SFS results to determine the highest "scoring" features
    as selected by 5 different models across an outer fold.


    Parameters:
    -----------
        outer(int): the outer fold to be considered. can only be 1, 2 or 3

        num_features(int) : the number of top features to be selected. Maximum of 180

    Returns:
    --------
        selected_features(list) : list of selected features with length corresponding to the value
        of num_features e.g. if num_features = 3, selected_features = [28, 64, 32]
    """

    # Check input params are valid
    if num_features > 180:
        print("Number of requested features exceds total number of features")
        return
    
    if outer > 3:
        print("Outer fold", outer, "does not exist")
        return

    #find files
    folder_names = os.listdir(MODEL_PATH + "FSS/")
    folder_names.sort()
    fold_feature = np.zeros(180)
    selected_features = []

    for folder_name in folder_names:
        for inner in range(4):
            best_features = []
            file_name = MODEL_PATH + "FSS/" + folder_name + "/features_outer_" + str(outer) + "_inner_" + str(inner) + ".txt"
            with open(file_name, 'r') as f:
                for line in f:
                    best_features.append(line.split('\n')[0])

            for i in range(len(best_features)):
                fold_feature[int(best_features[i])] += i

    sorted_list = sorted(fold_feature)

    #find top num_features features
    count = 0
    for i in range(num_features):
        while sorted_list[i] != fold_feature[count]:
            count += 1
        selected_features.append(count)
        fold_feature[count] = 99999
        count = 0
    
    return selected_features



def dataset_fss(total_features, num_features, feature_path):
    """
    Uses previously generated SFS results to determine the highest "scoring" features across all outer folds.

    Parameters:
    -----------
        num_features(int) : the number of top features to be selected. Maximum of 180

    Returns:
    --------
        selected_features(list) : list of selected features with length corresponding to the value
        of num_features e.g. if num_features = 3, selected_features = [28, 64, 32]
    """

    #find files
    fold_feature = np.zeros(total_features)
    selected_features = []

    for outer in range(3):
        for inner in range(4):
            best_features = []
            file_name = f'{feature_path}features_outer_{outer}_inner_{inner}.txt'
            with open(file_name, 'r') as f:
                for line in f:
                    best_features.append(line.split('\n')[0])

            for i in range(len(best_features)):
                fold_feature[int(best_features[i])] += i

    sorted_list = sorted(fold_feature)

    #find top num_features features
    count = 0
    for i in range(num_features):
        while sorted_list[i] != fold_feature[count]:
            count += 1
        selected_features.append(count)
        fold_feature[count] = 99999
        count = 0
    
    return selected_features