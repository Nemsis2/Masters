import numpy as np

def outer_fss(outer, total_features, num_features, feature_path):
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
    
    if outer > 3:
        print("Outer fold", outer, "does not exist")
        return


    fold_feature = np.zeros(total_features)
    selected_features = []

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



for feature_type in ['mfcc', 'melspec', 'lfb']:
    if feature_type == 'melspec' or feature_type == 'lfb':
        features = [80, 128, 180]
    elif feature_type == 'mfcc':
        features = [13, 26, 39]
        
    for n_feature in features:
        resnet_top_features = []
        lr_top_features = []
        resnet_feature_path = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        lr_feature_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        
        for outer in range(3):
            if feature_type == 'melspec' or feature_type == 'lfb':
                lr_top_features.append(outer_fss(outer, n_feature, n_feature, lr_feature_path)) 
            elif feature_type == 'mfcc':
                lr_top_features.append(outer_fss(outer, n_feature*3, n_feature*3, lr_feature_path)) 

            if n_feature == 80:
                resnet_top_features.append(outer_fss(outer, n_feature, n_feature, resnet_feature_path))
            elif n_feature == 13:
                resnet_top_features.append(outer_fss(outer, n_feature*3, n_feature*3, resnet_feature_path))

        print(f'{feature_type} with {n_feature} features for lr: {lr_top_features}')
        print(f'{feature_type} with {n_feature} features for resnet: {resnet_top_features}')