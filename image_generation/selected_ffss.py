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


def count_common_elements(lists):
    set1, set2, set3 = map(set, lists)
    
    common_elements = set1 & set2 & set3

    return len(common_elements)


def count_common_elements_two_lists(list1, list2):
    set1, set2 = map(set, [list1, list2])

    common_elements = set1 & set2

    return len(common_elements)


# get outer fss
feature_type = 'melspec'
n_feature = 80
lr_feature_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
resnet_feature_path = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'

# calculate how frequently similar features occur in the top 10%, 20% and 50% for the same model different outer folds
resnet_top_features = []
lr_top_features = []
count = 0
for fraction_of_feature in [0.1, 0.2, 0.5]:
    resnet_top_features.append([])
    lr_top_features.append([])
    for outer in range(3):
        resnet_top_features[count].append(outer_fss(outer, n_feature, int(fraction_of_feature*n_feature), resnet_feature_path)) 

        lr_top_features[count].append(outer_fss(outer, n_feature, int(fraction_of_feature*n_feature), lr_feature_path)) 

    
    print(f'% of features common for resnet between all outer folds for {fraction_of_feature*100}% of total features: {count_common_elements(resnet_top_features[count])/len(resnet_top_features[count][0])*100}%')
    print(f'% of features common for lr between all outer folds for {fraction_of_feature*100}% of total features: {count_common_elements(lr_top_features[count])/len(lr_top_features[count][0])*100}%')
    for outer in range(3):
        print(f'% of features common between resnet and lr for outer fold {outer}: {count_common_elements_two_lists(resnet_top_features[count][outer], lr_top_features[count][outer])/len(lr_top_features[count][outer])*100}%')

    count +=1