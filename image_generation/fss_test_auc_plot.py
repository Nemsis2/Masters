# libraries
import os
import numpy as np 
import matplotlib.pyplot as plt


def dev_outer_auc_fss_average(feature_path):
    """
    Uses previously generated SFS inner dev AUCs to 
    calculate the average outer fold AUC when training the model.


    Parameters:
    -----------



    Returns:
    --------
        selected_features(list) : list of selected features with length corresponding to the value
        of num_features e.g. if num_features = 3, selected_features = [28, 64, 32]
    """
    
    outer_aucs = []
    for outer in range(3):
        aucs = []

        for inner in range(4):
            file_name = f'{feature_path}auc_outer_{outer}_inner_{inner}.txt'
            aucs.append([])
            with open(file_name, 'r') as f:
                for line in f:
                    aucs[inner].append(float(line.split('\n')[0]))

        aucs = sum(np.array(aucs))/len(aucs)
        outer_aucs.append(np.array(aucs))
    
    outer_aucs = sum(np.array(outer_aucs))/len(outer_aucs)

    return outer_aucs


def test_outer_auc_fss_average(feature_path):
    """
    Uses previously generated SFS inner dev AUCs to 
    calculate the average outer fold AUC when training the model.


    Parameters:
    -----------



    Returns:
    --------
        selected_features(list) : list of selected features with length corresponding to the value
        of num_features e.g. if num_features = 3, selected_features = [28, 64, 32]
    """
    
    outer_aucs = []
    for outer in range(3):
        file_name = f'{feature_path}test_auc_outer_{outer}.txt'
        outer_aucs.append([])
        with open(file_name, 'r') as f:
            for line in f:
                outer_aucs[outer].append(float(line.split('\n')[0]))
    
    outer_aucs = sum(np.array(outer_aucs))/len(outer_aucs)

    return outer_aucs

lr_dev_mfb, lr_test_mfb, resnet_dev_mfb, resnet_test_mfb = [], [], [], []
for feature_type in ['melspec']:
    n_feature = 80
    # get dev averages
    feature_path  = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
    lr_dev_mfb = dev_outer_auc_fss_average(feature_path)

    feature_path  = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
    resnet_dev_mfb = dev_outer_auc_fss_average(feature_path)

    # get test averages
    feature_path  = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
    lr_test_mfb = test_outer_auc_fss_average(feature_path)

    feature_path  = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
    resnet_test_mfb = test_outer_auc_fss_average(feature_path)

plt.figure()
plt.xlabel('Number of Features')
plt.ylabel('Area Under Curve (AUC)')
plt.grid()

num_features = np.arange(len(lr_dev_mfb))
plt.plot(num_features, lr_dev_mfb, label='LR Dev 80 MFB')

num_features = np.arange(len(lr_test_mfb))
plt.plot(num_features, lr_test_mfb, label='LR Test 80 MFB')

num_features = np.arange(len(resnet_dev_mfb))
a, b = np.polyfit(num_features, resnet_dev_mfb,1)
plt.plot(num_features, resnet_dev_mfb, color='red', label='ResNet Dev 80 MFB')
plt.plot(num_features, a*num_features+b, color='red', label='Line of Best Fit ResNet Dev MFB')

num_features = np.arange(len(resnet_test_mfb))
a, b = np.polyfit(num_features, resnet_test_mfb,1)
plt.plot(num_features, resnet_test_mfb, color='green', label='ResNet Test 80 MFB')
plt.plot(num_features, a*num_features+b, color='green', label='Line of Best Fit ResNet Test MFB')

plt.legend()
plt.show()
