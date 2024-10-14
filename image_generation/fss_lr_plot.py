# libraries
import os
import numpy as np 
import matplotlib.pyplot as plt


def outer_auc_fss_average(total_features, feature_path):
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

    return  outer_aucs

lr_mfcc, lr_lfb, lr_mfb, resnet_mfcc, resnet_lfb, resnet_mfb = [], [], [], [], [], []
for feature_type in ['mfcc', 'melspec', 'lfb']:
    # get lr average
    if feature_type == 'mfcc':
        n_feature = 39
        feature_path  = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        lr_mfcc = outer_auc_fss_average(n_feature, feature_path)
    
    elif feature_type == 'melspec':
        n_feature = 180
        feature_path  = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        lr_mfb = outer_auc_fss_average(n_feature, feature_path)
    
    elif feature_type == 'lfb':
        n_feature = 128
        feature_path  = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        lr_lfb = outer_auc_fss_average(n_feature, feature_path)

    # get resnet average
    if feature_type == 'mfcc':
        n_feature = 13
        feature_path  = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        resnet_mfcc = outer_auc_fss_average(n_feature, feature_path)
    
    elif feature_type == 'melspec':
        n_feature = 80
        feature_path  = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        resnet_mfb = outer_auc_fss_average(n_feature, feature_path)
    
    elif feature_type == 'lfb':
        n_feature = 80
        feature_path  = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        resnet_lfb = outer_auc_fss_average(n_feature, feature_path)

plt.figure()
plt.xlabel('Number of Features')
plt.ylabel('Area Under Curve (AUC)')
plt.grid()
num_features = np.arange(len(lr_mfcc))
plt.plot(num_features, lr_mfcc, label='LR MFCC')

num_features = np.arange(len(lr_mfb))
plt.plot(num_features, lr_mfb, label='LR MFB')

num_features = np.arange(len(lr_lfb))
plt.plot(num_features, lr_lfb, label='LR LFB')

num_features = np.arange(len(resnet_mfcc))
a, b = np.polyfit(num_features, resnet_mfcc,1)
plt.plot(num_features, resnet_mfcc, color='red', label='ResNet MFCC')
plt.plot(num_features, a*num_features+b, color='red', label='Line of Best Fit ResNet MFCC')

num_features = np.arange(len(resnet_mfb))
a, b = np.polyfit(num_features, resnet_mfb,1)
plt.plot(num_features, resnet_mfb, color='purple', label='ResNet MFB')
plt.plot(num_features, a*num_features+b, color='purple', label='Line of Best Fit ResNet MFB')

a, b = np.polyfit(num_features-1, resnet_lfb,1)
num_features = np.arange(len(resnet_lfb))
a, b = np.polyfit(num_features, resnet_lfb,1)
plt.plot(num_features, resnet_lfb, color='green', label='ResNet LFB')
plt.plot(num_features, a*num_features+b, color='green', label='Line of Best Fit ResNet LFB')

plt.legend()
plt.show()
