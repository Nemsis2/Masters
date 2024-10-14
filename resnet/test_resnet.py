# libraries
import torch as th

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *
from get_best_features import *
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 128
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4
EPOCHS = 16

# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


def dev_resnet(feature_type, n_feature, model_type):
    """
    Description:
    ---------

    Inputs:
    ---------

    Outputs:
    --------

    """
    total_auc = 0
    count  = 0
    valid_folds = []
    for outer in range(NUM_OUTER_FOLDS):
        valid_folds.append([])
        for inner in range(NUM_INNER_FOLDS):
            # get the dev model
            model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/dev/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            model = pickle.load(open(model_path, 'rb')) # load in the model
            
            # grab the testing data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_dev_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)
            results = test(data, model.model, lengths) # do a forward pass through the models
            
            results = np.vstack(results)
            labels = np.vstack(labels)
            results, labels = gather_results(results, labels, names)
            auc = roc_auc_score(labels, results)
            if auc > 0.5:
                total_auc += auc
                count +=1
                valid_folds[outer].append(1)
            else:
                valid_folds[outer].append(0)
    
    print(total_auc/count)
    print(valid_folds)
    return valid_folds, total_auc/count


def dev_resnet_fss(feature_type, n_feature, model_type, fss_features):
    """
    Description:
    ---------

    Inputs:
    ---------

    Outputs:
    --------

    """
    total_auc = 0
    count  = 0
    valid_folds = []
    for outer in range(NUM_OUTER_FOLDS):
        valid_folds.append([])
        for inner in range(NUM_INNER_FOLDS):
            # get the fss model
            model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/fss_on_resnet/{model_type}_{feature_type}_{n_feature}_fss_{fss_features}_outer_fold_{outer}_inner_fold_{inner}'
            model = pickle.load(open(model_path, 'rb')) # load in the model
            
            # grab the testing data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            data, labels, names = extract_dev_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)

            # select only the relevant features
            feature_path = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
            if feature_type == 'mfcc':
                selected_features = outer_fss(outer, n_feature*3, fss_features, feature_path)
            else:
                selected_features = outer_fss(outer, n_feature, fss_features, feature_path)

            for batch in range(len(data)):
                chosen_features = []
                for feature in selected_features:
                    chosen_features.append(np.asarray(data[batch][:,:,feature]))
                data[batch] = th.as_tensor(np.stack(chosen_features, -1))

            results = test(data, model.model, lengths) # do a forward pass through the models
            
            results = np.vstack(results)
            labels = np.vstack(labels)
            results, labels = gather_results(results, labels, names)
            auc = roc_auc_score(labels, results)
            if auc > 0.5:
                total_auc += auc
                count +=1
                valid_folds[outer].append(1)
            else:
                valid_folds[outer].append(0)
    
    return valid_folds, total_auc/count


def test_em_resnet(feature_type, n_feature, model_type, valid_folds):
    """
    Description:
    ---------
    Calculates the auc, sens and spec for a LR model on the given features.
    
    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.


    Outputs:
    --------
    auc: average auc over all outer folds.

    sens: average sensitivity over all outer folds.

    spec: average specificity over all outer folds.
    """
    count = 0
    performance_metrics = np.zeros(5)

    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        if sum(valid_folds[outer]) != 0:
            count +=1

            for inner in range(NUM_INNER_FOLDS):
                if valid_folds[outer][inner] == 1:
                    # get the testing models
                    model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/dev/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
                    models.append(pickle.load(open(model_path, 'rb'))) # load in the model

            # grab the testing data
            k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
            data, labels, names = extract_test_data(k_fold_path)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)

            results = []
            for model in models:
                results.append(test(data, model.model, lengths)) # do a forward pass through the models

            for i in range(len(results)):
                results[i] = np.vstack(results[i])
            
            labels = np.vstack(labels)
            output = []
            for i in range(len(results)):
                new_results, new_labels = gather_results(results[i], labels, names)
                output.append(new_results)

            # average over models and calculate auc
            results = sum(output)/len(output)

            fpr, tpr, thresholds = roc_curve(new_labels, results) 
            roc_auc = roc_auc_score(new_labels, results)

            if outer == 0:
                plt_results = results
                plt_labels = new_labels
            else:
                plt_results = [*plt_results, *results]
                plt_labels = [*plt_labels, *new_labels]

            performance_metrics += calculate_metrics(new_labels, results)

    performance_metrics = performance_metrics/count

    fpr, tpr, thresholds = roc_curve(plt_labels, plt_results) 
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()  
    plt.plot(fpr, tpr, label=f'Outer fold {outer} (area = {round(performance_metrics[0],2)})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for ResNet-18 in EM configuration using LFB features')
    plt.legend()

    return performance_metrics


def test_sm_resnet(feature_type, n_feature, model_type):
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):
        
        dev_auc = 0
        for inner in range(NUM_INNER_FOLDS):
            # get the dev model
            model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/dev/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            model = pickle.load(open(model_path, 'rb')) # load in the model

            # grab the dev data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl' 
            data, labels, names = extract_dev_data(k_fold_path, inner)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)

            # assess model performance
            results = test(data, model.model, lengths)
            results = np.vstack(results)
            labels = np.vstack(labels)
            results, labels = gather_results(results, labels, names)
            inner_auc = roc_auc_score(labels, results)

            # check if current dev auc is better than previous best
            if inner_auc > dev_auc:
                dev_auc = inner_auc
                best_inner = inner

        # get the best performing dev model
        model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/dev/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{best_inner}'
        model = pickle.load(open(model_path, 'rb')) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)
        if feature_type == 'mfcc':
            data = normalize_mfcc(data)
        data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)

        # test and gather results
        results = test(data, model.model, lengths) # do a forward pass through the models
        results = np.vstack(results)
        labels = np.vstack(labels)
        results, labels = gather_results(results, labels, names)

        if outer == 0:
            plt_results = results
            plt_labels = labels
        else:
            plt_results = [*plt_results, *results]
            plt_labels = [*plt_labels, *labels]

        performance_metrics += calculate_metrics(labels, results)
    
    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

    fpr, tpr, thresholds = roc_curve(plt_labels, plt_results) 
    roc_auc = roc_auc_score(plt_labels, plt_results)
    plt.plot(fpr, tpr, label='SM ROC curve using MFB features (area = %0.2f)' % roc_auc)
    plt.legend()
    return performance_metrics


def test_ts_resnet(feature_type, n_feature, model_type):
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):
        # get the ts model
        model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/ts/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}'
        model = pickle.load(open(model_path, 'rb')) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)
        if feature_type == 'mfcc':
            data = normalize_mfcc(data)
        data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)

        # test and gather results
        results = test(data, model.model, lengths) # do a forward pass through the models
        results = np.vstack(results)
        labels = np.vstack(labels)
        results, labels = gather_results(results, labels, names)

        if outer == 0:
            plt_results = results
            plt_labels = labels
        else:
            plt_results = [*plt_results, *results]
            plt_labels = [*plt_labels, *labels]

        performance_metrics += calculate_metrics(labels, results)
    
    performance_metrics = performance_metrics/NUM_OUTER_FOLDS


    fpr, tpr, thresholds = roc_curve(plt_labels, plt_results) 
    roc_auc = auc(fpr, tpr)
    print(roc_auc)
    plt.figure()  
    plt.plot(fpr, tpr, label=f'Outer fold {outer} (area = {round(performance_metrics[0],2)})')
    plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for ResNet-10 in TS configuration using LFB features')
    plt.legend()
    return performance_metrics


def test_fss_resnet(feature_type, n_feature, fss_feature, model_type, valid_folds):
    count = 0
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        if sum(valid_folds[outer]) != 0:
            count +=1
            for inner in range(NUM_INNER_FOLDS):
                if valid_folds[outer][inner] == 1:
                    # get the testing models
                    model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/fss_on_resnet/{model_type}_{feature_type}_{n_feature}_fss_{fss_feature}_outer_fold_{outer}_inner_fold_{inner}'
                    models.append(pickle.load(open(model_path, 'rb'))) # load in the model

            # grab the testing data
            k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
            data, labels, names = extract_test_data(k_fold_path)
            if feature_type == 'mfcc':
                data = normalize_mfcc(data)
            data, labels, lengths, names = create_batches(data, labels, names, BATCH_SIZE)

            # select only the relevant features
            feature_path = f'../../models/tb/resnet/resnet_18/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
            if feature_type == 'mfcc':
                selected_features = outer_fss(outer, n_feature*3, fss_feature, feature_path)
            else:
                selected_features = outer_fss(outer, n_feature, fss_feature, feature_path)

            for batch in range(len(data)):
                chosen_features = []
                for feature in selected_features:
                    chosen_features.append(np.asarray(data[batch][:,:,feature]))
                data[batch] = th.as_tensor(np.stack(chosen_features, -1))

            results = []
            for model in models:
                results.append(test(data, model.model, lengths)) # do a forward pass through the models

            for i in range(len(results)):
                results[i] = np.vstack(results[i])

            labels = np.vstack(labels)
            output = []
            for i in range(len(results)):
                new_results, new_labels = gather_results(results[i], labels, names)
                output.append(new_results)

            results = sum(output)/4

            fpr, tpr, thresholds = roc_curve(new_labels, results) 
            roc_auc = roc_auc_score(new_labels, results)

            if outer == 0:
                plt.figure()  
                plt.plot(fpr, tpr, label=f'Outer fold {outer} (area = {round(roc_auc,2)})')
                plt.plot([0, 1], [0, 1], 'k--', label='No Skill')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title(f'ROC Curve for ResNet-18 in EM configuration using LFB features')
                plt.legend()
            else:
                plt.plot(fpr, tpr, label=f'Outer fold {outer} (area = {round(roc_auc,2)})')
                plt.legend()

            performance_metrics += calculate_metrics(new_labels, results)
    

    performance_metrics = performance_metrics/count

    return performance_metrics


def main():
    # for feature_type in ['mfcc', 'melspec', 'lfb']:
    # #for feature_type in ['melspec']:
    #     if feature_type == 'mfcc':
    #         features = [13]
    #     elif feature_type == 'melspec':
    #         features = [80] 
    #     elif feature_type == 'lfb':
    #         features = [80] 
        
        #for model in ['resnet_18', 'resnet_10', 'resnet_6_2Deep', 'resnet_6_4Deep']:
        for model in ['resnet_10']:
            for feature_type in ['lfb']:
                if feature_type == 'mfcc':
                    features = [13]
                elif feature_type == 'melspec':
                    features = [80] 
                elif feature_type == 'lfb':
                    features = [180] 

                print(f'feature: {feature_type} and model: {model}')
                best_em ,best_sm, best_ts = [], [], []
                best_fss, em_dev_aucs, fss_dev_aucs = [],[], []
                for n_feature in features:
                    # if feature_type == 'lfb':
                    #     valid_folds, dev_auc = dev_resnet(feature_type, n_feature, model)
                    #     em_dev_aucs.append(dev_auc)
                    #     results = test_em_resnet(feature_type, n_feature, model, valid_folds)
                    #     best_em.append(results)

                    # results = test_sm_resnet(feature_type, n_feature, model)
                    # best_sm.append(results)

                    if feature_type == 'lfb':
                        results = test_ts_resnet(feature_type, n_feature, model)
                        best_ts.append(results)

                #for fraction_of_feature in [0.1, 0.2, 0.5]:
                # for fraction_of_feature in [0.2]:
                #     if feature_type == 'mfcc':
                #         valid_folds, dev_auc = dev_resnet_fss(feature_type, n_feature, model, int(fraction_of_feature*n_feature*3))
                #         results = test_fss_resnet(feature_type, n_feature, int(fraction_of_feature*n_feature*3), model, valid_folds)
                #         results = np.append(results, fraction_of_feature)
                #         best_fss.append(results)
                #         fss_dev_aucs.append(dev_auc)
                #     else:
                #         valid_folds, dev_auc = dev_resnet_fss(feature_type, n_feature, model, int(fraction_of_feature*n_feature))
                #         results = test_fss_resnet(feature_type, n_feature, int(fraction_of_feature*n_feature), model, valid_folds)
                #         results = np.append(results, fraction_of_feature)
                #         best_fss.append(results)
                #         fss_dev_aucs.append(dev_auc)

            plt.show()
            # print(f'Results for {model} using {feature_type}s:')
            # print(em_dev_aucs)
            # em_index = np.argmax(em_dev_aucs,0)
            # print(em_index)
            # print(f'EM: best_#_features: {features[em_index]}, AUC: {round(best_em[em_index][0],4)}, Sens: {round(best_em[em_index][1],4)}, Spec: {round(best_em[em_index][2],4)}')
            # print(f'{features[em_index]} & {round(best_em[em_index][0],4)} & {round(best_em[em_index][1],4)} & {round(best_em[em_index][2],4)} & {round(best_em[em_index][3],4)} & {round(best_em[em_index][4],4)}')

            # print(f'SM: best_#_features: {features[em_index]}, AUC: {round(best_sm[em_index][0],4)}, Sens: {round(best_sm[em_index][1],4)}, Spec: {round(best_sm[em_index][2],4)}')
            # print(f'{features[em_index]} & {round(best_sm[em_index][0],4)} & {round(best_sm[em_index][1],4)} & {round(best_sm[em_index][2],4)} & {round(best_sm[em_index][3],4)} & {round(best_sm[em_index][4],4)}')

            # print(f'TS: best_#_features: {features[em_index]}, AUC: {round(best_ts[em_index][0],4)}, Sens: {round(best_ts[em_index][1],4)}, Spec: {round(best_ts[em_index][2],4)}')
            # print(f'{features[em_index]} & {round(best_ts[em_index][0],4)} & {round(best_ts[em_index][1],4)} & {round(best_ts[em_index][2],4)} & {round(best_ts[em_index][3],4)} & {round(best_ts[em_index][4],4)}')

            # fss_index = np.argmax(fss_dev_aucs,0)
            # print(fss_index)
            # print(f'FSS: best_#_features: {features[int(np.ceil(fss_index/3)-1)]}, AUC: {round(best_fss[fss_index][0],4)}, Sens {best_fss[fss_index][1]}, Spec: {round(best_fss[fss_index][2],4)}, features: {round(best_fss[fss_index][5],4)}')
            # print(f'{features[int(np.ceil(fss_index/3)-1)]} & {round(best_fss[fss_index][5],4)} & {round(best_fss[fss_index][0],4)} & {round(best_fss[fss_index][1],4)} & {round(best_fss[fss_index][2],4)} & {round(best_fss[fss_index][3],4)} & {round(best_fss[fss_index][4],4)}')

if __name__ == "__main__":
    main()