# libraries
import torch as th

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from model_scripts import *
from get_best_features import *

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


def test_em_resnet(feature_type, n_feature, model_type):
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
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/dev/{model_type}_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)
        if feature_type == 'mfcc':
            data = normalize_mfcc(data)
        data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

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
        results = sum(output)/4
        performance_metrics += calculate_metrics(new_labels, results)

    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

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
            data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

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
        data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

        # test and gather results
        results = test(data, model.model, lengths) # do a forward pass through the models
        results = np.vstack(results)
        labels = np.vstack(labels)
        results, labels = gather_results(results, labels, names)

        performance_metrics += calculate_metrics(labels, results)
    
    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

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
        data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

        # test and gather results
        results = test(data, model.model, lengths) # do a forward pass through the models
        results = np.vstack(results)
        labels = np.vstack(labels)
        results, labels = gather_results(results, labels, names)

        performance_metrics += calculate_metrics(labels, results)
    
    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

    return performance_metrics


def test_fss_resnet(feature_type, n_feature, fss_feature, model_type):
    performance_metrics = np.zeros(5)
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/resnet/{model_type}/{feature_type}/{n_feature}_{feature_type}/fss/{model_type}_{feature_type}_{n_feature}_fss_{fss_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = extract_test_data(k_fold_path)
        if feature_type == 'mfcc':
            data = normalize_mfcc(data)
        data, labels, lengths, names = create_batches(data, labels, names, 'image', BATCH_SIZE)

        # select only the relevant features
        feature_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'
        if feature_type == 'mfcc':
            selected_features = dataset_fss(n_feature*3, fss_feature, feature_path)
        else:
            selected_features = dataset_fss(n_feature, fss_feature, feature_path)

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
        performance_metrics += calculate_metrics(new_labels, results)
    
    performance_metrics = performance_metrics/NUM_OUTER_FOLDS

    return performance_metrics


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for model in ['resnet_18', 'resnet_10', 'resnet_6_2Deep', 'resnet_6_4Deep']:
            
            best_em ,best_sm, best_ts = [], [], []
            for n_feature in features:

                # results = test_em_resnet(feature_type, n_feature, model)
                # best_em.append(results)

                # results = test_sm_resnet(feature_type, n_feature, model)
                # best_sm.append(results)

                # results = test_ts_resnet(feature_type, n_feature, model)
                # best_ts.append(results)

                best_fss = []
                for fraction_of_feature in [0.1, 0.2, 0.5]:
                   if feature_type == 'mfcc':
                       results = test_fss_resnet(feature_type, n_feature, int(fraction_of_feature*n_feature*3), model)
                       results = np.append(results, fraction_of_feature)
                       best_fss.append(results)
                   else:
                       results = test_fss_resnet(feature_type, n_feature, int(fraction_of_feature*n_feature), model)
                       results = np.append(results, fraction_of_feature)
                       best_fss.append(results)

            print(f'Results for {model} using {feature_type}s:')
            # em_index = np.argmax(best_em,0)[0]
            # print(f'EM: best_#_features: {features[em_index]}, AUC: {round(best_em[em_index][0],4)}, Sens: {round(best_em[em_index][1],4)}, Spec: {round(best_em[em_index][2],4)}')
            # print(f'{features[em_index]} & {round(best_em[em_index][0],4)} & {round(best_em[em_index][1],4)} & {round(best_em[em_index][2],4)} & {round(best_em[em_index][3],4)} & {round(best_em[em_index][4],4)}')

            # sm_index = np.argmax(best_sm,0)[0]
            # print(f'SM: best_#_features: {features[sm_index]}, AUC: {round(best_sm[sm_index][0],4)}, Sens: {round(best_sm[sm_index][1],4)}, Spec: {round(best_sm[sm_index][2],4)}')
            # print(f'{features[sm_index]} & {round(best_sm[sm_index][0],4)} & {round(best_sm[sm_index][1],4)} & {round(best_sm[sm_index][2],4)} & {round(best_sm[sm_index][3],4)} & {round(best_sm[sm_index][4],4)}')

            # ts_index = np.argmax(best_ts,0)[0]
            # print(f'TS: best_#_features: {features[ts_index]}, AUC: {round(best_ts[ts_index][0],4)}, Sens: {round(best_ts[ts_index][1],4)}, Spec: {round(best_ts[ts_index][2],4)}')
            # print(f'{features[ts_index]} & {round(best_ts[ts_index][0],4)} & {round(best_ts[ts_index][1],4)} & {round(best_ts[ts_index][2],4)} & {round(best_ts[ts_index][3],4)} & {round(best_ts[ts_index][4],4)}')

            fss_index = np.argmax(best_fss,0)[0]
            print(f'FSS: best_#_features: {features[fss_index]}, AUC: {round(best_fss[fss_index][0],4)}, Sens {best_fss[fss_index][1]}, Spec: {round(best_fss[fss_index][2],4)}, features: {round(best_fss[fss_index][5],4)}')
            print(f'{features[fss_index]} & {round(best_fss[fss_index][5],4)} & {round(best_fss[fss_index][0],4)} & {round(best_fss[fss_index][1],4)} & {round(best_fss[fss_index][2],4)} & {round(best_fss[fss_index][3],4)} & {round(best_fss[fss_index][4],4)}')

if __name__ == "__main__":
    main()