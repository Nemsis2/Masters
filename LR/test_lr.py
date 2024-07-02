# libraries
import torch as th
import pickle

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from get_best_features import *
from lr_model_scripts import *

# declare global variables
# set hyperpaperameters
BATCH_SIZE = 64
NUM_OUTER_FOLDS = 3
NUM_INNER_FOLDS = 4


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)



def test_lr(feature_type, n_feature, model_path, data_path):
    auc, sens, spec, oracle_sens, oracle_spec = 0, 0, 0, 0, 0
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []
        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            inner_model_path = f'{model_path}lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(inner_model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'{data_path}{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = load_test_data(k_fold_path, feature_type)

        # do a forward pass through the models
        results = []
        for model in models:
            results.append(model.predict_proba(data))

        # gather the results over all 4 models and average by patient
        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        # average over models and calculate auc
        results = sum(output)/4
        auc += roc_auc_score(new_labels, results)
        
        # get eer, and oracle thresholds
        eer_threshold = get_EER_threshold(new_labels, results)
        sens_threshold, spec_threshold = get_oracle_thresholds(new_labels, results)

        # eer sens and spec
        eer_results = (np.array(results)>eer_threshold).astype(np.int8)
        inner_eer_sens, inner_eer_spec = calculate_sens_spec(new_labels, eer_results)

        # oracle sens and spec
        # using locked sens = 0.9
        sens_results = (np.array(results)>sens_threshold).astype(np.int8)
        _, inner_oracle_spec = calculate_sens_spec(new_labels, sens_results)

        # using locked spec = 0.7
        spec_results = (np.array(results)>spec_threshold).astype(np.int8)
        inner_oracle_sens, _ = calculate_sens_spec(new_labels, spec_results)

        # eer sens and spec
        sens += inner_eer_sens
        spec += inner_eer_spec
        
        # oracle sens and spec
        oracle_sens += inner_oracle_sens
        oracle_spec += inner_oracle_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS, oracle_sens/NUM_OUTER_FOLDS, oracle_spec/NUM_OUTER_FOLDS


def test_lr_sm(feature_type, n_feature):
    auc, sens, spec, oracle_sens, oracle_spec = 0, 0, 0, 0, 0
    for outer in range(NUM_OUTER_FOLDS):
        k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl' 
        max_auc, best_model = 0, 0
        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            model = pickle.load(open(model_path, 'rb')) # load in the model
            data, labels, names = load_dev_data(k_fold_path, feature_type, inner)
            results, labels = gather_results(model.predict_proba(data), labels, names) # do a forward pass through the model
            inner_auc = roc_auc_score(labels, results)
            if inner_auc > max_auc:
                max_auc = inner_auc
                best_model = inner

            
        model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{best_model}'
        model = pickle.load(open(model_path, 'rb')) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = load_test_data(k_fold_path, feature_type)

        results = (model.predict_proba(data)) # do a forward pass through the models
        results, labels = gather_results(results, labels, names)
        inner_auc = roc_auc_score(labels, results)
        
        # get the thresholds on the test set
        eer_threshold = get_EER_threshold(labels, results)
        sens_threshold, spec_threshold = get_oracle_thresholds(labels, results)
        
        # eer sens and spec
        eer_results = (np.array(results)>eer_threshold).astype(np.int8)
        inner_eer_sens, inner_eer_spec = calculate_sens_spec(labels, eer_results)

        # oracle sens and spec
        # using locked sens=0.9
        sens_results = (np.array(results)>sens_threshold).astype(np.int8)
        _, inner_oracle_spec = calculate_sens_spec(labels, sens_results)

        # using locked spec=0.7
        spec_results = (np.array(results)>spec_threshold).astype(np.int8)
        inner_oracle_sens, _ = calculate_sens_spec(labels, spec_results)
        
        # add to the total auc, sens and spec
        auc += inner_auc
        sens += inner_eer_sens
        spec += inner_eer_spec

        # oracle sens and spec
        oracle_sens += inner_oracle_sens
        oracle_spec += inner_oracle_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS, oracle_sens/NUM_OUTER_FOLDS, oracle_spec/NUM_OUTER_FOLDS


def test_lr_multi_feature():
    auc = 0
    for outer in range(NUM_OUTER_FOLDS):
        results = []
        for feature_type in ['mfcc', 'melspec', 'lfb']:
            if feature_type == 'mfcc':
                n_feature = '13'
            if feature_type == 'melspec' or feature_type == 'lfb':
                n_feature = '180'
            
            # grab all models to be tested for that outer fold
            models = []
            for inner in range(NUM_INNER_FOLDS):
                # get the testing models
                model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
                models.append(pickle.load(open(model_path, 'rb'))) # load in the model

            # grab the testing data
            k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
            data, labels, names = load_test_data(k_fold_path, feature_type)
        
            for model in models:
                results.append(model.predict_proba(data)) # do a forward pass through the models

        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        results = sum(output)/4
        inner_auc = roc_auc_score(new_labels, results)
        
        # add to the total auc, sens and spec
        auc += inner_auc

    return auc/NUM_OUTER_FOLDS


def test_lr_fss(feature_type, n_feature, fss_features):
    feature_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/docs/'

    auc, sens, spec, oracle_sens, oracle_spec = 0,0,0,0,0
    for outer in range(NUM_OUTER_FOLDS):
        models = []

        if feature_type == 'mfcc':
            selected_features = outer_fss(outer, n_feature*3, fss_features, feature_path)
            #selected_features = dataset_fss(n_feature*3, fss_features, feature_path)
        else:
            selected_features = outer_fss(outer, n_feature, fss_features, feature_path)
            #selected_features = dataset_fss(n_feature, fss_features, feature_path)
        
        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/lr_{feature_type}_{n_feature}_fss_{fss_features}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names = load_test_data(k_fold_path, feature_type)

        # select only the relevant features
        chosen_features = []
        for i in range(len(selected_features)):
            chosen_features.append(np.asarray(data[:,selected_features[i]]))
        chosen_features = th.as_tensor(np.stack(chosen_features, -1))

        results = []
        for model in models:
            results.append(model.predict_proba(chosen_features)) # do a forward pass through the models

        output = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            output.append(new_results)

        results = sum(output)/4
        inner_auc = roc_auc_score(new_labels, results)

        # get the thresholds on the test set
        eer_threshold = get_EER_threshold(new_labels, results)
        sens_threshold, spec_threshold = get_oracle_thresholds(new_labels, results)

        #eer sens and spec
        eer_results = (np.array(results)>eer_threshold).astype(np.int8)
        inner_sens, inner_spec = calculate_sens_spec(new_labels, eer_results)

        # oracle sens and spec
        # using locked sens=0.9
        sens_results = (np.array(results)>sens_threshold).astype(np.int8)
        _, inner_oracle_spec = calculate_sens_spec(new_labels, sens_results)

        # using locked spec=0.7
        spec_results = (np.array(results)>spec_threshold).astype(np.int8)
        inner_oracle_sens, _ = calculate_sens_spec(new_labels, spec_results)
        
        # add to the total auc, sens and spec
        auc += inner_auc
        sens += inner_sens
        spec += inner_spec

        # oracle sens and spec
        oracle_sens += inner_oracle_sens
        oracle_spec += inner_oracle_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS, oracle_sens/NUM_OUTER_FOLDS, oracle_spec/NUM_OUTER_FOLDS


def test_lr_tb_index(feature_type, n_feature):
    auc, sens, spec, oracle_sens, oracle_spec = 0, 0, 0, 0, 0
    for outer in range(NUM_OUTER_FOLDS):
        # grab all models to be tested for that outer fold
        models = []

        for inner in range(NUM_INNER_FOLDS):
            # get the testing models
            model_path = f'../../models/tb/lr_per_frame/{feature_type}/{n_feature}_{feature_type}/dev/lr_{feature_type}_{n_feature}_outer_fold_{outer}_inner_fold_{inner}'
            models.append(pickle.load(open(model_path, 'rb'))) # load in the model

        # grab the testing data
        k_fold_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl' 
        data, labels, names, cough_labels = load_test_per_frame_data(k_fold_path, feature_type)

        results = []
        for model in models:
            results.append(model.predict_proba(data)) # do a forward pass through the models

        # calculate tbi1
        # get the average prediction per cough
        cough_averages = []
        for i in range(len(results)):
            unq,ids,count = np.unique(cough_labels,return_inverse=True,return_counts=True)
            out = np.column_stack((unq, np.bincount(ids,results[i][:,1])/count, np.bincount(ids,labels)/count))
            cough_averages.append(out[:,1])
            new_labels = out[:,2]

        # get average cough prediction per model
        cough_averages = sum(cough_averages)/4

        # get eer, and oracle thresholds
        eer_threshold = get_EER_threshold(new_labels, cough_averages)
        sens_threshold, spec_threshold = get_oracle_thresholds(new_labels, cough_averages)

        # get the average prediction per patient for eer
        eer_cough_averages = (np.array(cough_averages)>eer_threshold).astype(np.int8)
        eer_tbi1, eer_labels = tbi1(names, eer_cough_averages, new_labels)

        # get the average prediction per patient for oracle sens
        oracle_sens_cough_averages = (np.array(cough_averages)>sens_threshold).astype(np.int8)
        sens_tbi1, sens_labels = tbi1(names, oracle_sens_cough_averages, new_labels)

        # get the average prediction per patient for oracle spec
        oracle_spec_cough_averages = (np.array(cough_averages)>spec_threshold).astype(np.int8)
        spec_tbi1, spec_labels = tbi1(names, oracle_spec_cough_averages, new_labels)

        # calculate tbi2
        # get the average prediction per patient over all frames and models
        data, labels, names = load_test_per_frame_data_tbi2(k_fold_path, feature_type)
        results = []
        for model in models:
            results.append(model.predict_proba(data)) # do a forward pass through the models
        
        tbi2 = []
        for i in range(len(results)):
            new_results, new_labels = gather_results(results[i], labels, names)
            tbi2.append(new_results)
        tbi2 = sum(tbi2)/4

        # get eer, and oracle thresholds
        eer_threshold = get_EER_threshold(new_labels, tbi2)
        sens_threshold, spec_threshold = get_oracle_thresholds(new_labels, tbi2)

        #if tbi1 > 0.5 or if tbi2 > threshold predict the patient as positive
        inner_auc = roc_auc_score(new_labels, eer_tbi1)
        eer_tbi1 = (np.array(eer_tbi1)>0.50000000).astype(np.int8)
        eer_tbi2 = (np.array(tbi2)>eer_threshold).astype(np.int8)
        eer_predictions = eer_tbi1*eer_tbi2
        inner_sens, inner_spec = calculate_sens_spec(new_labels, eer_predictions)

        sens_tbi1 = (np.array(eer_tbi1)>0.50000000).astype(np.int8)
        sens_tbi2 = (np.array(tbi2)>sens_threshold).astype(np.int8)
        sens_predictions = sens_tbi1*sens_tbi2
        _, inner_oracle_spec = calculate_sens_spec(new_labels, sens_predictions)

        spec_tbi1 = (np.array(eer_tbi1)>0.50000000).astype(np.int8)
        spec_tbi2 = (np.array(tbi2)>spec_threshold).astype(np.int8)
        spec_predictions = spec_tbi1*spec_tbi2
        inner_oracle_sens, _ = calculate_sens_spec(new_labels, spec_predictions)
        
        # add to the total auc, sens and spec
        auc += inner_auc
        sens += inner_sens
        spec += inner_spec
        oracle_sens += inner_oracle_sens
        oracle_spec += inner_oracle_spec

    return auc/NUM_OUTER_FOLDS, sens/NUM_OUTER_FOLDS, spec/NUM_OUTER_FOLDS, oracle_sens/NUM_OUTER_FOLDS, oracle_spec/NUM_OUTER_FOLDS



def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180]

        for n_feature in features:
            # test the em setup
            # data_path = f'../../data/tb/combo/new/test/test_dataset_'
            # val_data = f'../../data/tb/combo/new/'
            # model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/dev/'
            # auc, sens, spec, oracle_sens, oracle_spec = test_lr(feature_type, n_feature, model_path, data_path)

            # print(f'AUC for em {n_feature}_{feature_type}: {auc}')
            # print(f'Sens for em {n_feature}_{feature_type}: {sens}')
            # print(f'Spec for em {n_feature}_{feature_type}: {spec}')
            # print(f'Oracle sens for em {n_feature}_{feature_type}: {oracle_sens}')
            # print(f'Oracle spec for em {n_feature}_{feature_type}: {oracle_spec}')
            # print(f'{feature_type} & {round(n_feature,4)} & {round(auc,4)} & {round(sens,4)} & {round(spec,4)} & {round(oracle_sens,4)} & {round(oracle_spec,4)}')


            # test the tb_index setup
            # auc, sens, spec, oracle_sens, oracle_spec = test_lr_tb_index(feature_type, n_feature)

            # print(f'AUC for tb_index {n_feature}_{feature_type}: {auc}')
            # print(f'Sens for tb_index {n_feature}_{feature_type}: {sens}')
            # print(f'Spec for tb_index {n_feature}_{feature_type}: {spec}')
            # print(f'Oracle sens for tb_index {n_feature}_{feature_type}: {oracle_sens}')
            # print(f'Oracle spec for tb_index {n_feature}_{feature_type}: {oracle_spec}')
            # print(f'{feature_type} & {round(n_feature,4)} & {round(auc,4)} & {round(sens,4)} & {round(spec,4)} & {round(oracle_sens,4)} & {round(oracle_spec,4)}')

            # test the sm setup
            # auc, sens, spec, oracle_sens, oracle_spec = test_lr_sm(feature_type, n_feature)

            # print(f'AUC for sm {n_feature}_{feature_type}: {auc}')
            # print(f'Sens for sm {n_feature}_{feature_type}: {sens}')
            # print(f'Spec for sm {n_feature}_{feature_type}: {spec}')
            # print(f'Oracle sens for SMOTE sm {n_feature}_{feature_type}: {oracle_sens}')
            # print(f'Oracle spec for SMOTE sm {n_feature}_{feature_type}: {oracle_spec}')
            # print(f'{feature_type} & {round(n_feature,4)} & {round(auc,4)} & {round(sens,4)} & {round(spec,4)} & {round(oracle_sens,4)} & {round(oracle_spec,4)}')

            

            # # # test the SMOTE setup
            # data_path = f'../../data/tb/combo/new/test/test_dataset_'
            # model_path = f'../../models/tb/SMOTE_lr/{feature_type}/{n_feature}_{feature_type}/dev/'
            # auc, sens, spec, oracle_sens, oracle_spec = test_lr(feature_type, n_feature, model_path, data_path)

            # print(f'AUC for SMOTE em {n_feature}_{feature_type}: {auc}')
            # print(f'Sens for SMOTE em {n_feature}_{feature_type}: {sens}')
            # print(f'Spec for SMOTE em {n_feature}_{feature_type}: {spec}')
            # print(f'Oracle sens for SMOTE em {n_feature}_{feature_type}: {oracle_sens}')
            # print(f'Oracle spec for SMOTE em {n_feature}_{feature_type}: {oracle_spec}')
            # print(f'{feature_type} & {round(n_feature,4)} & {round(auc,4)} & {round(sens,4)} & {round(spec,4)} & {round(oracle_sens,4)} & {round(oracle_spec,4)}')

            # # test the frame_skip setup
            # data_path = f'../../data/tb/frame_skip/test/test_dataset_'
            # model_path = f'../../models/tb/lr_frame_skip/{feature_type}/{n_feature}_{feature_type}/dev/'
            # auc, sens, spec, oracle_sens, oracle_spec = test_lr(feature_type, n_feature, model_path, data_path)

            # print(f'AUC for frame_skip em {n_feature}_{feature_type}: {auc}')
            # print(f'Sens for frame_skip em {n_feature}_{feature_type}: {sens}')
            # print(f'Spec for frame_skip em {n_feature}_{feature_type}: {spec}')
            # print(f'Oracle sens for frame_skip em {n_feature}_{feature_type}: {oracle_sens}')
            # print(f'Oracle spec for frame_skip em {n_feature}_{feature_type}: {oracle_spec}')
            # print(f'{feature_type} & {round(n_feature,4)} & {round(auc,4)} & {round(sens,4)} & {round(spec,4)} & {round(oracle_sens,4)} & {round(oracle_spec,4)}')


            
            for fraction_of_feature in [0.1, 0.2, 0.5]:
                if feature_type == 'mfcc':
                    auc, sens, spec, oracle_sens, oracle_spec = test_lr_fss(feature_type, n_feature, int(n_feature*fraction_of_feature*3))
                else:
                    auc, sens, spec, oracle_sens, oracle_spec = test_lr_fss(feature_type, n_feature, int(n_feature*fraction_of_feature))

                print(f'AUC for {n_feature}_{feature_type} with {int(fraction_of_feature*n_feature)}: {auc}')
                print(f'Sens for {n_feature}_{feature_type} with {int(fraction_of_feature*n_feature)}: {sens}')
                print(f'Spec for {n_feature}_{feature_type} with {int(fraction_of_feature*n_feature)}: {spec}')
                print(f'Oracle sens for frame_skip fss {n_feature}_{feature_type}: {oracle_sens}')
                print(f'Oracle spec for frame_skip fss {n_feature}_{feature_type}: {oracle_spec}')
                print(f'{feature_type} & {round(n_feature,4)} & {round(auc,4)} & {round(sens,4)} & {round(spec,4)} & {round(oracle_sens,4)} & {round(oracle_spec,4)}')
            


    auc = test_lr_multi_feature()

    print(f'AUC for multi feature: {auc}')


if __name__ == "__main__":
    main()