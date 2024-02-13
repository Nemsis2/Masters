# libraries
import torch as th
import gc

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


def do_fss_lr(feature_type, n_feature):
    """
    Description:
    ---------

    Inputs:
    ---------
    feature_type: (string) type of the feature to be extracted. (mfcc, lfb or melspec)

    n_feature: (int) number of features.
    """
    model_path = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/'
    for outer in range(NUM_OUTER_FOLDS):
        print("train_outer_fold=", outer)
        for inner in range(NUM_INNER_FOLDS):
            print("train_inner_fold=", inner)
            feature_priority, auc_priority = [], []
            features = np.arange(0,n_feature)
            
            # load in training data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            train_data, train_labels = load_inner_data(k_fold_path, feature_type, inner)
            
            # load in dev data
            k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            dev_data, dev_labels, dev_names = load_dev_data(k_fold_path, feature_type, inner)

            # get the best params for this model and feature combination
            model, params = grid_search_lr(train_data, train_labels)
            print(params)

            # iterate feature-1 times
            while len(feature_priority) != n_feature:
                performance = []
                # Pass through all unselected features
                for feature in features:
                    # create new model
                    model = LogisticRegression(C = params['C'], l1_ratio = params['l1_ratio'],
                                                max_iter=1000000, solver='saga', penalty='elasticnet', multi_class = 'multinomial', n_jobs = -1)
                        
                    # get the chosen features    
                    chosen_features, chosen_features_dev = select_features(train_data, dev_data, feature_priority, feature)
                    
                    # train the model
                    model.fit(chosen_features, train_labels)
                    results = model.predict_proba(chosen_features_dev)
                    results, dev_labels_ = gather_results(results, dev_labels, dev_names)
                    auc = roc_auc_score(dev_labels_, results)
                    performance.append(auc)
                    print("new feature:", feature, "AUC:", auc)

                    # force delete loaded in model
                    del model
                    gc.collect()

                # select best performing feature from list
                best_feature = np.argmax(np.array(performance))
                print("Features array:", features)
                print("Best feature:", best_feature)
                print("Array selection", features[best_feature])
                
                feature_priority.append(str(features[best_feature]))
                auc_priority.append(str(performance[best_feature]))
                print("Best performing feature:", best_feature, "with an auc of:", performance[best_feature])

                # delete the chosen feature so it cannot be reselected
                features = np.delete(features, best_feature)
                
                #save current feature list
                file_name = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/features_outer_{outer}_inner_{inner}.txt'
                with open(file_name, 'w') as f:
                    for feature in feature_priority:
                        f.write("%s\n" % feature)

                # save current auc list
                file_name = f'../../models/tb/lr/{feature_type}/{n_feature}_{feature_type}/fss/auc_outer_{outer}_inner_{inner}.txt'
                with open(file_name, 'w') as f:
                    for auc in auc_priority:
                        f.write("%s\n" % auc)


def main():
    for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [13, 26, 39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80, 128, 180] 
        
        for n_feature in features:
            do_fss_lr(feature_type, n_feature)
        

if __name__ == "__main__":
    main()