# libraries
import torch as th

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from lstm_model_scripts import *


# Find gpu. If it cannot be found exit immediately
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)


def load_model(model_path):
      model = pickle.load(open(model_path, 'rb')) # load in the model
      th.manual_seed(model.seed) # set the seed to be the same as the one the model was generated on
      return model


def test_lstm(feature_type, n_feature):
    performance_metrics = np.zeros(5)
    for outer in range(1,11):
        for model_outer in range(3):
            outer_results = []
            for inner in range(4):
                model = load_model(f'../../../models/tb/lstm/{feature_type}/{n_feature}_{feature_type}/dev/lstm_{feature_type}_{n_feature}_outer_fold_{model_outer}_inner_fold_{inner}')
                results, labels, names = model.test(outer) # do a forward pass through the models
                results, labels = gather_results(results, labels, names) # average prediction over all coughs for a single patient
                outer_results.append(results)

        # get results gather by patient and calculate auc
        outer_results = sum(outer_results)/len(outer_results) # average prediction over the number of models in the outer fold
        performance_metrics += calculate_metrics(labels, outer_results)

    return performance_metrics/10

for feature_type in ['mfcc', 'melspec', 'lfb']:
        if feature_type == 'mfcc':
            features = [39]
        elif feature_type == 'melspec' or feature_type == 'lfb':
            features = [80]

        for n_feature in features:
            print(f'testing {n_feature} with {feature_type}:')
            auc = test_lstm(feature_type, n_feature)
            print(auc)