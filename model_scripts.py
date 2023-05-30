# libraries
import torch as th
import torch.nn.functional as F
import gc

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *

# set the device
device = "cuda" if th.cuda.is_available() else "cpu"

# choose which melspec we will be working on
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"


"""
complete a training step for a batch of data
"""
def train(x, y, model):
      model.model = model.model.to(device)
      # use batches when loading to prevent memory overflow
      #optimizer.zero_grad() # set the optimizer grad to zero
      # loss = 0
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu
            y_batch = th.as_tensor(y[i]).to(device) # grab the label

            # run through the model
            results = model.model(x_batch) # get the model to make predictions
            loss = model.criterion(results, y_batch) # calculate the loss
            loss.backward() # use back prop
            model.optimizer.step() # update the model weights
            model.optimizer.zero_grad() # set the optimizer grad to zero


"""
complete a training step for a batch of data
"""
def ensemble_train(x, y, model, inner_model, criterion_kl):
      model.model = model.model.to(device)
      inner_model = inner_model.to(device)
      
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu
            y_batch = th.as_tensor(y[i]).to(device) # grab the label

            # run through the model
            results = model.model(x_batch) # get the model to make predictions
            inner_results = get_predictions(x_batch, inner_model).to(device) #returns softmax predictions of the inner model
            ce_loss = model.criterion(results, y_batch) # calculate the loss
            ce_loss.backward(retain_graph=True) # use back prop
            results = F.log_softmax(results, dim=1) # gets the log softmax of the output of the ensemble model
            kl_loss = criterion_kl(results, inner_results) # calculate the loss
            kl_loss.backward(retain_graph=True) # use back prop
            model.optimizer.step() # update the model weights
            model.optimizer.zero_grad() # set the optimizer grad to zero


"""
get predictions for an entire batched dataset
"""
def test(x, model):
      results = []
      for i in range(len(x)):
            results.append(get_predictions(x[i], model))

      return results


"""
get predictions for a single batch
"""
def get_predictions(x_batch, model):
      with th.no_grad():
            results = (to_softmax((model(th.tensor(x_batch).to(device))).cpu()))

      return results


"""
validates a model by testing its performance on the corresponding validation fold.
"""
def validate_model(model, train_outer_fold, train_inner_fold, interpolate, batch_size):
      if train_inner_fold == None:
            val_data, val_labels = extract_outer_val_data(K_FOLD_PATH + MELSPEC, train_outer_fold)
      else:
            # get the train fold
            val_data, val_labels = extract_val_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)

      # preprocess data and get batches
      val_data, val_labels = create_batches(val_data, val_labels, interpolate, batch_size)
      
      # test model
      results = test(val_data, model)
      
      # assess the accuracy of the model
      auc, sens, spec  = performance_assess(val_labels, results)
      
      # display and log the AUC for the test set
      #print("AUC for test_fold",train_inner_fold, "=", auc)
      #log_test_info(train_inner_fold, auc, sens, spec)

      del val_data, val_labels
      gc.collect()

      return auc, sens, spec


"""
train a model on a specific inner fold within an outer fold.
can be set to include or exclude validation set as neccessary.
"""
def train_model(train_outer_fold, train_inner_fold, model, working_folder, epochs, batch_size, interpolate, model_path, final_model=0):
      # run through all the epochs
      for epoch in range(epochs):
            print("epoch=", epoch)

            if train_inner_fold == None:
                  data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold, final_model)
            else:
                  # get the train fold
                  data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold, final_model)

            data, labels = create_batches(data, labels, interpolate, batch_size)
            train(data, labels, model)

            # collect the garbage
            del data, labels
            gc.collect()

      save_model(model, working_folder, train_outer_fold, train_inner_fold, final_model, epochs, model_path, MODEL_MELSPEC)


"""
test a singular model on the corresponding test set.
"""
def test_model(model, test_fold, interpolate, batch_size):
      # read in the test set
      test_data, test_labels = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)

      # preprocess data and get batches
      test_data, test_labels = create_batches(test_data, test_labels, interpolate, batch_size)
      
      # test model
      results = test(test_data, model)
      
      # assess the accuracy of the model
      auc, sens, spec  = performance_assess(test_labels, results)

      # display and log the AUC for the test set
      print("AUC for test_fold",test_fold, "=", auc)
      log_test_info(test_fold, auc, sens, spec)

      # mark variable and then call the garbage collector to ensure memory is freed
      del test_data, test_labels
      gc.collect()


"""
test multiple models on the test set using the average decision among all models to make a prediction.
"""
def test_models(models, test_fold, interpolate, batch_size):
      test_data, test_labels = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)

      # preprocess data and get batches
      test_data, test_labels = create_batches(test_data, test_labels, interpolate, batch_size)
            
      results = []
      for model in models:
            results.append(test(test_data, model)) # do a forward pass through the models

      final_results = results[0]
      for i in range(1,len(results)):
            print("i=",i)
            for j in range(len(results[i])):
                  final_results[j] += results[i][j]

      for i in range(len(final_results)):
            final_results[i] = final_results[i]/len(results)

      auc, sens, spec  = performance_assess(test_labels, final_results)

      # display and log the AUC for the test set
      print("AUC for test_fold",test_fold, "=", auc)
      log_test_info(test_fold, auc, sens, spec)

      del test_data, test_labels
      gc.collect()
      