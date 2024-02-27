# libraries
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import gc

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from sklearn.metrics import roc_auc_score
from get_best_features import *
from resnet import *

# set the device
device = "cuda" if th.cuda.is_available() else "cpu"
print("device=", device)
if device != "cuda":
    print("exiting since cuda not enabled")
    exit(1)

# choose which melspec we will be working on
MELSPEC = "180_melspec_fold_"
MODEL_MELSPEC = "melspec_180"


class Resnet18():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [2, 2 ,2 ,2], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=1e-4)
            self.name = "resnet_18"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=40)


class Resnet10():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [1, 1 ,1 ,1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_10"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=40)


class Resnet6_4Deep():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock1, [1, 1, 1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_6_4Deep"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=40)


class Resnet6_2Deep():
      def __init__(self):
            self.model = ResNet_2layer(ResidualBlock2, [1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_6_2Deep"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=40)


#############################################################
#                                                           #
#                                                           #
#                    Training Functions                     #
#                                                           #
#                                                           #
#############################################################


"""
complete a training step for a batch of data
"""
def train(x, y, lengths, model):
      model.model = model.model.to(device)
      # use batches when loading to prevent memory overflow
      for i in range(len(x)):
            model.optimizer.zero_grad() # set the optimizer grad to zero
            
            # prep the data
            x_batch = x[i].to(device)
            y_batch = th.as_tensor(y[i]).to(device) # grab the label

            # run through the model
            results = model.model(x_batch, lengths[i]) # get the model to make predictions
            loss = model.criterion(results, y_batch) # calculate the loss
            loss.backward() # use back prop
            model.optimizer.step() # update the model weights
            model.scheduler.step() # update the scheduler


"""
complete a training step for a batch of data
"""
def ensemble_train(x, y, model, inner_models, criterion_kl, lengths):
      model.model = model.model.to(device)
      for i in range(len(inner_models)):
            inner_models[i].model = inner_models[i].model.to(device)
      
      # update the student model using the student predictions and the teachers predictions            
      inner_results = []
      for inner_model in inner_models:
            if model.name == "bi_lstm":
                  th.manual_seed(inner_model.seed)
            inner_results.append(test(x, inner_model.model, lengths)) # do a forward pass through the models
      
      # total the predictions over all models
      inner_results = total_predictions(inner_results)
      
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu
            y_batch = th.as_tensor(y[i]).to(device) # grab the label
            inner_results[i] = th.as_tensor(inner_results[i].to(device))
            
            model.optimizer.zero_grad() # set the optimizer grad to zero
            
            # update the student model using the student predictions and the true labels (ce loss)
            results = model.model(x_batch, lengths[i]) # get the model to make predictions
            ce_loss = model.criterion(results, y_batch) # calculate the loss
            ce_loss.backward(retain_graph=True) # use back prop

            results = F.log_softmax(results, dim=1) # gets the log softmax of the output of the ensemble model
            kl_loss = criterion_kl(results, inner_results[i]) # calculate the loss
            kl_loss.backward(retain_graph=True) # use back prop
            
            model.optimizer.step() # update the model weights
            model.scheduler.step() # update the scheduler


"""
train a model on a specific inner fold within an outer fold.
"""
def train_model(train_outer_fold, train_inner_fold, model, working_folder, epochs, batch_size, interpolate, model_path):
      if train_inner_fold == None:
            data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold)
      else:
            data, labels = extract_inner_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold, train_inner_fold)

      data, labels, lengths = create_batches(data, labels, interpolate, batch_size)
      
      # run through all the epochs
      for epoch in range(epochs):
            print("epoch=", epoch)
            train(data, labels, lengths, model)

      # collect the garbage
      del data, labels, lengths
      gc.collect()

      save_model(model, working_folder, train_outer_fold, train_inner_fold, epochs, model_path, MODEL_MELSPEC)

"""
train a model on a specific inner fold within an outer fold.
"""
def train_model_on_features(train_outer_fold, train_inner_fold, model, working_folder, epochs, batch_size, interpolate, num_features, model_path):
      if train_inner_fold == None:
            data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold)
      else:
            data, labels = extract_outer_fold_data(K_FOLD_PATH + MELSPEC, train_outer_fold)

      features = dataset_fss(num_features)

      for i in range(len(data)):
            chosen_features = []
            for feature in features:
                chosen_features.append(np.asarray(data[i][:,feature]))
            data[i] = th.as_tensor(np.stack(chosen_features, -1))

      data, labels, lengths = create_batches(data, labels, interpolate, batch_size)
      
      # run through all the epochs
      for epoch in range(epochs):
            print("epoch=", epoch)
            train(data, labels, lengths, model)

      # collect the garbage
      del data, labels, lengths
      gc.collect()

      save_model(model, working_folder, train_outer_fold, train_inner_fold, epochs, model_path, MODEL_MELSPEC)







#############################################################
#                                                           #
#                                                           #
#                    Testing Functions                      #
#                                                           #
#                                                           #
#############################################################


"""
get predictions for an entire batched dataset
"""
def test(x, model, lengths):
      results = []
      for i in range(len(x)):
            results.append(get_predictions(x[i], model, lengths[i]))
      return results


"""
get predictions for a single batch
"""
def get_predictions(x_batch, model, lengths):
      with th.no_grad():
            results = (to_softmax((model((x_batch).to(device), lengths)).cpu()))
      return results




#############################################################
#                                                           #
#                                                           #
#                    Patient based testing                  #
#                                                           #
#                                                           #
#############################################################


"""
tests a singular model and makes predictions per patient.
"""
def test_patients(model, test_fold, interpolate, batch_size, threshold):
      # read in the test set
      test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)

      # preprocess data and get batches
      test_data, test_labels, test_names, lengths = create_test_batches(test_data, test_labels, test_names, interpolate, batch_size)
      
      # test model
      results = test(test_data, model, lengths)

      # stack the results
      results = np.vstack(results)
      test_labels = np.vstack(test_labels)

      # get the average prediction per patient
      unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
      out = np.column_stack((unq,np.bincount(ids,results[:,0])/count, np.bincount(ids,test_labels[:,0])/count))
      results = out[:,1]
      test_labels = out[:,2]

      # set results per threshold and get the auc
      auc = roc_auc_score(test_labels, results)
      results = (np.array(results)>threshold).astype(np.int8)
      sens, spec = calculate_sens_spec(test_labels, results)

      # mark variable and then call the garbage collector to ensure memory is freed
      del test_data, test_labels, test_names, results
      gc.collect()

      return auc, sens, spec


"""
uses group decision making to make predictions per patient.
"""
def test_models_patients(models, test_fold, interpolate, batch_size, threshold):
      test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)
      
      # preprocess data and get batches
      test_data, test_labels, test_names, lengths = create_test_batches(test_data, test_labels, test_names, interpolate, batch_size)
            
      results = []
      for model in models:
            results.append(test(test_data, model.model, lengths)) # do a forward pass through the models

      for i in range(len(results)):
            results[i] = np.vstack(results[i])

      test_labels = np.vstack(test_labels)

      for i in range(len(results)):
            unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
            out = np.column_stack((unq,np.bincount(ids,results[i][:,0])/count, np.bincount(ids,test_labels[:,0])/count))
            results[i] = out[:,1]

      test_labels = out[:,2]

      # total the predictions over all models
      results = sum(results)/4
      auc = roc_auc_score(test_labels, results)
      results = (np.array(results)>threshold).astype(np.int8)
      sens, spec = calculate_sens_spec(test_labels, results)

      del test_data, test_labels, test_names, results, lengths
      gc.collect()

      return auc, sens, spec


"""
uses group decision making to make predictions per patient.
"""
def test_models_patients_on_select(models, num_features, test_fold, interpolate, batch_size, threshold):
      test_data, test_labels, test_names = extract_test_data(K_FOLD_PATH + "test/test_dataset_mel_180_fold_", test_fold)

      features = dataset_fss(num_features)

      for i in range(len(test_data)):
            chosen_features = []
            for feature in features:
                chosen_features.append(np.asarray(test_data[i][:,feature]))
            test_data[i] = th.as_tensor(np.stack(chosen_features, -1))
      
      # preprocess data and get batches
      test_data, test_labels, test_names, lengths = create_test_batches(test_data, test_labels, test_names, interpolate, batch_size)
            
      results = []
      for model in models:
            results.append(test(test_data, model.model, lengths)) # do a forward pass through the models

      for i in range(len(results)):
            results[i] = np.vstack(results[i])

      test_labels = np.vstack(test_labels)

      for i in range(len(results)):
            unq,ids,count = np.unique(test_names,return_inverse=True,return_counts=True)
            out = np.column_stack((unq,np.bincount(ids,results[i][:,0])/count, np.bincount(ids,test_labels[:,0])/count))
            results[i] = out[:,1]

      test_labels = out[:,2]

      # total the predictions over all models
      results = sum(results)/4
      auc = roc_auc_score(test_labels, results)
      results = (np.array(results)>threshold).astype(np.int8)
      sens, spec = calculate_sens_spec(test_labels, results)

      del test_data, test_labels, test_names, results, lengths
      gc.collect()

      return auc, sens, spec