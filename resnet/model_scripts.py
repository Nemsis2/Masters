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
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=50)


class Resnet10():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock2, [1, 1 ,1 ,1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_10"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=50)


class Resnet6_4Deep():
      def __init__(self):
            self.model = ResNet_4layer(ResidualBlock1, [1, 1, 1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_6_4Deep"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=50)


class Resnet6_2Deep():
      def __init__(self):
            self.model = ResNet_2layer(ResidualBlock2, [1, 1], num_classes=2)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=0.0001)
            self.name = "resnet_6_2Deep"
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=16, steps_per_epoch=50)


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
            model.criterion.to(device)

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
def train_ts(x, y, model, inner_models, criterion_kl, lengths):
      model.model = model.model.to(device)
      for i in range(len(inner_models)):
            inner_models[i].model = inner_models[i].model.to(device)
      
      # update the student model using the student predictions and the teachers predictions            
      inner_results = []
      for inner_model in inner_models:
            inner_results.append(test(x, inner_model.model, lengths)) # do a forward pass through the models
      
      # total the predictions over all models
      inner_results = total_predictions(inner_results)
      
      for i in range(len(x)):
            model.criterion.to(device)
            criterion_kl.to(device)
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

