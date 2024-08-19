# libraries
import torch as th
import torch.optim as optim
import torch.nn.functional as F
import gc

# custom scripts
from helper_scripts import *
from data_grab import *
from data_preprocessing import *
from sklearn.metrics import roc_auc_score
from get_best_features import *
from tqdm import tqdm

# set the device
device = "cuda" if th.cuda.is_available() else "cpu"


"""
Create a bi_lstm model
"""
class bi_lstm(nn.Module):
      def __init__(self, input_dim, hidden_dim, layers):
            super(bi_lstm, self).__init__()
            self.hidden_dim = hidden_dim
            self.layers = layers        
            if layers < 1:
                  self.bi_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, dropout=0.5, bidirectional=True)
            else:
                  self.bi_lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, num_layers=layers, batch_first=True, bidirectional=True)

            self.drop1 = nn.Dropout(p=0.5)
            self.batchnorm = nn.BatchNorm1d(self.hidden_dim * 2)
            self.fc1 = nn.Linear(hidden_dim*2, 32)
            self.mish = nn.Mish()
            self.fc2 = nn.Linear(32,2)

      def forward(self, x, lengths):
            total_length = x.shape[1]
            x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            self.bi_lstm.flatten_parameters()
            out, (h_n, c_n) = self.bi_lstm(x)
            out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=total_length)
            
            out_forward = out[range(len(out)), (np.array(lengths) - 1), :self.hidden_dim]
            out_reverse = out[range(len(out)), 0, self.hidden_dim:]
            out_reduced = th.cat((out_forward, out_reverse), dim=1)
            
            result = self.drop1(out_reduced)
            result = self.batchnorm(result)
            result = self.fc1(result)
            result = self.mish(result)
            result = self.fc2(result)
            return result 


"""
Create a bi_lstm package including
Model
Optimizer(RMSprop)
Model name
"""
class bi_lstm_package():
      def __init__(self, input_size, hidden_dim, layers, outer, inner, epochs, batch_size, model_type, n_feature, feature_type):
            self.seed = th.seed()
            self.epochs = epochs
            self.batch_size = batch_size
            self.model_type = model_type
            self.outer = outer
            self.inner = inner
            self.n_feature = n_feature
            self.feature_type = feature_type
            self.k_fold_path = f'../../data/tb/combo/new/{n_feature}_{feature_type}_fold_{outer}.pkl'
            self.test_path = f'../../data/tb/combo/new/test/test_dataset_{feature_type}_{n_feature}_fold_{outer}.pkl'
            
            self.model = bi_lstm(input_size, hidden_dim, layers)
            self.criterion = nn.CrossEntropyLoss()
            self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)
            self.scheduler = th.optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=1e-3, epochs=180, steps_per_epoch=30)

      def train(self):
            data, labels = extract_inner_fold_data(self.k_fold_path, self.inner)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)

            data, labels, lengths = create_batches(data, labels, "linear", self.batch_size)

            # run through all the epochs
            for epoch in tqdm(range(self.epochs)):
                  train(data, labels, lengths, self)

            # collect the garbage
            del data, labels, lengths
            gc.collect()

      
      def train_on_select_features(self, data, labels, lengths):
            # run through all the epochs
            for epoch in tqdm(range(self.epochs)):
                  train(data, labels, lengths, self)

            # collect the garbage
            del data, labels, lengths
            gc.collect()


      
      def train_exclusive_ts(self, models):
            data, labels = extract_outer_fold_data(self.k_fold_path)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)

            data, labels, lengths = create_batches(data, labels, "linear", self.batch_size)

            # run through all the epochs
            for epoch in tqdm(range(self.epochs)):
                  train_ts(data, labels, self, models, lengths)

            # collect the garbage
            del data, labels, lengths
            gc.collect()


      def train_ts(self, models):
            data, labels = extract_outer_fold_data(self.k_fold_path)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)

            data, labels, lengths = create_batches(data, labels, "linear", self.batch_size)
            criterion_kl = nn.KLDivLoss()
            
            # run through all the epochs
            for epoch in tqdm(range(self.epochs)):
                  train_ts_2(data, labels, self, models, lengths, criterion_kl)

            # collect the garbage
            del data, labels, lengths
            gc.collect()


      def train_fss(self, fss_feature):
            data, labels = extract_inner_fold_data(self.k_fold_path, self.inner)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)

            data, labels, lengths = create_batches(data, labels, "linear", self.batch_size)
            # select only the relevant features
            feature_path = f'../../models/tb/lr/{self.feature_type}/{self.n_feature}_{self.feature_type}/fss/docs/'
            if self.feature_type == 'mfcc':
                  selected_features = outer_fss(self.outer, self.n_feature*3, fss_feature, feature_path)
            else:
                  selected_features = outer_fss(self.outer, self.n_feature, fss_feature, feature_path)
            
            print(selected_features)
            for batch in range(len(data)):
                  chosen_features = []
                  for feature in selected_features:
                        chosen_features.append(np.asarray(data[batch][:,:,feature]))
                  data[batch] = th.as_tensor(np.stack(chosen_features, -1))
            
            # run through all the epochs
            for epoch in tqdm(range(self.epochs)):
                  train(data, labels, lengths, self)

            # collect the garbage
            del data, labels, lengths
            gc.collect()

      def save(self, fss_feature=0):
            model_path = f'../../models/tb/lstm/{self.feature_type}/{self.n_feature}_{self.feature_type}/{self.model_type}'
            if self.model_type == 'dev':
                  pickle.dump(self, open(f'{model_path}/lstm_{self.feature_type}_{self.n_feature}_outer_fold_{self.outer}_inner_fold_{self.inner}', 'wb')) # save the model
            elif self.model_type == 'ts' or self.model_type == 'ts_2':
                  pickle.dump(self, open(f'{model_path}/lstm_{self.feature_type}_{self.n_feature}_outer_fold_{self.outer}', 'wb')) # save the model
            elif self.model_type =='fss':
                  pickle.dump(self, open(f'{model_path}/lstm_{self.feature_type}_{self.n_feature}_fss_{fss_feature}_outer_fold_{self.outer}_inner_fold_{self.inner}', 'wb')) # save the model
        

      def dev_fss(self, fss_feature):
            # read in the dev set
            data, labels, names = extract_dev_data(self.k_fold_path, self.inner)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)
            
            data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)
            # select only the relevant features
            feature_path = f'../../models/tb/lr/{self.feature_type}/{self.n_feature}_{self.feature_type}/fss/docs/'
            if self.feature_type == 'mfcc':
                  selected_features = dataset_fss(self.n_feature*3, fss_feature, feature_path)
            else:
                  selected_features = dataset_fss(self.n_feature, fss_feature, feature_path)
            
            for batch in range(len(data)):
                  chosen_features = []
                  for feature in selected_features:
                        chosen_features.append(np.asarray(data[batch][:,:,feature]))
                  data[batch] = th.tensor(np.stack(chosen_features, -1))

            results = []
            for i in range(len(data)):
                  with th.no_grad():
                        results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

            results = np.vstack(results)
            labels = np.vstack(labels)
            results, labels = gather_results(results, labels, names)
            threshold  = get_EER_threshold(labels, results)
            
            return threshold
      
      
      def dev_on_select_features(self, fss_feature):
            # read in the dev set
            data, labels, names = extract_dev_data(self.k_fold_path, self.inner)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)
            
            data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)
            # select only the relevant features
            feature_path = f'../../models/tb/lr/{self.feature_type}/{self.n_feature}_{self.feature_type}/fss/docs/'
            if self.feature_type == 'mfcc':
                  selected_features = dataset_fss(self.n_feature*3, fss_feature, feature_path)
            else:
                  selected_features = dataset_fss(self.n_feature, fss_feature, feature_path)
            
            for batch in range(len(data)):
                  chosen_features = []
                  for feature in selected_features:
                        chosen_features.append(np.asarray(data[batch][:,:,feature]))
                  data[batch] = th.tensor(np.stack(chosen_features, -1))

            results = []
            for i in range(len(data)):
                  with th.no_grad():
                        results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

            results = np.vstack(results)
            labels = np.vstack(labels)
            results, labels = gather_results(results, labels, names)
            auc = roc_auc_score(labels, results)

            return auc

      def dev(self,):
            data, labels, names = extract_dev_data(self.k_fold_path, self.inner)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)

            # preprocess data and get batches
            data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)

            results = []
            for i in range(len(data)):
                  with th.no_grad():
                        results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

            results = np.vstack(results)
            labels = np.vstack(labels)

            results, labels = gather_results(results, labels, names)
            auc = roc_auc_score(labels, results)

            return auc

      def test(self):
            # read in the test set
            data, labels, names = extract_test_data(self.test_path)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)

            # preprocess data and get batches
            data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)

            results = []
            for i in range(len(data)):
                  with th.no_grad():
                        results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

            results = np.vstack(results)
            labels = np.vstack(labels)

            return results, labels, names

      def test_fss(self, fss_feature):
            # read in the test set
            data, labels, names = extract_test_data(self.test_path)

            if self.feature_type=="mfcc":
                  data = normalize_mfcc(data)
            
            data, labels, names, lengths = create_test_batches(data, labels, names, "linear", self.batch_size)
            # select only the relevant features
            feature_path = f'../../models/tb/lr/{self.feature_type}/{self.n_feature}_{self.feature_type}/fss/docs/'
            if self.feature_type == 'mfcc':
                  selected_features = dataset_fss(self.n_feature*3, fss_feature, feature_path)
            else:
                  selected_features = dataset_fss(self.n_feature, fss_feature, feature_path)
            
            for batch in range(len(data)):
                  chosen_features = []
                  for feature in selected_features:
                        chosen_features.append(np.asarray(data[batch][:,:,feature]))
                  data[batch] = th.tensor(np.stack(chosen_features, -1))

            results = []
            for i in range(len(data)):
                  with th.no_grad():
                        results.append(to_softmax((self.model((data[i]).to(device), lengths[i])).cpu()))

            results = np.vstack(results)
            labels = np.vstack(labels)
            
            return results, labels, names





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
complete a training step for a batch of data
"""
def train_ts(x, y, model, inner_models, lengths):
      model.model = model.model.to(device)
      for i in range(len(inner_models)):
            inner_models[i].model = inner_models[i].model.to(device)
      
      
      for i in range(len(x)):
            # prep the data
            x_batch = th.as_tensor(x[i]).to(device) # grab data of size batch and move to the gpu

            # update the student model using the student predictions and the teachers predictions            
            inner_results = []
            for inner_model in inner_models:
                  th.manual_seed(inner_model.seed)
                  inner_results.append(get_predictions(x_batch, inner_model.model, lengths[i])) # do a forward pass through the models

            # total the predictions over all models
            inner_results = sum(inner_results)/len(inner_results)
            inner_results = th.as_tensor(inner_results.to(device))
            
            model.optimizer.zero_grad() # set the optimizer grad to zero
            
            # update the student model using the student predictions and the teacher predictions (ce loss)
            results = model.model(x_batch, lengths[i]) # get the model to make predictions
            ce_loss = model.criterion(results, inner_results) # calculate the loss
            ce_loss.backward(retain_graph=True) # use back prop
            
            model.optimizer.step() # update the model weights
            model.scheduler.step() # update the scheduler



def train_ts_2(x, y, model, inner_models, lengths, criterion_kl):
      model.model = model.model.to(device)
      for i in range(len(inner_models)):
            inner_models[i].model = inner_models[i].model.to(device)
      
      # update the student model using the student predictions and the teachers predictions            
      inner_results = []
      for inner_model in inner_models:
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