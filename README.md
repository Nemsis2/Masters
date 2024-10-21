# Masters

### Overview
The code contained here is the code used to create the models used for my Thesis titled "The Role of Features in Predictive Deep Learning Models for Auditory Tuberculosis Classification" which will be available on Stellenbosch University archive.
As a general rule for each model architecture a new folder was created resulting in a lot of code duplication with small edits.
Unfortunately this code is not is not consumer grade, therefore there is an incredible lack of sanitisation of inputs and comments were added retrospectively not during development or are missing.
For this reason, it is important when using this code in anything which is not the state in which it was found to read through the code carefully as it frequently may operate in unexpected manners.


### High Level File breakdown
```
CAGE - Scripts for LR, LSTM and ResNet architectures built to run on the most recent CAGE data (16th October 2024).

CAGE_dataset_generation - Script to create the usable dataset from raw audio from the most recent CAGE data 16th October 2024).

dataset_generation - Scripts to create the usable dataset from the Geoff dataset.

frame_skip_dataset_generation - Near identical to dataset_generation however this generates data where frames are skipped to maintain a uniform window length as per the paper written by Madhu titled "COVID-19 cough classification using machine learning and global smartphone recordings" and explained in section 3 of this paper.

helper_scripts - Contains a baseline of scripts that can be used when implementing a new model architecture including things such as: grabbing data, post fss feature selection etc. 

image_generation - Contains code for generating images and graph.

LR - Contains all code used to train, validate, test and perform FSS using logistic regression models.

lr_fss_graph - Similar to what is contained in the LR folder primarily used to generate graphs.

LSTM - Contains all code used to train, validate, test and perform FSS using LSTM models.

outdatated scripts - Contains code from outdated scripts. Largely uncommented and unexplained this code is essentially useless and is kept only for legacy reasons.

resnet - Contains all code used to train, validate, test and perform FSS using resnet models.

resnet_fss_graph - Similar to what is contained in the resnet folder primarily used to generate graphs.
```

### Individual File Breakdown
```
Below is a brief breakdown of each of the files contained within the main folders as well as a list of ignored
folders and a reason for why each should be ignored.
Some folders may not have every script covered, if this is the case, look under the helper_scripts section to see if it is covered there.
For files covered under the helper_scripts heading it is important to remember each file is edited to fit the model it is being used for therefore the data_grab.py folder in the LR or LSTM folders may be slightly different.
```

# Ignored folders
```
frame_skip_dataset_generation - near identical copy of dataset_generation with the only edit
allowing for the frame skipping. Review dataset_generation instead.

image_generation - Largely useless for anyone reusing this repo. 

lr_fss_graph - Largely useless for anyone reusing this repo. Review LR instead.

outdated scripts - Largely useless for anyone reusing this repo. Only kept for legacy reasons.

resnet_fss_graph - Largely useless for anyone reusing this repo. Review resnet instead.
```

# Helper_scripts
```
├── data_grab.py #contains functions for grabbing preprocessed (from audio to lfb, mfb, mfcc) training, validation and testing data
├── data_preprocessing.py #contains function for preproccesing before use in models such as batching and padding.
├── get_best_features.py #grabs lists of the optimal features determined by FSS
├── helper_scripts.py #any kind of supporive script
├── model_scripts.py #scripts pertaining to the running of a model such as training and testing. Mostly used for resnets.
```

# LR
```
├── data_check.py #mostly just for sanity checks on the data; number of labels etc.
├── fss_lr.py #for performing fss using LR
├── lr_model_scripts.py #for training, testing and gathering results
├── test_lr.py #for testing using different configs and models
├── train_lr.py #for training new LR models in different configs
├── val_check.py #sanity check to ensure no overlap between train/val
```

# LSTM
```
├── fss_lstm.py #for performing fss using LSTMs
├── lstm_model_scripts.py #contains the LSTM class as well as training, testing and other scripts
├── test_lstm.py #for testing LSTMs using different configs
├── train_lstm.py #for training LSTMs using different configs
```

# resnet
```
├── fss_resnet.py #for performing fss using resnets
├── model_scripts.py #scripts for traning, validating and testing different resnet configs
├── resnet.py #implementation script of the resnet architecture
├── resnet_main.py #legacy script for training, validating and testing resnet models
├── test_resnet.py #script for testing resnet models
├── train_resnet.py #script for training resnet models
```

# CAGE
```
├── lstm - see lstm section
├── resnet - see resnet section
├── test_lr.py - see lr section
├── test_resnet.py - ignore this 
├── train_lr.py -  see lr section
```

# CAGE_dataset_generation
```
├── piepline.py #script for doing the audio to spectrogram processing for each fold. No speed perturbation.
```

# dataset_generation
```
├── Cough_patient_metadata.xlsx #metadata for geoff data
├── extraction.py #script for splitting data into nested k-folds. Includes speed perturbation. Adapted from geoff. 
├── pipeline.py #script for doing the audio to spectrogram processing. Includes speed perturbation. Adapated from geoff
```


# running experiments
Firstly ensure that you have the relevant data and correct file structure. You can assess this by opening the relevant model architecture folder (LR, LSTM, resnet) and checking the pathways in the train file to see where data will be grabbed from.
If you have only the raw files (3way_data.pkl or 3way_renier_data.pkl) place them into the dataset_generation folder and run the extraction file.
After this simply run the train file (check the main function to see what will be trained) and then the test file (again check the main function).

