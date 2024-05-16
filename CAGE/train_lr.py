# libraries
import torch as th
import pickle
import numpy as np


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

def create_inner_lr():
    data_dir = f'../../data/tb/CAGE_03_04_2023/melspecs/'

    for test in range(1,11):
        train_data = []
        test_data = pickle.load(open(f'{data_dir}fold_{test}', 'rb'))
        for train in range(1,11):
            if train != test:
                data = pickle.load(open(f'{data_dir}fold_{train}', 'rb'))
                train_data = train_data + data
        
        print(np.array(train_data, dtype=object).shape)
        
        


def main():
    create_inner_lr()
            
            

if __name__ == "__main__":
    main()