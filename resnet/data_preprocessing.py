import numpy as np
import torch as th
from torchvision import transforms
from helper_scripts import to_categorical

"""
reshape the data to size 1 x max x 180
"""
def linearly_interpolate(data):
    # find the maximum size to interpolate all points to
    max = 0
    for i in range(len(data)):
        if(data[i].shape[0] > max):
            max = data[i].shape[0]

    # reshape the data
    for i in range(len(data)):
        # zero pad the data to be max x 180
        data[i] = (np.pad(data[i], [(0,(max - data[i].shape[0])), (0,0)], mode='constant', constant_values=0))
        data[i] = np.expand_dims(data[i], 0)

    return data



def reshape_data(data):
    """
    reshape data to size 1 x feature x 224
    """
    del_indx = []
    for i in range(len(data)):
        data[i] = np.transpose(data[i])
        if 224 > data[i].shape[-1]:
                # zero pad the data to be feature x 224
                data[i] = th.tensor((np.pad(data[i], [(0,0), (0,(224 - data[i].shape[-1]))], mode='constant', constant_values=0))).unsqueeze(0).unsqueeze(0)
        else:
                del_indx.append(i)

    # delete the large data
    data = np.delete(data, del_indx)

    return data, del_indx



def create_batches(data, labels, names, batch_size):
    """
    creates a data batch of size batch_size for only given data, labels and patient names
    """
    batched_data = []
    batched_labels = []
    data_lengths = []
    batched_lengths = []

    for i in range(len(data)):
        data_lengths.append(data[i].shape[0])
    
    data, del_indx = reshape_data(data)
    labels = np.delete(labels, del_indx)
    names = np.delete(names, del_indx)

    for i in range(int(np.ceil(len(data)/batch_size))):
        if(len(data) > (i+1)*batch_size):
            batched_data.append(data[i*batch_size:(i+1)*batch_size])
            batched_labels.append(labels[i*batch_size:(i+1)*batch_size])
            batched_lengths.append(data_lengths[i*batch_size:(i+1)*batch_size])
        else: 
            batched_data.append(data[i*batch_size:]) 
            batched_labels.append(labels[i*batch_size:]) 
            batched_lengths.append(data_lengths[i*batch_size:])

    for i in range(len(batched_data)):
        batched_data[i] = th.as_tensor(np.vstack(batched_data[i])).float()
        batched_labels[i] = th.as_tensor(np.vstack(to_categorical(batched_labels[i],2)))

    return batched_data, batched_labels, batched_lengths, names


def create_data_batches(data, batch_size):
    """
    creates a data batch of size batch_size for only a single input
    """

    batched_data = []
    del_indx = []
    for i in range(data.shape[0]):
        if 224 > data[i].shape[-1]:
                # zero pad the data to be feature x 224
                data[i] = th.tensor((np.pad(data[i], [(0,0), (0,(224 - data[i].shape[-1]))], mode='constant', constant_values=0))).unsqueeze(0).unsqueeze(0)
        elif data[i].shape[-1] > 224:
                del_indx.append(i)

    # delete the large data
    data = np.delete(data, del_indx, axis=0)
    for i in range(int(np.ceil(len(data)/batch_size))):
        if(len(data) > (i+1)*batch_size):
            batched_data.append(data[i*batch_size:(i+1)*batch_size,:,:,:])
        else: 
            batched_data.append(data[i*batch_size:,:,:,:]) 

    for i in range(len(batched_data)):
        batched_data[i] = th.as_tensor(np.stack(batched_data[i], axis=0)).float()

    return batched_data