from DataSet import HMI_Dataset
import multiprocessing
import numpy as np
import random
import torch
import os

def reduce_dataset(dir_path:str):
    """
    This function reduces the dataset such that we have an equal amount of positive 
    and negative examples. It does also fiter the Dataset such that the minimal shape of
    the input is (1,17,17) 

    dir_path: where to save the valid indices
    """
    Data = HMI_Dataset(type = "continuum", crop = True, binning = False, augmentaion = False, valid_indices = False)
    valid_input_shape = (1, 17, 17)

    label_positive = []
    label_negative = []

    if not os.path.exists(dir_path):
            try:
                os.makedirs(dir_path)
            except FileExistsError:
                pass

    for i, (data, label) in enumerate(Data):
        _,time,height,width = data.shape                   # remove the channel dimension
        if time>=valid_input_shape[0] and height>=valid_input_shape[1] and width >= valid_input_shape[2]:
            if label ==1:
                label_positive.append(i)
            elif label ==0:
                label_negative.append(i)
    
    amount_of_positive_labels = len(label_positive)
    amount_of_negative_labels = len(label_negative)

    if amount_of_negative_labels > amount_of_positive_labels:
        label_negative = random.sample(label_negative, amount_of_positive_labels)
    elif amount_of_positive_labels > amount_of_negative_labels:
        label_positive = random.sample(label_positive, amount_of_negative_labels)
    
    valid_indices = label_positive + label_negative
    valid_indices = torch.tensor(np.array(valid_indices))
    torch.save(valid_indices, f'{dir_path}/valid_indices.pt')
    print(f"relative amount of data: {len(valid_indices)/len(Data)*100} \%")

def get_dataset_specs():
    Data = HMI_Dataset(type='continuum', crop=True, binning=False, augmentaion=False, valid_indices = True)
    max_len = 0
    max_width = 0
    max_height = 0
    for i in range(len(Data)):
        data,_ = Data[i]
        length, height, width = data.shape[1:]
        if length > max_len:
            max_len = length
        if width > max_width:
            max_width = width
        if height>max_height:
            max_height = height
    print(f'The max length for mode = {mode} is: {max_len}')
    print(f'The max height for mode = {mode} is: {max_height}')
    print(f'The max width for mode = {mode}  is: {max_width}')
    print(f'Total amount of data in {mode}-mode: {len(Data)}\n')


if __name__ == "__main__":
    dir_path = f'/path/to/saving/directory'

    reduce_dataset(dir_path)
    get_dataset_specs()