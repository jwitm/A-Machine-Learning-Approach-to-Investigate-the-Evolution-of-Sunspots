import os
import json
import random
import torch
import h5py
import numpy as np
import torchvision.transforms.functional as F


# Define a function to randomly rotate a sequence of images
def random_rotate_sequence(sequence, angle_range=(0, 360)):
    """
    Apply the same random rotation to all frames in the sequence.
    Assumes sequence of shape [T, H, W].

    :param sequence: Input sequence of shape [T, H, W].
    :param angle_range: Range of angles for rotation (min, max).
    :return: Rotated sequence.
    """
    angle = torch.rand(1).item() * (angle_range[1] - angle_range[0]) + angle_range[0]
    rotated_sequence = torch.stack([F.rotate(frame.unsqueeze(0), angle) for frame in sequence])
    return rotated_sequence.squeeze(1)

# Define a function to add random noise to each image in a sequence
def add_random_noise(sequence, noise_level=0.05):
    """
    Add random noise to each image in the sequence.

    :param sequence: Input sequence of shape [T, H, W].
    :param noise_level: The magnitude of the noise to be added.
    :return: Sequence with added noise.
    """
    # Generate random noise
    noise = torch.randn(sequence.size()) * noise_level

    # Add noise to the sequence
    noisy_sequence = sequence + noise

    # Clip the values to maintain the valid image range (e.g., 0-1 or 0-255)
    noisy_sequence = torch.clamp(noisy_sequence, 0, 1)

    return noisy_sequence

# Define a class for the HMI dataset
class HMI_Dataset():
    def __init__(self, type='continuum', crop=True, binning=False, augmentaion=False, valid_indices = True):
        super().__init__()

        basepath = "path/to/your/sequences"
        if crop:
            crop_idx = "path/to/your/cropping/idx.json"
            self.crop_idx = self.load_indices_from_json(crop_idx)
        else:
            self.crop_idx = None  
        valid_indices = torch.load('path/to/your/valid_indices.pt')
        
        self.augmentaion = augmentaion
        self.path_to_data = "path/to/your/basedata"
        all_paths = self._get_all_paths(basepath)
        counts = self._count_elements_in_files(all_paths)
        paths = self._duplicate_path_list(all_paths, counts)
        index = self._get_index(counts)
        self.full_paths = list(zip(paths, index))

        if valid_indices:
            valid_indices = np.array(valid_indices)
            self.full_paths = [self.full_paths[i] for i in valid_indices]

        if crop:
            self.crop_idx = [self.crop_idx[i] for i in valid_indices]
            self.crop_idx.pop(179)      # NOTE: This is a temporary fix. The 179th index is not valid.

        self.full_paths.pop(179)        # NOTE: This is a temporary fix. The 179th index is not valid.
        self.type = type
        self.crop = crop
        self.binning = binning

    def _get_all_paths(self, directory):
        # Get all file paths in the given directory and its subdirectories
        all_paths = []
        for root, _, files in os.walk(directory):
            for file in files:
                full_path = os.path.join(root, file)
                all_paths.append(full_path)
        return all_paths

    def _count_elements_in_files(self, directories):
        # Count the number of elements in each file in the given list of directories
        counts_list = []
        for json_path in directories:
            with open(json_path, 'r') as json_file:
                data = json.load(json_file)
                num_elements = len(data)
                counts_list.append(num_elements)
        return counts_list

    def _duplicate_path_list(self, paths, counts):
        # Duplicate the paths in the given list based on the corresponding counts
        duplicated_paths_list = []
        for path, count in zip(paths, counts):
            duplicated_paths_list.extend([path] * count)
        return duplicated_paths_list

    def _get_index(self, counts):
        # Generate an index list based on the counts
        index = []
        for c in counts:
            for i in range(0, c):
                index.append(i)
        return index
    
    def _get_label_harp(self, path):
        label = int(path.split('/')[-2])
        harp_number = path.split('/')[-1].split('.')[0]
        return label, harp_number

    def load_indices_from_json(self, filename):
        with open(filename, 'r') as json_file:
            cropped_id = json.load(json_file)
        return cropped_id

    def binning_func(self, data, bin_size = 8):
        shape = (data.shape[0], data.shape[1]//bin_size, bin_size, data.shape[2]//bin_size, bin_size)
        h = data.shape[1]%bin_size
        w = data.shape[2]%bin_size
        if h == 0 and w != 0:
            data = data[:,:, :-w]
        elif h != 0 and w ==0:
            data = data[:,:-h, :]
        elif h == 0 and w ==0:
            data = data
        else:
            data = data[:,:-h, :-w]

        data = data.reshape(shape).mean(-1).mean(-2)
        return data

    def __len__(self):
        return len(self.full_paths)

    def __getitem__(self, idx): # which item do we want to take
        path, index = self.full_paths[idx]
        label, harp_number = self._get_label_harp(path)
        with open(path, 'r') as json_file:
            tuple = json.load(json_file)[index]
        data = h5py.File(self.path_to_data, 'r')[harp_number][self.type][tuple[0]:tuple[1]]
        data = torch.tensor(data)

        if self.type == 'Dopplergram':
            mean = torch.mean(data, dim = (1,2))
            std = torch.std(data, dim = (1,2))
            data = (data-mean[:,None,None])/std[:,None,None]      # Normalize the data

        if self.type != 'bitmap' and self.type!= 'Dopplergram':
            max =torch.max(data)
            min = torch.min(data)

            data = (data-min)/(max-min)

        if self.type == 'continuum':
            data = 1-data 

        if self.crop:
            start_y,end_y,start_x,end_x = self.crop_idx[idx]
            new_data = torch.zeros((data.shape[0], end_y[0]-start_y[0], end_x[0]-start_x[0]))
            for i in range(data.shape[0]):
                new_data[i,:,:] = data[i, start_y[i]:end_y[i], start_x[i]:end_x[i]]
            data = new_data

        if self.binning != False:
            data = self.binning_func(data, bin_size = self.binning)
        data = torch.nan_to_num(data, nan=0.0)

        if self.augmentaion:
            data = add_random_noise(data, noise_level=0.1)

            ra = torch.rand(1).item()
            if ra < 0.5:
                data = random_rotate_sequence(data, angle_range=(0, 360))

        label = torch.tensor(label)
        label = label.float()

        data = data.unsqueeze(0)
        data = data.expand(3, -1, -1, -1)
        return data, label.unsqueeze(0)