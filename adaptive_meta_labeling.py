from tqdm import tqdm
import numpy as np
import random
import torch
import json
import h5py
import sys
import os

class MetaLabeling:
    def __init__(self, path_to_data, path_to_labels, save = False):
        self.save = save                                # boolean to decide if data should be saved

        # pathes where sequence labels should be saved
        self.path_negative = "/path/to/negative/labels"
        self.path_positive = "/path/to/positive/labels"

        # check if directory already exists otherwise create it (assuming that if one directory exists all exist)
        if os.path.exists(self.path_negative):
            if save:
                user_input = input(f"Warning: The meta label directory already exist. Data will be overwritten. Do you want to continue? (y/n): ")
                if user_input.lower() != 'y':
                    print("Operation aborted.")
                    sys.exit()
                else:
                    print("Continuing with the operation...")

        else:
            os.makedirs(self.path_negative)
            os.makedirs(self.path_positive)
            print(f"Directory {self.path_negative} created.")
            print(f"Directory {self.path_positive} created.")

        self.path_to_data = path_to_data                # path to raw data
        self.path_to_labels = path_to_labels            # path to labels

        self.harp_numbers = self._get_harp_numbers()    # get all harp_numbers

        self.total_length = len(self.harp_numbers)      # total length of harp_numbers

        self.counter = 0                                # counter for harp_numbers  

    def _get_harp_numbers(self):
        """
        returns a list of all harp_numbers in the h5 file
        """
        with h5py.File(self.path_to_data, 'r') as h5_file:  # open h5 file and get all harp_numbers 
            harp_numbers = list(h5_file.keys())
        return harp_numbers
    
    def _get_data(self, count = True):
        """
        returns the bitmap and the labels for the current harp_number
        """
        Bitmap = h5py.File(self.path_to_data, 'r')[self.harp_numbers[self.counter]]["bitmap"] # get bitmap for harp_number
        labels = torch.load(f'{self.path_to_labels}/{self.harp_numbers[self.counter]}.pt')    # get labels for harp_number
        if count:
            self.counter += 1                                                                       # increase counter
        return torch.tensor(np.array(Bitmap)), labels

    def _crop_data(self, max_elements = 3093095.4956709878):
        """
        returns the maximal window size of the current harp_number in the time dimension
        """
        Bitmap,_ = self._get_data()                                                         # get bitmap for harp_number
        d = torch.zeros((Bitmap.shape[0],2),dtype = torch.int)                              # initialize tensor for dimensions

        for i in range(Bitmap.shape[0]):                                                    # loop over all images
            image = Bitmap[i,:,:]                                                           # get image
            if torch.any(image>33):
                y,x = torch.where(image>33)
                d[i,0] = int(torch.max(x)-torch.min(x))
                d[i,1] = int(torch.max(y)-torch.min(y))
        dx = 2*(torch.max(d[:,0])//2)
        dy = 2*(torch.max(d[:,1])//2)

        # below the division by 3 happens, because the max_element is calculated for a tensor of shape (3,t,dy,dx)
        max_T = int(max_elements/(3*dx*dy))                                                 # get maximal time dimension
        return max_T, dx, dy

    def meta_labeling_split(self):
        """
        applies the meta labeling to the data
        """

        indices0 = []
        indices1 = []

        _, labels = self._get_data(count = False)                                               # get labels for harp_number
        max_T, dx, dy = self._crop_data()                                                       # get maximal time dimension

        if max_T >10 and dx>=17 and dy >=17:                                                    # max_T has to be bigger than the window size from smoothing, height and width have to be grater than 17 because of kernel size
        
            changes = []
            for i in range(len(labels)-1):                                                      # loop over all labels
                if labels[i] != labels[i+1]:                                                    # check if label changes
                    changes.append(i)                                                           # if so, append index to list (e.g. -->0,1)

            if len(changes) == 0 and labels[0] == 0:                                            # if no label change occurs
                if max_T > len(labels):
                    indices0.append((0,len(labels)))
                else:
                    i = random.randint(0,len(labels)-max_T)                                         # get random index
                    indices0.append((i,i+max_T))                                                    # only take one sequence per HARP
            elif len(changes) <= 2 and len(changes)!=0:                                             # if no label change occurs
                intervals = [second - first for first, second in zip(changes, changes[1:])]
                intervals.append(len(labels)-changes[-1])
                intervals.insert(0,changes[0])

                for i,c in enumerate(changes):
                    # all the cases where transition from 0 to 1 occours
                    current_interval = intervals[i]
                    if labels[c] == 0:
                        if current_interval>=max_T:
                            indices1.append((c-max_T+1,min(c+1, len(labels))))
                        elif intervals[i+1]>=max_T-current_interval:
                            indices1.append((c-current_interval+1,min(c-current_interval+1+max_T, len(labels))))
                        else:
                            indices1.append((c-current_interval+1,min(c+intervals[i+1], len(labels))))
        return indices0, indices1

    def apply_labeling(self):
        """
        applies the meta labeling to all harp_numbers and splits the data into test and
        train set with split of 1/3 and 2/3 respectively. 

        This method is not necessary to execute, because only maximally one sequence per HARP is taken.
        """
        test0, test1, train0, train1 = [], [], [], []
        for i in tqdm(range(self.total_length)):
            harp_number = self.harp_numbers[self.counter]
            indices0, indices1 = self.meta_labeling_split()
            r = random.randint(1,1e6)
            if r <=333333:
                path0 = self.path_test_0
                path1 = self.path_test_1
                for tup0 in indices0:
                    test0.append(tup0)
                if self.save and len(indices0) !=0:
                    directory = os.path.join(path0, f'{harp_number}.json')
                    with open(directory, 'w') as file:
                        json.dump(indices0, file)
                for tup1 in indices1:
                    test1.append(tup1)
                if self.save and len(indices1) !=0:
                    directory = os.path.join(path1, f'{harp_number}.json')
                    with open(directory, 'w') as file:
                        json.dump(indices1, file)
            else:
                path0 = self.path_train_0
                path1 = self.path_train_1
                for tup0 in indices0:
                    train0.append(tup0)
                if self.save and len(indices0) !=0:
                    directory = os.path.join(path0, f'{harp_number}.json')
                    with open(directory, 'w') as file:
                        json.dump(indices0, file)
                for tup1 in indices1:
                    train1.append(tup1)
                if self.save and len(indices1) !=0:
                    directory = os.path.join(path1, f'{harp_number}.json')
                    with open(directory, 'w') as file:
                        json.dump(indices1, file)
        return test0, test1, train0, train1
    
    def produce_dataset(self):
        """
        produce and save the dataset, i.e. the labels for the positive and negative class
        """
        positive, negative = [], []
        for i in tqdm(range(self.total_length)):
            harp_number = self.harp_numbers[self.counter]
            indices0, indices1 = self.meta_labeling_split()
            path0 = self.path_negative
            path1 = self.path_positive
            for tup0 in indices0:
                    negative.append(tup0)
            if self.save and len(indices0) !=0:
                directory = os.path.join(path0, f'{harp_number}.json')
                with open(directory, 'w') as file:
                    json.dump(indices0, file)
            for tup1 in indices1:
                positive.append(tup1)
            if self.save and len(indices1) !=0:
                directory = os.path.join(path1, f'{harp_number}.json')
                with open(directory, 'w') as file:
                    json.dump(indices1, file)
        return positive, negative

def convert_minutes_to_hours_minutes_seconds(minutes):
    # Convert minutes to hours, minutes, and seconds
    hours, remainder = divmod(minutes, 60)
    minutes, seconds = divmod(remainder*60, 60)

    return hours, minutes, seconds
                 
if __name__ == "__main__":
    path_to_data = '/path/to/the/base/data'
    path_to_labels = 'batch_to_labels'
    ML = MetaLabeling(path_to_data, path_to_labels, save = True)
    positive, negative = ML.produce_dataset()
    print(f'legth of positive: {len(positive)}')
    print(f'legth of negative: {len(negative)}')