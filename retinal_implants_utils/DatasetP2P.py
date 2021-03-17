"""
DatasetP2P.py

MIT License

Copyright (c) 2020 Alexandros Benetatos

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

"""
DESCRIPTION

A torch.utils.data.Dataset subclass for pytorch to load the custom dataset created by the percepted images
generated from stimulating square retinal implants with images from MNIST and CIFAR10 datasets
"""

import numpy as np
import pickle
import torch
import os

class DatasetP2P(torch.utils.data.Dataset):
    def __init__(self, path, val_perc=0, train=True, val=False, transform=None, subset_samples=None):
        self.val   = val
        self.train = train
        # check if path exists
        if not os.path.exists(path):
            raise NameError("Path does not exist, change it or run create_dataset first")

        # if not test load train dataset
        if (train or val):
            dset_type = "train"
        else:
            dset_type = "test"

        # data (images) saved as pickle object
        with open(os.path.join(path, dset_type+"_set_data.pk"), "rb") as f:
            self.data = pickle.load(f)

        # labels (integers) saved with torch.save (from a list of ints)
        self.targets = torch.load(os.path.join(path, dset_type+"_set_labels.pt"))

        if isinstance(self.targets, list):
            self.targets = np.array(self.targets)

        # save the desired transforms to apply to each data sample
        self.transform = transform

        if subset_samples is not None:
            self.get_equal_subset(subset_samples)

#         self.orig_data    = self.data.copy()
#         self.orig_targets = self.targets.copy()

        # get dataset dimensions
        data_size  = len(self.data)

        if val:
            self.get_equal_subset(round(val_perc * data_size))
            # # Reserve val_size samples for validation
            # self.data    = self.data[   train_size:]
            # self.targets = self.targets[train_size:]
        elif train:
            self.get_equal_subset(round((1 - val_perc) * data_size))
            # self.data    = self.data[   :train_size]
            # self.targets = self.targets[:train_size]

    def get_equal_subset(self, samples, val_split=False):
        if samples == 0 or samples == len(self.data):
            return

        if val_split:
            samples = len(self.data) - samples

        targets_number     = len(np.unique(self.orig_targets))
        samples_per_target = int(samples/targets_number)

#         whr_subset = np.concatenate([np.argwhere(labels==i).flatten()[:samples_per_label]
#                                      for i in range(labels_number)]).flatten()

        np.random.seed(77) # for predictable subsets using numpy.random.choice
        whr_subset = np.concatenate([np.random.choice(np.argwhere(self.targets==i).flatten(),
                                                      samples_per_target, replace=False)
                                     for i in range(targets_number)]).squeeze().flatten()

        if val_split:
            mask = np.ones(len(self.data), np.bool)
            mask[whr_subset] = 0
        else:
            mask = whr_subset

#         self.targets = [dt for idx, dt in enumerate(self.orig_targets) if idx in whr_subset]
        self.data    = self.data[mask]
        self.targets = labels[mask]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label  = int(self.targets[idx])

        # apply any transform desired
        if self.transform:
            sample = self.transform(sample)

        return [sample, label]