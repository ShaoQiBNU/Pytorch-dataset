###################### load packages ########################
import torch.utils.data
import pandas as pd
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.utils import shuffle


###################### IrisDataset class ########################
class IrisDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.train = train
        self.target_transform = target_transform
        self.data = pd.read_csv(data_path, sep=',')
        self.train_data, self.test_data = train_test_split(self.data, test_size = 0.3)

    ############ get data ###########
    def __getitem__(self, index):
        labels = {'setosa': 0, 'virginica': 1, 'versicolor': 2}

        if self.train:
            feature, label = self.train_data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]
        else:
            feature, label = self.test_data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]

        return feature, int(labels[label])


    ############ get data length ###########
    def __len__(self):
        if self.train:
           return len(self.train_data)
        else:
           return len(self.test_data)
