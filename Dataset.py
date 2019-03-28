###################### load packages ########################
import torch.utils.data
import pandas as pd
import numpy as np


###################### IrisDataset class ########################
class IrisDataset(torch.utils.data.Dataset):

    ############ init ###########
    def __init__(self, data_path, train=True, transform=None, target_transform=None):
        self.transform = transform
        self.train = train
        self.target_transform = target_transform
        self.data = pd.read_csv(data_path, sep=',')

    ############ get data ###########
    def __getitem__(self, index):
        labels = {'setosa': 0, 'virginica': 1, 'versicolor': 2}

        if self.train:
            feature, label = self.data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]
        else:
            feature, label = self.data.iloc[index:index + 1, 0:-1].values.reshape(4), self.data.iloc[index:index + 1, -1].values[0]

        return feature, int(labels[label])


    ############ get data length ###########
    def __len__(self):
        return len(self.data)