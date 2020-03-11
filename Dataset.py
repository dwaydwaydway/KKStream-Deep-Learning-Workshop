import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils import data

class Dataset(data.Dataset):
    def __init__(self, train_df, label_df=None, training=True):
        self.train_df = train_df
        self.label_df = label_df
        self.training = training

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, index):
        if self.training:
            return np.asarray(self.train_df[index]).astype(float), np.asarray(self.label_df[index]).astype(float)
        else:
            return np.asarray(self.train_df[index]).astype(float)