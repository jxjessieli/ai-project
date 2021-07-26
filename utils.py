import pandas as pd 
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class FeatureDataset(Dataset):

    def __init__(self, file_name):
        
        # read csv file and load row data into variables
        file_out = pd.read_feather(file_name)
        x = file_out.iloc[:, :28].values
        entities = file_out.iloc[:, 28]
        mentions = file_out.iloc[:, 29]
        hashtags = file_out.iloc[:, 30]
        urls = file_out.iloc[:, 31]
        y = file_out.iloc[:, -1].values

        # convert to torch tensors
        # x = x.astype(float)
        self.x = torch.from_numpy(x)
        self.entities = entities.values.tolist()
        self.mentions = mentions.values.tolist()
        self.hashtags = hashtags.values.tolist()
        self.urls = urls.values.tolist()
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.x[index], self.entities[index], self.mentions[index], self.hashtags[index], self.urls[index], self.y[index]
        # return self.x[index],  self.mentions[index],self.y[index]