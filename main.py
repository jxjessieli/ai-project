import torch
from torch.utils.data import Dataset 
import pandas as pd 
from torch.utils.data.dataloader import DataLoader 
from utils import FeatureDataset


train_dataset = FeatureDataset('./data/train.feather')
val_dataset = FeatureDataset('./data/val.feather')
test_dataset = FeatureDataset('./data/test.feather')
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_dataloader = DataLoader(val_dataset)
test_dataloader = DataLoader(test_dataset)

train_features, train_entities, train_mentions, train_hashtags, train_urls, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Entities batch shape: {train_entities.size()}")
print(f"Mentions batch shape: {train_mentions.size()}")
print(f"Hashtags batch shape: {train_hashtags.size()}")
print(f"URLs batch shape: {train_urls.size()}")
print(f"Labels batch shape: {train_labels.size()}")

