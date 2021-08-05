import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules import padding

class neuralNet(nn.Module):
    def __init__(self, inputSize, dropout):
        super(neuralNet, self).__init__()
        self.layer_1 = nn.Linear(inputSize, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.layer_2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer_3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer_4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.layer_1(x)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.layer_2(out)))
        out = self.dropout(out)
        out = F.relu(self.bn3(self.layer_3(out)))
        out = self.dropout(out)
        out = F.relu(self.layer_4(out))
        return out
    
class MSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    
    def forward(self, pred, actual):
        return self.mse(torch.log(pred+1), torch.log(actual+1))
    
class MLPModel(nn.Module):
    def __init__(self, config):
        super(MLPModel, self).__init__()
        self.config = config
        self.embedding_entities = nn.Embedding(config.entities_dim, config.emb_dim, padding_idx=0)
        # TODO: check padding index, add to other non-numerical features
        # TODO: add configuration 
        self.embedding_mentions = nn.Embedding(config.mentions_dim, config.emb_dim, padding_idx=0)
        self.embedding_hashtags = nn.Embedding(config.hashtags_dim, config.emb_dim, padding_idx=0)
        self.embedding_urls = nn.Embedding(config.urls_dim, config.emb_dim, padding_idx=0)
        # self.embedding_entities.weight.data.uniform_(-0.1, 0.1)
        # self.embedding_mentions.weight.data.uniform_(-0.1, 0.1)
        # self.embedding_hashtags.weight.data.uniform_(-0.1, 0.1)
        # self.embedding_urls.weight.data.uniform_(-0.1, 0.1)

        self.linear_entities = nn.Linear(config.emb_dim, config.hidden_size)
        self.linear_mentions = nn.Linear(config.emb_dim, config.hidden_size)
        self.linear_hashtags = nn.Linear(config.emb_dim, config.hidden_size)
        self.linear_urls = nn.Linear(config.emb_dim, config.hidden_size)

        self.layer_1 = nn.Linear(config.num_feature_size+4*config.hidden_size, 2048)
        self.bn1 = nn.BatchNorm1d(2048)
        self.layer_2 = nn.Linear(2048, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.layer_3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.layer_4 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, nu_features, entities, mentions, hashtags, urls):
        entities_embs = self.embedding_entities(entities)
        mentions_embs = self.embedding_mentions(mentions)
        hashtags_embs = self.embedding_hashtags(hashtags)
        urls_embs = self.embedding_urls(urls)

        hid_entities = self.linear_entities(self.dropout(entities_embs))
        hid_mentions = self.linear_mentions(self.dropout(mentions_embs))
        hid_hashtags = self.linear_hashtags(self.dropout(hashtags_embs))
        hid_urls = self.linear_urls(self.dropout(urls_embs).squeeze())

        # MEANPOOLING
        if self.config.meanpooling:
            mask_entities = hid_entities!=0
            mask_mentions = hid_mentions!=0
            mask_hashtags = hid_hashtags!=0
            hid_entities = hid_entities.sum(dim=1) / mask_entities.sum(dim=1)
            hid_mentions = hid_mentions.sum(dim=1) / mask_mentions.sum(dim=1)
            hid_hashtags = hid_hashtags.sum(dim=1) / mask_hashtags.sum(dim=1)
        else:
            hid_entities = torch.max(hid_entities, 1)[0]
            hid_mentions = torch.max(hid_mentions, 1)[0]
            hid_hashtags = torch.max(hid_hashtags, 1)[0]
        

        # TODO: check size of each layer, check dim of cat
        hidden = torch.cat([hid_entities.float(), hid_mentions.float(), hid_hashtags.float(), hid_urls.float(), nu_features.float()], dim=1)
        out = F.relu(self.bn1(self.layer_1(hidden)))
        out = self.dropout(out)
        out = F.relu(self.bn2(self.layer_2(out)))
        out = self.dropout(out)
        out = F.relu(self.bn3(self.layer_3(out)))
        out = self.dropout(out)
        out = F.relu(self.layer_4(out))

        return out.squeeze()

