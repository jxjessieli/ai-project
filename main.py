import torch
import pandas as pd
from torch.serialization import load
from torch.utils.data.dataloader import DataLoader
from utils import FeatureDataset
from argparse import ArgumentParser
import random
import numpy as np
import torch.nn as nn
from model import MSLELoss, MLPModel
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

def get_varlen_features_size(train_dataloader):
    max_entities = 0
    max_mentions = 0
    max_hashtags = 0
    max_urls = 0
    for step, (train_features, train_entities, train_mentions, train_hashtags, train_urls, train_labels) in enumerate(train_dataloader):
        max_entities = max(torch.max(train_entities).item(), max_entities)
        max_mentions = max(torch.max(train_mentions).item(), max_mentions)
        max_hashtags = max(torch.max(train_hashtags).item(), max_hashtags)
        max_urls = max(torch.max(train_urls).item(), max_urls)
    print(max_entities, max_mentions, max_hashtags, max_urls)
    return max_entities, max_mentions, max_hashtags, max_urls

def load_data(config):
    train_dataset = FeatureDataset(config.train_path)
    val_dataset = FeatureDataset(config.val_path)
    test_dataset = FeatureDataset(config.test_path)

    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size)

    return train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader

def evaluate(config):
    ### LOAD DATA ###
    _, _, test_dataset, train_dataloader, val_dataloader, test_dataloader = load_data(config)
    best_model = MLPModel(config)
    best_model.load_state_dict(copy.deepcopy(torch.load(str(config.meanpooling)+'best_model.pth', torch.device(config.device))))
    best_model.to(config.device)
    test_loss = 0.0
    train_preds = []
    val_preds = []
    test_preds = []
    with torch.no_grad():
        msle = MSLELoss()
        for step, (train_features, train_entities, train_mentions, train_hashtags, train_urls, train_labels) in enumerate(train_dataloader):
            pred = best_model(train_features.to(config.device), train_entities.to(config.device), train_mentions.to(config.device), train_hashtags.to(config.device), train_urls.to(config.device))
            train_preds.extend(pred.tolist())

        for step, (val_features, val_entities, val_mentions, val_hashtags, val_urls, val_labels) in enumerate(val_dataloader):
            pred = best_model(val_features.to(config.device), val_entities.to(config.device), val_mentions.to(config.device), val_hashtags.to(config.device), val_urls.to(config.device))
            val_preds.extend(pred.tolist())
                
        for step, (test_features, test_entities, test_mentions, test_hashtags, test_urls, test_labels) in enumerate(test_dataloader):
            pred = best_model(test_features.to(config.device), test_entities.to(config.device), test_mentions.to(config.device), test_hashtags.to(config.device), test_urls.to(config.device))
            test_preds.extend(pred.tolist())
            test_loss += msle(pred.cpu(), test_labels.float()) * len(test_labels)
        print('Test loss: ', str(test_loss/len(test_dataset)))

    ### SAVE PREDICTIONS ###
    pd.DataFrame({'train': train_preds}).to_csv(config.save_dir + config.model_name + '_train_pred.csv')
    pd.DataFrame({'val': val_preds}).to_csv(config.save_dir + config.model_name + '_val_pred.csv')
    pd.DataFrame({'test': test_preds}).to_csv(config.save_dir + config.model_name + '_test_pred.csv')


def train(config):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    ### LOAD DATA ###
    train_dataset, val_dataset, test_dataset, train_dataloader, val_dataloader, test_dataloader = load_data(config)

    # get_varlen_features_size(train_dataloader)

    if torch.cuda.is_available():
        config.device = 'cuda:0'
        print(config.device)
    else:
        config.device = 'cpu'

    ### TRAINING ###
    model = MLPModel(config)
    model.to(config.device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    train_losses = []
    val_losses = []
    test_losses = []
    test_mses = []

    best_val_loss = np.inf

    for epoch in range(config.epochs):
        train_loss = 0.0
        val_loss = 0.0
        test_loss = 0.0
        test_mse = 0.0

        for step, (train_features, train_entities, train_mentions, train_hashtags, train_urls, train_labels) in enumerate(tqdm(train_dataloader)):

            optimizer.zero_grad()
            outputs = model(train_features.to(config.device), train_entities.to(config.device), train_mentions.to(config.device), train_hashtags.to(config.device), train_urls.to(config.device))
            train_loss_batch = criterion(outputs.cpu(), train_labels.float())
            train_loss_batch.backward()
            optimizer.step()
            train_loss += train_loss_batch.item() * len(train_labels)

        with torch.no_grad():
            ### EVALUATE ON VAL AND TEST SET ###
            for step, (val_features, val_entities, val_mentions, val_hashtags, val_urls, val_labels) in enumerate(val_dataloader):
                outputs = model(val_features.to(config.device), val_entities.to(config.device), val_mentions.to(config.device), val_hashtags.to(config.device), val_urls.to(config.device))
                val_loss += criterion(outputs.cpu(), val_labels.float()).item() * len(val_labels)
            print('epoch: {}, train_loss: {}, val_loss: {}'.format(epoch, train_loss/len(train_dataset), val_loss/len(val_dataset)))

            msle = MSLELoss()
            for step, (test_features, test_entities, test_mentions, test_hashtags, test_urls, test_labels) in enumerate(test_dataloader):
                pred = model(test_features.to(config.device), test_entities.to(config.device), test_mentions.to(config.device), test_hashtags.to(config.device), test_urls.to(config.device))
                test_mse += criterion(pred.cpu(), test_labels.float()).item() * len(test_labels)
                test_loss += msle(pred.cpu(), test_labels.float()).item() * len(test_labels)

            print('Test loss: ', str(test_loss/len(test_dataset)))
            print('Test mse: ', str(test_mse/len(test_dataset)))

        ### SAVE MODEL WITH THE LOWEST val_loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), str(config.meanpooling)+'best_model.pth')

        train_losses.append(train_loss / len(train_dataset))
        val_losses.append(val_loss / len(val_dataset))
        test_losses.append(test_loss / len(test_dataset))
        test_mses.append(test_mse / len(test_dataset))

    ### PLOT LOSSES ###
    plt.figure()
    plt.plot(range(1, config.epochs+1), train_losses, 'tab:green', label='Training loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./figures/' + str(config.meanpooling) + str(config.model_name) + 'train_loss.pdf', bbox_inches='tight', dpi=900)

    plt.figure()
    plt.plot(range(1, config.epochs+1), val_losses, 'tab:orange', label='Validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./figures/' + str(config.meanpooling) + str(config.model_name) + 'val_loss.pdf', bbox_inches='tight', dpi=900)

    plt.figure()
    plt.plot(range(1, config.epochs+1), test_losses, 'tab:blue', label='Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./figures/' + str(config.meanpooling) + str(config.model_name) + 'test_loss.pdf', bbox_inches='tight', dpi=900)

    plt.figure()
    plt.plot(range(1, config.epochs+1), test_mses, 'tab:purple', label='Test MSE')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('./figures/' + str(config.meanpooling) + str(config.model_name) + 'test_mse.pdf', bbox_inches='tight', dpi=900)

    ### EVALUATE ON TEST SET WITH THE BSET MODEL ###
    evaluate(config)

# if __name__ == 'main':
parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--batch_size', type=int, default=2048)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--save_dir', type=str, default='./data/models/')
parser.add_argument('--model_name', type=str, default='model1')
parser.add_argument('--train_path', type=str, default='./data/train.feather')
parser.add_argument('--val_path', type=str, default='./data/val.feather')
parser.add_argument('--test_path', type=str, default='./data/test.feather')
parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--entities_dim', type=int, default=45677)
parser.add_argument('--mentions_dim', type=int, default=64197)
parser.add_argument('--hashtags_dim', type=int, default=37489)
parser.add_argument('--urls_dim', type=int, default=7508)
parser.add_argument('--emb_dim', type=int, default=256)
parser.add_argument('--hidden_size', type=int, default=256)
parser.add_argument('--num_feature_size', type=int, default=28)
parser.add_argument('--meanpooling', default=False, action='store_true')
parser.add_argument('--train', default=False, action='store_true')

config = parser.parse_args()
print(config)

if config.train:
    train(config)
else:
    evaluate(config)