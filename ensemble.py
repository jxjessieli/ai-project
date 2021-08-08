import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
import math
from sklearn.linear_model import Ridge, LinearRegression
from argparse import ArgumentParser
from model import MSLELoss, MLPModel
from utils import FeatureDataset
import random
from sklearn.metrics import mean_squared_log_error as msle

ENSEMBLE_MODELS = ['model1', 'model2'] # saved models


def main(config, ENSEMBLE_MODELS):
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if torch.cuda.is_available():
        config.device = 'cuda:0'
        print(config.device)
    else:
        config.device = 'cpu'

    train_dataset = FeatureDataset(config.train_path)
    val_dataset = FeatureDataset(config.val_path)
    test_dataset = FeatureDataset(config.test_path)

    # Inference using saved models
    preds_train_models = pd.DataFrame()
    preds_val_models = pd.DataFrame()
    preds_test_models = pd.DataFrame()

    for model in ENSEMBLE_MODELS:
        # Read saved predictions
        preds_train_models[model] = pd.read_csv(config.save_dir + model + '_train_pred.csv')['train']
        preds_val_models[model] = pd.read_csv(config.save_dir + model + '_val_pred.csv')['val']
        preds_test_models[model] = pd.read_csv(config.save_dir + model + '_test_pred.csv')['test']

    if config.en_model == 'regression':
        # Stacking using regression
        en_model = LinearRegression(fit_intercept=False)
        en_model.fit(preds_train_models, train_dataset.y)

        val_pred = np.around(en_model.predict(preds_val_models))
        test_pred = np.around(en_model.predict(preds_test_models))
    elif config.en_model == 'average':
        # Model averaging
        val_pred = preds_val_models.mean(axis=1).round().to_numpy()
        test_pred = preds_test_models.mean(axis=1).round().to_numpy()

    # Calculate MSLE loss

    val_msle = msle(val_pred, val_dataset.y)
    print('Validation loss: ' + str(val_msle))

    test_msle = msle(test_pred, test_dataset.y)
    print('Test loss: ' + str(test_msle))

##########################
parser = ArgumentParser()
parser.add_argument('--seed', type=int, default=2021)
parser.add_argument('--save_dir', type=str, default='./data/models/')
parser.add_argument('--train_path', type=str, default='./data/train.feather')
parser.add_argument('--val_path', type=str, default='./data/val.feather')
parser.add_argument('--test_path', type=str, default='./data/test.feather')
parser.add_argument('--en-model', type=str, default='average', help='ensemble model - choose from regression or average')


config = parser.parse_args()
print(config)

main(config, ENSEMBLE_MODELS)
