import streamlit as st
import pandas as pd
import numpy as np
import random
import torch
from sklearn.metrics import mean_squared_log_error as msle
import config
from sklearn.linear_model import LinearRegression

import sys
sys.path.append('../')
from utils import FeatureDataset

st.set_page_config(page_title='Retweet Prediction Challenge', page_icon=":shark:", layout='centered', initial_sidebar_state='auto')

st.title('COVID-19 Retweet Prediction Challenge')

# st.write('Select a model or a combination of models that will be trained using an ensemble method.')
model_1 = st.sidebar.checkbox('Model 1')
model_2 = st.sidebar.checkbox('Model 2')

ens = st.sidebar.selectbox('Select an ensemble method',
                    ('Regression', 'Average'))


ENSEMBLE_MODELS = []

if model_1:
    ENSEMBLE_MODELS.append('model1')
if model_2:
    ENSEMBLE_MODELS.append('model2')

### BUILD ENSEMBLE ###
random.seed(config.seed)
np.random.seed(config.seed)
torch.manual_seed(config.seed)

if torch.cuda.is_available():
    config.device = 'cuda:0'
else:
    config.device = 'cpu'

train_dataset = FeatureDataset(config.train_path)
val_dataset = FeatureDataset(config.val_path)
test_dataset = FeatureDataset(config.test_path)

preds_train_models = pd.DataFrame()
preds_val_models = pd.DataFrame()
preds_test_models = pd.DataFrame()

with st.spinner('Wait for it...'):
    for model in ENSEMBLE_MODELS:
        # Read saved predictions
        preds_train_models[model] = pd.read_csv(config.save_dir + model + '_train_pred.csv')['train']
        preds_val_models[model] = pd.read_csv(config.save_dir + model + '_val_pred.csv')['val']
        preds_test_models[model] = pd.read_csv(config.save_dir + model + '_test_pred.csv')['test']

if len(preds_train_models) != 0:
    
    if ens == 'Regression':
        # Regression
        en_model = LinearRegression(fit_intercept=False)
        en_model.fit(preds_train_models, train_dataset.y)

        val_pred = np.around(en_model.predict(preds_val_models))
        test_pred = np.around(en_model.predict(preds_test_models))

    elif ens == 'Average':
        # Average
        val_pred = preds_val_models.mean(axis=1).round().to_numpy()
        test_pred = preds_test_models.mean(axis=1).round().to_numpy()

    # Calculate MSLE loss
    val_msle = msle(val_pred, val_dataset.y)
    test_msle = msle(test_pred, test_dataset.y)

    st.subheader('View model results:')
    st.write('Validation MSLE: ' + str(val_msle))
    st.write('Test MSLE: ' + str(test_msle))
    st.subheader('View pre-processed data:')
    st.write('Validation Set')
    val_df = pd.read_feather('../data/val.feather')
    target = val_df['0']
    val_df.drop(labels=['0'], axis=1, inplace=True)
    val_df.insert(0, 'target', target)
    val_df = pd.concat([pd.DataFrame(data=val_pred, columns=['predicted']), val_df], axis=1)
    val_table = st.dataframe(val_df.head(100))

    st.write('Test Set') 
    test_df = pd.read_feather('../data/test.feather')
    target = test_df['0']
    test_df.drop(labels=['0'], axis=1, inplace=True)
    test_df.insert(0, 'target', target)
    test_df = pd.concat([pd.DataFrame(data=test_pred, columns=['predicted']), test_df], axis=1)
    test_table = st.dataframe(test_df.head(100))
        
else:
    st.info('Select a model or a combination of models that will be trained using an ensemble method.')
    st.image('https://images.unsplash.com/photo-1616469829167-0bd76a80c913?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=2550&q=80')