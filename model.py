import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

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
    

### READ DATA ###
train_df = pd.read_csv('train.csv', index_col=0)
X_train = train_df.loc[:, train_df.columns != 'target']
y_train = train_df.loc[:, train_df.columns == 'target']
val_df = pd.read_csv('val.csv', index_col=0)
X_val = val_df.loc[:, val_df.columns != 'target']
y_val = val_df.loc[:, val_df.columns == 'target']
test_df = pd.read_csv('test.csv', index_col=0)
X_test = test_df.loc[:, test_df.columns != 'target']
y_test = test_df.loc[:, test_df.columns == 'target']


### TRAINING ###
cols = ['#followers', '#friends', '#favorites', '#followers__#favorites', '#friends__#favorites_z', '#followers__#friends__#favorites', 'sentiment_pos_ce', 'sentiment_neg_ce', 'weekday_ce', 'hour_ce', 'day_ce', 'week_of_month_ce', 'TFIDF_svd_0', 'TFIDF_svd_1', 'TFIDF_svd_2', 'TFIDF_svd_3', 'TFIDF_svd_4', 'entities_ce', 'mentions_ce', 'hashtags_ce', 'urls_ce', 'url_domain_ce', 'user_stats_cluster_1000', 'user_topic_cluster_1000', 'user_stats_topic_cluster_1000']

inputDim = len(cols)
learningRate = 1e-4
epochs = 100
dropout = 0.25

model = neuralNet(inputDim, dropout)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learningRate)

train_losses = []
val_losses = []

for epoch in range(epochs):
    inputs = torch.tensor(X_train[cols].to_numpy()).float()
    labels = torch.tensor(y_train.to_numpy()).float()
    
    optimizer.zero_grad()
    
    outputs = model(inputs)
    
    train_loss = criterion(outputs, labels)
    
    train_loss.backward()
    
    optimizer.step()
    
    with torch.no_grad():
        inputs = torch.tensor(X_val[cols].to_numpy()).float()
        labels = torch.tensor(y_val.to_numpy()).float()
        
        outputs = model(inputs)
        val_loss = criterion(outputs, labels)
        
    print('epoch: {}, train_loss: {}, val_loss: {}'.format(epoch, train_loss.item(), val_loss.item()))
    train_losses.append(train_loss.item())
    val_losses.append(val_loss.item())
   
    
### PLOT LOSSES ###
plt.plot(range(1, epochs+1), train_losses, 'g', label='Training loss')
plt.plot(range(1, epochs+1), val_losses, 'b', label='Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


### EVALUATE ON TEST SET ###
with torch.no_grad():
    pred = model(torch.tensor(X_test[cols].to_numpy()).float())
    actual = torch.tensor(y_test.to_numpy()).float()
    msle = MSLELoss()
    print(msle(pred, actual))
        