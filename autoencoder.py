import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class TabularAutoencoder(nn.Module):
    def __init__(self, input_dim=67, latent_dim=20, alpha=1.0, dropout_rate=0.2,
                 inner1 = 45, inner2 = 30):
        super(TabularAutoencoder, self).__init__()
        if (inner2 > inner1):
            inner1, inner2 = inner2, inner1
        
        inner1 = max(inner1,latent_dim)
        inner2 = max(inner2,latent_dim)

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, inner1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inner1, inner2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inner2, latent_dim)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, inner2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inner2, inner1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(inner1, input_dim)
        )

        self.alpha = alpha

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed
    
    def fit(self, X):
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        optimizer = optim.Adam(self.parameters(), lr=0.1, weight_decay=self.alpha)
        MSELoss = nn.MSELoss()
        self.decoder.train()
        self.encoder.train()
        for epoch in range(300):

            # Forward pass
            outputs = self.decoder(self.encoder(X_tensor))
            loss = MSELoss(outputs, X_tensor)
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32)
        else:
            X_tensor = torch.tensor(X.values, dtype=torch.float32)
        self.encoder.eval()
        return self.encoder(X_tensor).detach().numpy()


from sklearn.linear_model import BayesianRidge
from data import getTrainBase, loadDataBase
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV
import pandas as pd

class AutoencoderWrapper:
    def __init__(self, **params):
        global data
        self.model = TabularAutoencoder()
        self.model2 = BayesianRidge()
        self.data = data
        if len(params) == 0:
            self.params = {
                'latent_dim': 20,
                'alpha': 1.0,
                'dropout_rate': 0.2,
                'inner1': 45,
                'inner2': 30
            }
        else:
            self.params = params
            self.model = TabularAutoencoder(**params)

    def set_params(self, **params):
        self.model = TabularAutoencoder(**params)
        self.params = params
        return self

    def get_params(self, deep=True):
        # Return parameters for cloning and grid search
        return self.params
    
    def fit(self, X, y):
        self.model.fit(self.data)
        latent_X = self.model.transform(X)
        self.model2.fit(latent_X,y)
        return self

    def score(self, X, y):
        """
        Custom scorer that trains BayesianRidge on a subset of the data
        and scores it on the remaining data.
        """
        latent_X = self.model.transform(X)

        return self.model2.score(latent_X, y)

data = None

def makeEncoder():
    x, y = getTrainBase()
    x = x[:-100]
    y = y[:-100]

    train_data = loadDataBase(pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\home-prices\\train.csv'))
    test_data = loadDataBase(pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\home-prices\\test.csv'))
    global data
    data = pd.concat([train_data, test_data], ignore_index=True)

    params = {
        'latent_dim': (3,50),
        'inner1': (20,60),
        'inner2': (20,60),
        'alpha': (1e-10,1e5),
        'dropout_rate': (1e-10,0.6),
    }
    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = BayesSearchCV(estimator=AutoencoderWrapper(), search_spaces=params, n_jobs=-1, cv=cv, n_points=1, n_iter=100, verbose=3)

    model.fit(x,y)

    print(model.score(x,y))
    print(model.best_params_)

if (__name__ == '__main__'):
    #getHistagram()
    makeEncoder()