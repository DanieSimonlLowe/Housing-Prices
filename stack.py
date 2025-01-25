from AdaBoost import makeModel as makeAda
from forest import makeModel as makeForest
from neighbors import makeModel as makeNeigh
from network import makeModel as makeNetwork
from Ridge import makeModel as makeRidge
from SVR import makeModel as makeSVR
from tree import makeModel as makeTree
from gpr import makeModel as makeGPR
from LGBM import makeModel as makeLGBM
from xgb import makeModel as makeXGB
from cat import makeModel as makeCat
from data import getTrain, loadData, getScalerY
import numpy as np
from sklearn.linear_model import Lasso
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    models = [
        makeAda(),
        makeForest(),
        makeNeigh(),
        makeNetwork(),
        makeRidge(),
        makeSVR(),
        makeTree(),
        makeGPR(),
        makeLGBM(),
        makeXGB(),
        makeCat(),
    ]

    #data = np.column_stack([model.fit(x_val,y_val) for model in models])
    params = {
        'alpha': Real(1e-5,1e5,'log-uniform')
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    meta_model = BayesSearchCV(estimator=Lasso(fit_intercept=True), search_spaces=params, n_jobs=-1, n_points=3, cv=cv, n_iter=50, verbose=3)
    meta_model.fit(np.column_stack([model.predict(x_val) for model in models]),y_val)

    test_data_base = pd.read_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\home-prices\\test.csv')
    test_data = loadData(test_data_base)
    predictions = meta_model.predict(np.column_stack([model.predict(test_data) for model in models]))
    predictions = getScalerY().inverse_transform(predictions.reshape(-1, 1)).flatten()

    output = pd.DataFrame({'Id': test_data_base.Id, 'SalePrice': predictions})
    output.to_csv('C:\\Users\\Danie\\Desktop\\work\\kaggle\\home-prices\\submission.csv', index=False)
    print("Your submission was successfully saved!")

if (__name__ == '__main__'):
    makeModel()
