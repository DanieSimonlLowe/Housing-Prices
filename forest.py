from data import getTrain
from sklearn.ensemble import RandomForestRegressor
from skopt import BayesSearchCV
from skopt.space import Real, Integer

def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    model = RandomForestRegressor()
    model.fit(x,y)
    # 0.9256716348108855
    print(model.score(x_val,y_val))

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()