from data import getTrain
from sklearn.linear_model import LassoLars
from skopt import BayesSearchCV
from skopt.space import Real

def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    model = LassoLars(alpha=0.1)
    model.fit(x,y)
    # is bad
    # -0.002048930911647151
    print(model.score(x_val,y_val))

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()