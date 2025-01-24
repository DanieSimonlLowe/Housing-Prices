from data import getTrain
from sklearn.linear_model import BayesianRidge
from sklearn.ensemble import BaggingRegressor


def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    model = BayesianRidge()
    model.fit(x,y)
    # 0.9112801831575257
    print(model.score(x_val,y_val))
    
    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()