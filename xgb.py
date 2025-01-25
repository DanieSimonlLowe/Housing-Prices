from data import getTrain
from xgboost import XGBRegressor


def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    model = XGBRegressor()
    model.fit(x,y)
    # 0.9153427481651306
    print(model.score(x_val,y_val))
    
    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()