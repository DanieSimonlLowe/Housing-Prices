from data import getTrain
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import AdaBoostRegressor

def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    model = AdaBoostRegressor(estimator=LinearRegression())
    model.fit(x,y)
    # 0.8168838506876233 # dose worse then no aba boost
    print(model.score(x_val,y_val))

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()