from data import getTrain
from sklearn.neighbors import KNeighborsRegressor
from skopt.space import Real, Integer
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV


def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    params = {
        'n_neighbors': Integer(1,50),
        'weights': ['uniform', 'distance']
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = BayesSearchCV(estimator=KNeighborsRegressor(), search_spaces=params, n_jobs=-1, n_points=3, cv=cv, n_iter=50, verbose=3)

    model.fit(x,y)
    # is bad
    # 0.7353090069752763
    print(model.score(x_val,y_val))

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()