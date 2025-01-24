from data import getTrain
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from sklearn.tree import DecisionTreeRegressor

def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    params = {
        'criterion': ["squared_error", "friedman_mse", 'absolute_error', 'poisson'],
        'max_depth': Integer(5,400),
        'min_samples_split': Integer(2,100),
        'min_samples_leaf': Integer(2,100),
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = BayesSearchCV(estimator=DecisionTreeRegressor(), search_spaces=params, n_jobs=-1, cv=cv, n_iter=50, n_points=3)
    model.fit(x,y)
    # 0.8123064098936387
    print(model.score(x_val,y_val))
    print(model.best_params_)

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()