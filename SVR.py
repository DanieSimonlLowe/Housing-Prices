from data import getTrain
from sklearn.svm import SVR
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
        'coef0': Real(0,10),
        'C': Real(1e-5,100, 'log-uniform'),
        'epsilon': Real(0,100),
        'shrinking': [False, True],
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = BayesSearchCV(estimator=SVR(), search_spaces=params, n_jobs=-1, n_points=3, cv=cv, n_iter=50, verbose=3)
    model.fit(x,y)
    # 0.9221894561940815
    print(model.score(x_val,y_val))
    print(model.best_params_)

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()