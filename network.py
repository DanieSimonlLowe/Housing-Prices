from data import getTrain
from sklearn.neural_network import MLPRegressor
from skopt.space import Real, Integer
from sklearn.base import BaseEstimator, RegressorMixin
from skopt.space import Real, Integer
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV


class CustomMLP(BaseEstimator, RegressorMixin):
    def __init__(self,hidden_layer_1=10, hidden_layer_2=12, hidden_layer_3=12, hidden_layer_4=10, 
                 hidden_layer_5=10, alpha=0.5):
        self.hidden_layer_1 = hidden_layer_1
        self.hidden_layer_2 = hidden_layer_2
        self.hidden_layer_3 = hidden_layer_3
        self.hidden_layer_4 = hidden_layer_4
        self.hidden_layer_5 = hidden_layer_5
        self.alpha = alpha
        self.model = None
        self._create_model()
    
    def _create_model(self):
        self.model = MLPRegressor(hidden_layer_sizes=[ num
            for num in sorted([self.hidden_layer_1, self.hidden_layer_2, self.hidden_layer_3, self.hidden_layer_4, self.hidden_layer_5], reverse=True) if num > 0
        ],alpha=self.alpha)
    
    def fit(self, x, y):
        return self.model.fit(x,y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x,y)
    
    def get_params(self, deep=True):
        # Return parameters for cloning and grid search
        return {
            'hidden_layer_1': self.hidden_layer_1,
            'hidden_layer_2': self.hidden_layer_2,
            'hidden_layer_3': self.hidden_layer_3,
            'hidden_layer_4': self.hidden_layer_4,
            'hidden_layer_5': self.hidden_layer_5,
            'alpha':self.alpha,
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._create_model()
        return self

def makeModel():
    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]

    params = {
        'hidden_layer_1': Integer(3, 50),  # Size of the first hidden layer
        'hidden_layer_2': Integer(0, 30),  # Size of the second hidden layer (0 means no layer)
        'hidden_layer_3': Integer(0, 30),  # Size of the third hidden layer
        'hidden_layer_4': Integer(0, 30),  # Size of the third hidden layer
        'hidden_layer_5': Integer(0, 30),  # Size of the third hidden layer
        'alpha': Real(1e-5, 1e5, 'log-uniform'),
    }

    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = BayesSearchCV(estimator=CustomMLP(), search_spaces=params, n_jobs=-1, n_points=3, cv=cv, n_iter=50, verbose=3)
    model.fit(x,y)
    # 0.89
    print(model.score(x_val,y_val))
    print(model.best_params_)

    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()