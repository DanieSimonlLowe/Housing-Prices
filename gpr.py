from data import getTrain
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.gaussian_process.kernels import RBF
from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Sum as KSum
from skopt.space import Real, Integer
from sklearn.model_selection import RepeatedKFold
from skopt import BayesSearchCV

class CustomGPR(BaseEstimator, RegressorMixin):
    def __init__(self,alpha=0.5, length_scale=1.0, noise_level=1.0):
        self.alpha = alpha
        self.model = None
        self.length_scale = length_scale
        self.noise_level = noise_level

        self._create_model()
    
    def _create_model(self):
        kernal = KSum(RBF(self.length_scale),WhiteKernel(self.noise_level))

        self.model = GaussianProcessRegressor(alpha=self.alpha,kernel=kernal)
    
    def fit(self, x, y):
        return self.model.fit(x,y)

    def predict(self, x):
        return self.model.predict(x)

    def score(self, x, y):
        return self.model.score(x,y)
    
    def get_params(self, deep=True):
        # Return parameters for cloning and grid search
        return {
            'alpha':self.alpha,
            'length_scale': self.length_scale,
            'noise_level': self.noise_level
        }
    
    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        self._create_model()
        return self


def makeModel():
    params = {
        'alpha': Real(1e-10,1e3, 'log-uniform'),
        'length_scale': Real(1e-10,1e3, 'log-uniform'),
        'noise_level': Real(1e-10,1e3, 'log-uniform'),
    }

    x, y = getTrain()
    x_val = x[-100:]
    y_val = y[-100:]
    x = x[:-100]
    y = y[:-100]


    cv = RepeatedKFold(n_splits=5, n_repeats=1, random_state=1)
    model = BayesSearchCV(estimator=CustomGPR(), search_spaces=params, n_jobs=-1, n_points=3, cv=cv, n_iter=50, verbose=3)
    model.fit(x,y)
    # 0.9263857343077551
    print(model.score(x_val,y_val))
    
    return model

if (__name__ == '__main__'):
    #getHistagram()
    makeModel()