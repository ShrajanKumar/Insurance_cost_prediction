import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class OutlierCapper(BaseEstimator, TransformerMixin):
    def __init__(self, quantile_upper=0.95, quantile_lower=0.05):
        self.quantile_upper = quantile_upper
        self.quantile_lower = quantile_lower
        self.upper_limit = None
        self.lower_limit = None
    
    def fit(self, X, y=None):
        self.upper_limit = int(X.quantile(self.quantile_upper))
        self.lower_limit = int(X.quantile(self.quantile_lower))
        return self
    
    def transform(self, X):
        capped_values = np.where(X > self.upper_limit, self.upper_limit,
                                 np.where(X < self.lower_limit, self.lower_limit, X))
        return capped_values.reshape(-1, 1)  # Ensure output is 2D array
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
