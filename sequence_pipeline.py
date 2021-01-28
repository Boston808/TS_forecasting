from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFE
import joblib

class LogTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, target=None, lambda_boxcox=0.31):
        self.target = target
        self.lambda_boxcox = lambda_boxcox
    
    def fit(self, X, y=None):
        #accomodating the pipeline
        return self
    
    def transform(self, X):
        X=X.copy()
        
        if self.target is None:
            self.target = X.columns[0]
            
        X[self.target] = boxcox1p(X[self.target], self.lambda_boxcox)
        
        return X

class FeatureExtractor(BaseEstimator, TransformerMixin):
    
    def __init__(self, target=None, shifts=[30], freq='D', rolls=[30]):
        self.shifts = shifts
        self.target = target
        self.freq=freq
        self.rolls = rolls
        
    def fit(self, X, y=None):
        #accomodating the pipeline
        return self
    
    def transform(self, X):
        X = X.copy()
        if self.target is None:
            self.target = X.columns[0]
            
        if self.freq == 'H':
            X['hour'] = X.index.hour
        X['day'] = X.index.day
        X['month'] = X.index.month
        X['dayofweek'] = X.index.dayofweek
        X['quarter'] = X.index.quarter
        X['dayofyear'] = X.index.dayofyear
        X['year'] = X.index.year
        
        for shift in self.shifts:
            X['shift{}'.format(shift)] = X[self.target].shift(shift).fillna(method='bfill')
        
        for roll in self.rolls:
            X['rolling_mean{}'.format(roll)] = (X[self.target].
                                                shift(1).
                                                rolling(roll).
                                                mean().
                                                fillna(method='bfill'))
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__ (self, model, target=None):
        self.model = model
        self.target = target
        
    def fit(self, X, y=None):
        X= X.copy()
        if self.target is None:
            self.target = X.columns[0]
        
        rfe = RFE(self.model)
        feats = [feat for feat in X.columns if feat != self.target]
        rfe.fit(X[feats], X[self.target])
        f = rfe.get_support(1)
        filtered_feats = list(X[feats].columns[f])
        self.feats = filtered_feats
        return self
    
    def transform(self, X):
        X=X.copy()
        return X[self.feats]
