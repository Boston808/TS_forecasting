import numpy as np
import pandas as pd

import config as cfg
import math
from scipy.special import boxcox1p
from scipy.special import inv_boxcox1p

from statsmodels.tsa.api import ExponentialSmoothing

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE

from sklearn.metrics import mean_squared_error, mean_absolute_error as mae

convert_from_log=lambda x: inv_boxcox1p(x, cfg.lambda_boxcox)

def weekday_breakdown(item):
    output = {}
    for i in list(range(7)):
        output[i] = list(item[item['dayofweek']==i].iloc[:,:1].values.ravel())
    return output
    
def mape(y_true, y_pred): 
    y_true = np.array(y_true, dtype=np.float)
    y_pred = np.array(y_pred, dtype=np.float)
    
    return np.mean(np.abs((y_true - y_pred) / (y_true+1e-10) )) * 100

def rmse(y, y_pred):
    return math.sqrt(mean_squared_error(y, y_pred))

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
    
    def __init__(self, target=None, shift=30, freq='D', rolls=[30]):
        self.shift = shift
        self.target = target
        self.freq=freq
        self.rolls = rolls
        
    def fit(self, X, y=None):
        #accomodating the pipeline
        return self
    
    def transform(self, X):
        
        if self.target is None:
            self.target = X.columns[0]
            
        ix = pd.date_range(start=X.index[0], periods = len(X) + self.shift, freq = self.freq)
        X = X.reindex(ix)
        
        if self.freq == 'H':
            X['hour'] = X.index.hour
        X['day'] = X.index.day
        X['month'] = X.index.month
        X['dayofweek'] = X.index.dayofweek
        X['quarter'] = X.index.quarter
        X['dayofyear'] = X.index.dayofyear
        X['week'] = X.index.week
        X['year'] = X.index.year
        X['shift{}'.format(self.shift)] = X[self.target].shift(self.shift).fillna(method='bfill')
        
        for roll in self.rolls:
            X['rolling_mean{}'.format(roll)] = (X[self.target].
                                                shift(self.shift).
                                                rolling(roll).
                                                mean().
                                                fillna(method='bfill'))
        return X

class FeatureSelector(BaseEstimator, TransformerMixin):
    
    def __init__ (self, model, target=None,  shift=30):
        self.model = model
        self.target = target
        self.shift = shift
        
    def fit(self, X, y=None):
        if self.target is None:
            self.target = X.columns[0]
        
        current_X = X[:-self.shift].copy()
        rfe = RFE(self.model)
        feats = [feat for feat in X.columns if feat != self.target]
        rfe.fit(current_X[feats], current_X[self.target])
        f = rfe.get_support(1)
        filtered_feats = list(current_X[feats].columns[f])
        self.feats = [self.target] + filtered_feats
        return self
    
    def transform(self, X):
        
        return X[self.feats]

class Estimator():
    
    def __init__(self, model, train_size, lambda_boxcox, target=None, shift=30):
        self.model = model
        self.train_size = train_size
        self.target = target
        self.shift = shift
        self.convert_from_log = lambda x: inv_boxcox1p(x, lambda_boxcox)
    
    def fit(self, X):
        if self.target is None:
            self.target = X.columns[0]
            
        #split between current data and future dataframe
        self.X = X[:-self.shift]
        self.future_X = X[-self.shift:]
        
        #split current data between train and test set 
        train_len = round(len(self.X) * self.train_size)
        X_train = self.X[:train_len]
        X_test = self.X[train_len:]
        
        #filter out target variable from features
        self.feats = [feat for feat in X.columns if feat != self.target]
        #training the model
        self.model.fit(X_train[self.feats], X_train[self.target])
        #inverted conversion for target variable
        self.y_train = X_train[self.target].map(self.convert_from_log)
        self.y_test = X_test[self.target].map(self.convert_from_log)
        #predicting training and test data, inverted conversion and assigning to self vars
        y_train_pred = list(map(self.convert_from_log, self.model.predict(X_train[self.feats])))
        y_test_pred = list(map(self.convert_from_log, self.model.predict(X_test[self.feats])))
        
        self.y_train_pred = pd.Series(y_train_pred, index=self.y_train.index).fillna(0)
        self.y_test_pred = pd.Series(y_test_pred, index=self.y_test.index).fillna(0)
        
        return self
    
    def forecast(self):
        y_pred = self.model.predict(self.future_X[self.feats])
        y_pred = list(map(self.convert_from_log, y_pred))
        forecast =  pd.Series(data = y_pred, index = self.future_X.index, )
        return forecast
    
    def evaluate(self):
        scores = {}
        
        scores['train_mae'] = mae(self.y_train, self.y_train_pred)
        scores['train_rmse'] = rmse(self.y_train, self.y_train_pred)
        
        scores['test_mae'] = mae(self.y_test, self.y_test_pred.fillna(0))
        scores['test_rmse'] = rmse(self.y_test, self.y_test_pred.fillna(0))
        
        return scores

    def eval_dayofweek(self):
        true = self.y_test.to_frame().copy()
        pred = self.y_test_pred.fillna(0).to_frame().copy()
        
        for item in (true, pred):
            item['dayofweek'] = item.index.dayofweek
            
        true_b = weekday_breakdown(true)
        pred_b = weekday_breakdown(pred)
        
        weekday_mae = {}
        for k in true_b:
            weekday_mae[k] = mae(true_b[k], pred_b[k])
        
        return weekday_mae


def ts_weekday_eval(X, periods, train_size):
    train_len = round(len(X) * train_size)
    test_len = len(X) - train_len
    split = train_len
    y_pred_agg = pd.Series()
    
    while test_len > periods:
        model = ExponentialSmoothing(X[:split], trend='add', seasonal='add', freq='D').fit()
        y_pred = model.forecast(steps=periods)
        y_pred_agg = y_pred_agg.append(y_pred)
        split += periods
        test_len -=periods
        
    true = X[train_len:train_len+len(y_pred_agg)].to_frame().copy()
    pred = y_pred_agg.to_frame().copy()
    
    for item in (true, pred):
            item['dayofweek'] = item.index.dayofweek
    
    true_b = weekday_breakdown(true)
    pred_b = weekday_breakdown(pred)
        
    weekday_mae = {}
    for k in true_b:
        weekday_mae[k] = mae(true_b[k], pred_b[k])
    
    return weekday_mae


def ts_cv(X, periods, train_size):
    train_len = round(len(X) * train_size)
    test_len = len(X) - train_len
    split = train_len
    mean_errors = []
    root_errors = []
    while test_len > periods:
        model = ExponentialSmoothing(X[:split], trend='add', seasonal='add', freq='D').fit()
        y_pred = model.forecast(steps=periods)
        mean_errors.append(mae(X[split:split+periods], y_pred))
        root_errors.append(rmse(X[split:split+periods], y_pred))
        split += periods
        test_len -=periods
    return {'test_mae':np.mean(mean_errors), 'test_rmse':np.mean(root_errors)}