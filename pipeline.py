from sklearn.pipeline import Pipeline
import xgboost as xgb
import preprocessors as pp

import config as cfg

model = xgb.XGBRegressor(**cfg.xgb_params)

pipeline = Pipeline(
    [
        ('log transformer',
        pp.LogTransformer(lambda_boxcox = cfg.lambda_boxcox)),
        
        ('feature extractor',
        pp.FeatureExtractor(shift=cfg.shift, freq=cfg.freq, rolls = cfg.rolls)),
        
        ('feature selector',
        pp.FeatureSelector(model = model, shift = cfg.shift))
        
    ]
)

estimator = pp.Estimator(model=model, train_size=0.8, lambda_boxcox=cfg.lambda_boxcox, shift=cfg.shift)

