lambda_boxcox = 0.31
freq = 'D'
rolls = [3,5,7,15,30,60]
shift = 30
train_size = 0.8

xgb_params = {'max_depth': 4, 
              'n_estimators': 100, 
              'learning_rate': 0.15, 
              'colsample_bytree': .7, 
              'subsample': 0.7, 
              'seed': 2018,
              'objective' : 'reg:squarederror'}