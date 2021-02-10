import pandas as pd
import gc
import json
import requests

def get_data():
    df = pd.read_csv('input/rossman_train.csv', parse_dates=['Date'], index_col='Date', dtype={'StateHoliday':str})

    if 'df' in locals():
        light_df = df[ ['Store', 'Sales', 'Customers'] ].copy()
        
        del df
        gc.collect() #release memory 

    top_stores = light_df['Store'].value_counts().head(5).index
    multi_ts = light_df[light_df['Store'].isin(top_stores)]
    return multi_ts

def get_forecast(url, data):
    data = data.to_json()
    response = requests.post(url, data=data)
    r = response.json()
    mae = r['scores']['test_mae']
    weekday_mae = r['weekday_scores']
    weekday_mae = {int(k):v for k,v in weekday_mae.items()}
    f = pd.read_json(r['forecast'], typ='series', orient='records').to_frame()
    f.columns = ['forecast']
    f['lower'] = f['forecast'] - mae
    f['upper'] = f['forecast'] + mae
    f['dayofweek'] = f.index.dayofweek
    f['w_lower'] = f.apply(lambda x: x['forecast'] - weekday_mae[x['dayofweek']], axis=1)
    f['w_upper'] = f.apply(lambda x: x['forecast'] + weekday_mae[x['dayofweek']], axis=1)
    return f.clip(0)