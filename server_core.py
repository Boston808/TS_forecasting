import pipeline as p
from flask import Flask
from flask import request
from flask import jsonify
import json
import pandas as pd
import joblib
from preprocessors import convert_from_log, ts_cv, ts_weekday_eval
from sequence_pipeline import LogTransformer, FeatureExtractor, FeatureSelector
import requests
from statsmodels.tsa.api import ExponentialSmoothing
import config as cfg

def forecast_window(ts):
    status = 'SUCCESS'
    ts = ts.to_frame()
    X = p.pipeline.fit_transform(ts)
    p.estimator.fit(X)
    forecast = p.estimator.forecast().fillna(0)
    scores = p.estimator.evaluate()
    weekday_scores = p.estimator.eval_dayofweek()

    output = {}
    output['status'] = status
    output['forecast'] = forecast.to_json()
    output['scores'] = scores
    output['weekday_scores'] = weekday_scores

    return output

def forecast_sequence(ts, steps, model, pipeline):
    ts = ts.to_frame().copy()
    ts.columns = ['Sales']
    initial_len = len(ts)
    
    for step in list(range(1, steps + 1)):
        
        last_date = ts.iloc[[-1]].index
        last_date = last_date + pd.Timedelta(days = 1)
        
        ts = ts.append(pd.DataFrame(index=last_date)).copy()
        X = pipeline.transform(ts)
        prediction = model.predict(X.iloc[-1:])
        
        new_value = list(map(convert_from_log, prediction))
        ts.iloc[-1] = round(new_value[0])
    
    return ts[initial_len:]

def forecast_hw(ts):
    status = 'SUCCESS'
    es = ExponentialSmoothing(ts, trend='add', seasonal='add', freq='D').fit()
    es_y = es.forecast(steps=cfg.shift)
    scores = ts_cv(ts, cfg.shift, cfg.train_size)
    weekday_scores = ts_weekday_eval(ts, cfg.shift, cfg.train_size)

    output = {}
    output['status'] = status
    output['forecast'] = es_y.to_json()
    output['scores'] = scores
    output['weekday_scores'] = weekday_scores

    return output

app = Flask(__name__)
@app.route('/', methods=['GET'])
def hello():
    ip = requests.get('https://checkip.amazonaws.com').text.strip()
    return("Server is running on {}!".format(ip))


@app.route("/train_predict", methods=["POST"])
def train_and_predict():
    try:
        requestData = request.data
    except Exception as e:
        return 'An error occured during request data: {}'.format(e)
        
    try:
        ts = pd.read_json(requestData, typ='series', orient='records')
    except Exception as e:
        return 'An error occured during data transformation into series: {}'.format(e)

    try:
        response = forecast_window(ts)
    except Exception as e:
        return 'An error occured during pipeline process: {}'.format(e)

    return jsonify(response)


@app.route("/predict_sequence", methods=["POST"])
def get_forecast():
    try:
        requestData = request.data
    except Exception as e:
        return 'An error occured during request data: {}'.format(e)

    try:
        requestData = requestData.decode("utf-8")
        requestData = json.loads(requestData)
        ts = pd.read_json(requestData['data'], typ='series', orient='records')
    except Exception as e:
        return 'An error occured during data transformation into series: {}'.format(e)
        
    try:
        steps = requestData['steps']
        model = joblib.load(r'models\xgboost-rossman.sav')
        pipeline = joblib.load(r'models\pipeline-rossman.sav')
        forecast = forecast_sequence(ts, steps, model, pipeline)
    except Exception as e:
        return 'An error occured during pipeline process: {}'.format(e)
    response = {}
    response['forecast'] = forecast['Sales'].to_json()   
    return jsonify(response)

@app.route("/holt_winters", methods=["POST"])
def hp_forecast():
    try:
        requestData = request.data
    except Exception as e:
        return 'An error occured during request data: {}'.format(e)
        
    try:
        ts = pd.read_json(requestData, typ='series', orient='records')
    except Exception as e:
        return 'An error occured during data transformation into series: {}'.format(e)

    try:
        response = forecast_hw(ts)
    except Exception as e:
        return 'An error occured during fit process: {}'.format(e)
    
    return jsonify(response)

if __name__ =='__main__':
    app.run(debug=False, port=5000)