import pipeline as p
from flask import Flask
from flask import request
from flask import jsonify
import json
import pandas as pd

def make_prediction(ts):
    status = 'SUCCESS'
    ts = ts.to_frame()
    X = p.pipeline.fit_transform(ts)
    p.estimator.fit(X)
    forecast = p.estimator.forecast()
    scores = p.estimator.evaluate()

    output = {}
    output['status'] = status
    output['forecast'] = forecast.to_json()
    output['scores'] = scores

    return output

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return("Server is running!")


@app.route("/predict", methods=["POST"])
def get_forecast():
    try:
        requestData = request.data
    except Exception as e:
        return 'An error occured during request data: {}'.format(e)
        
    try:
        ts = pd.read_json(requestData, typ='series', orient='records')
        # ts = pd.Series(data)
    except Exception as e:
        return 'An error occured during data transformation into series: {}'.format(e)

    try:
        response = make_prediction(ts)
    except Exception as e:
        return 'An error occured during pipeline process: {}'.format(e)

    
    return jsonify(response)

if __name__ =='__main__':
    app.run(debug=False)