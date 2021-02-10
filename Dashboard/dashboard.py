import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd
import json
import requests
from load_data import get_data, get_forecast

ip = requests.get('https://checkip.amazonaws.com').text.strip()
# ip = '18.156.118.16'
url = 'http://{}:5000/train_predict'.format(ip)

df = get_data()

app = dash.Dash(__name__)

server = app.server

app.layout = html.Div(children=[
    html.Div([
    dcc.Dropdown(
        id="ticker",
        options=[{"label": x, "value": x}
                 for x in df['Store'].unique()],
        value= df['Store'].unique()[0],
        clearable=False,
    ),
    html.H1(children='Rossman sales'),
    html.Div(children='''
            Daily breakdown by shop id.
        '''),
    dcc.Graph(id="time-series-chart"),
    ]),

    html.Div([
        html.Div(children='''
            Last 30 days of sales with forecast.
        '''),
    dcc.Graph(id="forecast-chart"),
    ]),
])

@app.callback(
    Output("time-series-chart", "figure"), 
    [Input("ticker", "value")])
def display_time_series(ticker):
    fig = px.line(df[df['Store']==ticker]['Sales'])
    return fig

@app.callback(
    Output("forecast-chart", "figure"), 
    [Input("ticker", "value")])
def display_forecast(ticker):
    data = df[df['Store']==ticker]['Sales'].sort_index()
    fig = px.line(data[-30:])
    forecast = get_forecast(url, data)
    fig.add_scatter(x= forecast.index ,y = forecast['forecast'], mode='lines', name='Forecast')
    fig.add_scatter(
        name='Upper Bound',
        x=forecast.index,
        y=forecast['w_upper'],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=1),
        showlegend=False
    )
    fig.add_scatter(
        name='Lower Bound',
        x=forecast.index,
        y=forecast['w_lower'],
        marker=dict(color="#444"),
        line=dict(width=1),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=False
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
