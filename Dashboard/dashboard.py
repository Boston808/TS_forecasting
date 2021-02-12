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
    html.Div(children='''Breakdown by shop id.'''),
    html.Br(),
    dcc.RadioItems(
                id='agg_type',
                options=[{'label': i, 'value': i} for i in ['Daily', 'Weekly', 'Monthly']],
                value='Daily',
                labelStyle={'display': 'inline-block'}
        ),
    dcc.Graph(id="time-series-chart"),
    ]),

    html.Div([
        html.Div(children='''
            Last 30 days of sales with forecast.
        '''),
        html.Br(),
    dcc.RadioItems(
                id='err_type',
                options=[{'label': i, 'value': i} for i in ['Day of week MAE', 'Simple MAE']],
                value='Day of week MAE',
                labelStyle={'display': 'inline-block'}
        ),
    dcc.Graph(id="forecast-chart"),
    ]),
])


@app.callback(
    Output("time-series-chart", "figure"), 
    [Input("ticker", "value"),
    Input("agg_type", "value")])
def display_time_series(ticker, agg_type):
    fig = px.line(df[df['Store']==ticker]['Sales'].resample(agg_type[0].lower()).sum())
    
    fig.update_layout(
    title="Sales of shop id {}".format(ticker),
    xaxis_title="Date",
    yaxis_title="Amount in EUR",
    legend_title="Legend",
    )
    return fig

@app.callback(
    Output("forecast-chart", "figure"), 
    [Input("ticker", "value"),
    Input("err_type", "value")])
def display_forecast(ticker, err_type):
    data = df[df['Store']==ticker]['Sales'].sort_index()
    fig = px.line(data[-30:])
    forecast = get_forecast(url, data)
    
    if err_type == 'Day of week MAE':
        upper = 'w_upper'
        lower = 'w_lower'
    else:
        upper = 'upper'
        lower = 'lower'
    
    fig.add_scatter(x= forecast.index ,y = forecast['forecast'], mode='lines', name='Forecast')
    fig.add_scatter(
        name='Upper Bound',
        x=forecast.index,
        y=forecast[upper],
        mode='lines',
        marker=dict(color="#444"),
        line=dict(width=1),
        showlegend=True
    )
    fig.add_scatter(
        name='Lower Bound',
        x=forecast.index,
        y=forecast[lower],
        marker=dict(color="#444"),
        line=dict(width=1),
        mode='lines',
        fillcolor='rgba(68, 68, 68, 0.3)',
        fill='tonexty',
        showlegend=True
    )
    fig.update_layout(
    title="Forcasted sales of shop id {}".format(ticker),
    xaxis_title="Date",
    yaxis_title="Amount in EUR",
    legend_title="Legend",
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=False)
