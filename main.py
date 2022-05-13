# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import requests
import numpy as np
import yfinance as yf
from plotly import graph_objs as go
import pandas as pd
import numpy as np
import base64
import plotly.offline as py
from PIL import Image
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import neuralprophet
from PIL import Image
import base64
import plotly.offline as py

image = Image.open('banner.jpg')
st.image(image, caption='Dadehkav Stock Prediction App')
st.title('Stock Prediction App 90m Intervals')

today = dt.date.today()

before = today - dt.timedelta(days=45)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

st.title("Forcaster")

function_list = ['Neural Networks']
sidebar_function = st.sidebar.selectbox("Choose the forecasting method", function_list)
crypotocurrencies = (
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SAND-USD', 'MANA-USD', 'XRP-USD', 'LTC-USD', 'EOS-USD', 'XLM-USD',
    'TRX-USD', 'ETC-USD', 'SHIB-USD', 'DOGE-USD', 'TRX-USD', 'SOL-USD', 'FTM-USD', 'MATIC-USD',)

selected_stock = st.selectbox('Select dataset for prediction', crypotocurrencies)

n_years = st.slider('Days of prediction:', 7, 90)
period = n_years * 16


@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date, interval='90m')
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data

# Prophet model

df_train = data[['Datetime', 'Close']]
df_train = df_train.rename(columns={"Datetime": "ds", "Close": "y"})
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce', utc=True )
df_train['ds'] = df_train['ds'].dt.strftime('%Y-%m-%d %H:%M')
r = px.line(df_train, x='ds', y='y')
st.write(r)
if sidebar_function == "Neural Networks":
    st.write("running the code for Neural Networks..."
             "IT MAY TAKE A WHILE")
     model = NeuralProphet(
     growth="discontinuous",  # Determine trend types: 'linear', 'discontinuous', 'off'
     #changepoints=None, # list of dates that may include change points (None -> automatic )
     n_changepoints=25,
     #changepoints_range=0.8,
     #trend_reg=0,
     #trend_reg_threshold=False,
     yearly_seasonality='auto',
     weekly_seasonality='auto',
     daily_seasonality='auto',
     seasonality_mode="multiplicative",
     #seasonality_reg=1,
     #n_forecasts=60,
     #n_lags=60,
     #ar_reg= 1,
     num_hidden_layers=2,
     d_hidden=2,     # Dimension of hidden layers of AR-Net
     #ar_sparsity=None,  # Sparcity in the AR coefficients
     #learning_rate=None,
     epochs=200,
     loss_func='Huber',
     collect_metrics= True,
     normalize="standardize",  # Type of normalization ('minmax', 'standardize', 'soft', 'off')
     impute_missing=True,
     #global_normalization=True,
     #log_level=None, # Determines the logging level of the logger object
     batch_size=32)
        
    # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    metrics = model.fit(df_train, freq='auto', progress='bar')
    future = model.make_future_dataframe(df_train, periods=period, n_historic_predictions=len(df_train))
    forecast = model.predict(future)
    st.write("Forecast Results")
    fign = model.plot(forecast)
    st.pyplot(fign)
    st.write("Forecast components")
    fig_comp = model.plot_components(forecast)
    st.write(fig_comp)
    fig_param = model.plot_parameters()
    st.pyplot(fig_param)
    st.write('Interactive chart')
    figu = px.line(forecast, x='ds', y=['yhat1', 'y'])
    st.write(figu)
