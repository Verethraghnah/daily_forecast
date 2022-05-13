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
st.title('Stock Prediction App (1H Interval)')

today = dt.date.today()

before = today - dt.timedelta(days=35)
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

n_years = st.slider('Hours of prediction:', 12, 72)
period = n_years * 24


@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date, interval='1h')
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())


# Prophet model

df_train = data[['index', 'Close']]
df_train = df_train.rename(columns={"index": "ds", "Close": "y"})
df_train['ds'] = pd.to_datetime(df_train['ds'], errors='coerce', utc=True )
df_train['ds'] = df_train['ds'].dt.strftime('%Y-%m-%d %H:%M')
r = px.line(df_train, x='ds', y='y')
st.write(r)
if sidebar_function == "Neural Networks":
    st.write("running the code for Neural Networks..."
             "IT MAY TAKE A WHILE")
    model = neuralprophet.NeuralProphet(growth="discontinuous",
                                        #n_changepoints=14,
                                        # changepoints_range=0.8,
                                        # trend_reg=0,
                                        # trend_reg_threshold=False,
                                        yearly_seasonality=False,
                                        weekly_seasonality='auto',
                                        daily_seasonality=8,
                                        seasonality_mode="multiplicative",
                                        epochs=250,
                                        loss_func="Huber",
                                        normalize="soft",
                                        impute_missing=True,
                                        num_hidden_layers=2,
                                        d_hidden=2,
                                        batch_size=16)
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
figure = px.line(forecast, x='ds', y=['yhat1', 'y'])
st.write(figure)
