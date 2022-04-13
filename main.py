# pip install streamlit fbprophet yfinance plotly
import streamlit as st
from datetime import date
import requests
import numpy as np
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
from plotly import graph_objs as go
import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly
import plotly.offline as py
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_components_plotly
from PIL import Image
import datetime as dt
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import neuralprophet
from PIL import Image
from prophet import Prophet
from prophet.diagnostics import performance_metrics
from prophet.diagnostics import cross_validation
from prophet.plot import plot_cross_validation_metric
import base64
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_plotly
import plotly.offline as py
from prophet.plot import add_changepoints_to_plot
from prophet.plot import plot_components_plotly

image = Image.open('banner.jpg')
st.image(image, caption='Dadehkav Stock Prediction App')
st.title('Stock Prediction App')

today = dt.date.today()

before = today - dt.timedelta(days=7)
start_date = st.sidebar.date_input('Start date', before)
end_date = st.sidebar.date_input('End date', today)

if start_date < end_date:
    st.sidebar.success('Start date: `%s`\n\nEnd date:`%s`' % (start_date, end_date))
else:
    st.sidebar.error('Error: End date must fall after start date.')

st.title("Forcaster")

function_list = ['fbprophet', 'Neural Networks']
sidebar_function = st.sidebar.selectbox("Choose the forecasting method", function_list)
crypotocurrencies = (
    'BTC-USD', 'ETH-USD', 'BNB-USD', 'ADA-USD', 'SAND-USD', 'MANA-USD', 'XRP-USD', 'LTC-USD', 'EOS-USD', 'XLM-USD',
    'TRX-USD', 'ETC-USD', 'SHIB-USD', 'DOGE-USD', 'TRX-USD', 'SOL-USD', 'FTM-USD', 'MATIC-USD',)

selected_stock = st.selectbox('Select dataset for prediction', crypotocurrencies)

n_years = st.slider('Hours of prediction:', 1, 24)
period = n_years / 24


@st.cache
def load_data(ticker):
    data = yf.download(ticker, start_date, end_date, interval='1min')
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.tail())

# Plot raw data
fig = px.line(data, x='Date', y='Close')
st.plotly_chart(fig)

# Prophet model
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

if sidebar_function == "Neural Networks":
    st.write("running the code for Neural Networks..."
             "IT MAY TAKE A WHILE")
    model = neuralprophet.NeuralProphet(growth="linear",
                                        n_changepoints=14,
                                        # changepoints_range=0.8,
                                        # trend_reg=0,
                                        # trend_reg_threshold=False,
                                        yearly_seasonality=False,
                                        weekly_seasonality=False,
                                        daily_seasonality=8,
                                        seasonality_mode="multiplicative",
                                        epochs=150,
                                        loss_func="Huber",
                                        normalize="minmax",
                                        impute_missing=True,
                                        num_hidden_layers=3,
                                        d_hidden=1,
                                        batch_size=36)
    # model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    metrics = model.fit(df_train, freq='1min', progress='bar')
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

else:
    st.write("running the code for fbprophet..."
             "IT MAY TAKE A WHILE")
    m = Prophet(seasonality_mode='multiplicative', seasonality_prior_scale=5, daily_seasonality=8,
                weekly_seasonality=False, yearly_seasonality=False)
    # m.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    # m.add_country_holidays(country_name='US')
    m.fit(df_train)
    fut = m.make_future_dataframe(periods=period, freq='1min')
    forecas = m.predict(fut)
    st.write("Forecast Results")
    figf = plot_plotly(m, forecas, trend=True, figsize=(700, 600))
    st.plotly_chart(figf)
    st.write("Forecast components")
    figff = plot_components_plotly(m, forecas, figsize=(700, 175))
    st.plotly_chart(figff)
