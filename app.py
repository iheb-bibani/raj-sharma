import yfinance as yf
import pandas as pd
import streamlit as st
import numpy as np
import talib as ta
from PIL import Image

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor

st.title('Cryptocurrencies')

image = Image.open('cryptocurrencies.jpeg')
st.sidebar.image(image, caption='Cryptocurrencies')

symbols = ['BTC-USD','ETH-USD','XRP-USD','ADA-USD','MATIC-USD',
           'DOGE-USD','SOL-USD','DOT-USD','SHIB-USD','LTC-USD',
           'TRX-USD','AVAX-USD','UNI-USD','LINK-USD','ATOM-USD','LEO-USD',
           'FIL-USD','OKB-USD']

models = ['RandomForestRegressor','MLPRegressor','KNeighborsRegressor',
          'DecisionTreeRegressor','ExtraTreesRegressor']

timeframes = ['1 day ahead','1 week ahead','1 month ahead']

selected_ticker = st.sidebar.selectbox('SELECT YOUR TICKER : ', symbols)
selected_timeframe = st.sidebar.selectbox('SELECT YOUR TIMEFRAME : ', timeframes)
selected_model = st.sidebar.selectbox('SELECT YOUR MODEL : ', models)


#st.subheader('Selected Cryptocurrency Low & High')

if selected_ticker:

    data = yf.download(tickers=selected_ticker,
                       period='5y',
                       interval='1d',
                       ignore_tz=True,
                       prepost=False)

    # Feature Engineering

    data['shift_High_1'] = data['High'].shift().fillna(data['High'].median())
    data['shift_Low_1'] = data['Low'].shift().fillna(data['Low'].median())

    data['shift_High_2'] = data['High'].shift(2).fillna(data['High'].median())
    data['shift_Low_2'] = data['Low'].shift(2).fillna(data['Low'].median())

    # Feature Selection

    X = data[['shift_High_1', 'shift_Low_1', 'shift_High_2', 'shift_Low_2']]
    y = data[['High', 'Low']]

    X = X[14:]
    y = y[14:]

    # RandomForestRegressor
    if selected_model == 'RandomForestRegressor':

        # Build model
        model = RandomForestRegressor()

        # Fit model
        model.fit(X,y)

        if selected_timeframe == '1 day ahead':
            # Predictions
            st.subheader('1 Day - Predicted ' + selected_ticker + ' High & Low')
            one_day_predicted = pd.DataFrame(model.predict(X)[-1]).transpose()
            st.write(one_day_predicted)

        if selected_timeframe == '1 week ahead':
            st.subheader('1 week - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-7:])

        if selected_timeframe == '1 month ahead':
            st.subheader('1 Month - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-30:])

    # MLPRegressor
    if selected_model == 'MLPRegressor':

        # Build model
        model = MLPRegressor()

        # Fit model
        model.fit(X,y)

        if selected_timeframe == '1 day ahead':
            st.subheader('1 Day - Predicted ' + selected_ticker + ' High & Low')
            one_day_predicted = pd.DataFrame(model.predict(X)[-1]).transpose()
            st.write(one_day_predicted)

        if selected_timeframe == '1 week ahead':
            st.subheader('1 week - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-7:])

        if selected_timeframe == '1 month ahead':
            st.subheader('1 Month - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-30:])

    # KNeighborsRegressor
    if selected_model == 'KNeighborsRegressor':
        # Build model
        model = KNeighborsRegressor()

        # Fit model
        model.fit(X, y)

        # predictions
        if selected_timeframe == '1 day ahead':
            st.subheader('1 Day - Predicted ' + selected_ticker + ' High & Low')
            one_day_predicted = pd.DataFrame(model.predict(X)[-1]).transpose()
            st.write(one_day_predicted)

        if selected_timeframe == '1 week ahead':
            st.subheader('1 week - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-7:])

        if selected_timeframe == '1 month ahead':
            st.subheader('1 Month - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-30:])

    # DecisionTreeRegressor
    if selected_model == 'DecisionTreeRegressor':
        # Build model
        model = DecisionTreeRegressor()

        # Fit model
        model.fit(X,y)

        # predictions
        if selected_timeframe == '1 day ahead':
            st.subheader('1 Day - Predicted ' + selected_ticker + ' High & Low')
            one_day_predicted = pd.DataFrame(model.predict(X)[-1]).transpose()
            st.write(one_day_predicted)

        if selected_timeframe == '1 week ahead':
            st.subheader('1 week - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-7:])

        if selected_timeframe == '1 month ahead':
            st.subheader('1 Month - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-30:])

    # ExtraTreesRegressor
    if selected_model == 'ExtraTreesRegressor':
        # Build model
        model = ExtraTreesRegressor()

        # Fit model
        model.fit(X, y)

        # predictions
        if selected_timeframe == '1 day ahead':
            st.subheader('1 Day - Predicted ' + selected_ticker + ' High & Low')
            one_day_predicted = pd.DataFrame(model.predict(X)[-1]).transpose()
            st.write(one_day_predicted)

        if selected_timeframe == '1 week ahead':
            st.subheader('1 week - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-7:])

        if selected_timeframe == '1 month ahead':
            st.subheader('1 Month - Predicted ' + selected_ticker + ' High & Low')
            st.write(model.predict(X)[-30:])
