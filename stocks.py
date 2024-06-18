import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Function to fetch data for a stock
@st.cache
def get_stock_data(stock, start_date, end_date):
    data = yf.download(stock, start=start_date, end=end_date)
    return data

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    # Calculate 20-day Moving Average
    data['SMA_20'] = data['Close'].rolling(window=20).mean()
    
    # Calculate Relative Strength Index (RSI)
    delta = data['Close'].diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate Moving Average Convergence Divergence (MACD)
    short_window = 12
    long_window = 26
    signal_window = 9
    data['EMA_12'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['EMA_26'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['EMA_12'] - data['EMA_26']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    
    return data

# Function to train a simple linear regression model for index movement prediction
def train_linear_regression_model(index_data):
    X = np.arange(len(index_data)).reshape(-1, 1)
    y = index_data['Close'].values.reshape(-1, 1)
    
    model = LinearRegression()
    model.fit(X, y)
    
    # Predict for the next day (future prediction)
    next_day = len(index_data) + 1
    prediction = model.predict([[next_day]])[0][0]
    
    return model, prediction

# Function to display charts
def display_charts(data, indicators):
    # Plot historical prices and indicators
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data.index, data['Close'], label='Close Price')
    for indicator in indicators:
        ax.plot(data.index, data[indicator], label=indicator)
    ax.set_xlabel('Date')
    ax.set_ylabel('Price / Indicator')
    ax.legend()
    st.pyplot(fig)

# Sidebar - Title and user input
st.sidebar.title('Nifty 50 Stock Analysis and Prediction')

# Get list of Nifty 50 stocks
nifty50_stocks = [
    'ADANIPORTS.NS', 'ASIANPAINT.NS', 'AXISBANK.NS', 'BAJAJ-AUTO.NS', 'BAJFINANCE.NS',
    'BAJAJFINSV.NS', 'BHARTIARTL.NS', 'BPCL.NS', 'CIPLA.NS', 'COALINDIA.NS',
    'DIVISLAB.NS', 'DRREDDY.NS', 'EICHERMOT.NS', 'GRASIM.NS', 'HCLTECH.NS',
    'HDFC.NS', 'HDFCBANK.NS', 'HDFCLIFE.NS', 'HEROMOTOCO.NS', 'HINDALCO.NS',
    'HINDUNILVR.NS', 'ICICIBANK.NS', 'INDUSINDBK.NS', 'INFY.NS', 'IOC.NS',
    'ITC.NS', 'JSWSTEEL.NS', 'KOTAKBANK.NS', 'LT.NS', 'M&M.NS',
    'MARUTI.NS', 'NESTLEIND.NS', 'NTPC.NS', 'ONGC.NS', 'POWERGRID.NS',
    'RELIANCE.NS', 'SBILIFE.NS', 'SBIN.NS', 'SHREECEM.NS', 'SUNPHARMA.NS',
    'TATAMOTORS.NS', 'TATASTEEL.NS', 'TCS.NS', 'TECHM.NS', 'TITAN.NS',
    'ULTRACEMCO.NS', 'UPL.NS', 'WIPRO.NS'
]

# Date range selector (replace with actual date range widget)
start_date = st.sidebar.date_input('Start Date', value=pd.to_datetime('2020-01-01'))
end_date = st.sidebar.date_input('End Date', value=pd.to_datetime('2023-01-01'))

# Fetch data for each stock and calculate indicators
st.header('Nifty 50 Stocks Technical Analysis')
for stock in nifty50_stocks:
    stock_data = get_stock_data(stock, start_date, end_date)
    stock_data = calculate_technical_indicators(stock_data)
    
    # Display charts for each stock
    st.subheader(f'Technical Analysis for {stock}')
    display_charts(stock_data, ['SMA_20', 'RSI', 'MACD', 'Signal_Line'])  # Example with SMA_20, RSI, MACD, Signal_Line

# Predict Nifty 50 index movement using linear regression
index_data = yf.download('^NSEI', start=start_date, end=end_date)
model, prediction = train_linear_regression_model(index_data)

# Display prediction and recommendation
st.header('Nifty 50 Index Movement Prediction')
st.subheader('Next Day Prediction for Nifty 50 Index')
st.write(f'Predicted Close Price: {prediction:.2f}')

# Footer
st.text('Disclaimer: This is a demo and not financial advice.')
