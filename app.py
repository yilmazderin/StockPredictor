#Import libraries
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import pandas_datareader as data
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
import streamlit as st

#Choose dates
start = '2013-01-01'
end = '2022-12-31'

#Title
st.title('Stock Price Predictor')

#Receives stock ticker
user_input = st.text_input('Enter stock ticker', 'AAPL')

#Gets stock data from Yahoo Finance
df = yf.download(user_input, start, end)

#Shows the data
st.subheader('Data from 2013 - 2022')
st.write(df.describe())

#Create closing price graph
st.subheader('Closing Price vs Time chart')
fig = plt.figure(figsize = (12,6))
plt.plot(df.Close)
st.pyplot(fig)

#Create MA100 graph
st.subheader('Closing Price vs Time chart with 100 days Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

#Create MA100 and MA200 graph
st.subheader('Closing Price vs Time chart with 100 and 200 days Moving Average')
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize = (12,6))
plt.plot(ma100, 'r')
plt.plot(ma200, 'g')
plt.plot(df.Close, 'b')
st.pyplot(fig)

#Seperate data into training and testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

#Create scaler to scale data between 0 and 1
scaler = MinMaxScaler(feature_range=(0,1))

data_training_array = scaler.fit_transform(data_training)

#Create x and y training arrays
x_train = []
y_train = []

#Populate arrays with corresponding training data
for i in range(100, data_training_array.shape[0]):
    x_train.append(data_training_array[i-100: i])
    y_train.append(data_training_array[i, 0]) 

#Convert to numPy arrays
x_train, y_train = np.array(x_train), np.array(y_train)

#Use model from Jupyter Notebooks
model = load_model('keras_model.h5')

#Get the last 100 days of training data for testing data
past_100_days = data_training.tail(100)

#Combine last 100 days of training data with testing data
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)

#Scale the data
input_data = scaler.fit_transform(final_df)

#Create x and y testing arrays
x_test = []
y_test = []

#Populate arrays wtih corresponding testing data
for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100: i])
    y_test.append(input_data[i, 0])

#Convert to numPy arrays
x_test, y_test = np.array(x_test), np.array(y_test)

#Get predicted values
y_predicted = model.predict(x_test)

#Scale data back to original
scale_factor = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

#Create predictions graph
st.subheader('Predictions vs Original Price')
fig2 = plt.figure(figsize = (12,6))
plt.plot(y_test, 'b', label = "Original Price")
plt.plot(y_predicted, 'r', label = "Predicted Price")
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)

#test