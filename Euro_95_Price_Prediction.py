# -*- coding: utf-8 -*-
"""
Created on Sat Jun  4 20:01:05 2022

@author: BojanR
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint, EarlyStopping
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import mean_squared_error
import math
from numba import jit, cuda
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if tf.test.gpu_device_name(): 
    print('Default GPU Device:{}'.format(tf.test.gpu_device_name()))
else:
   print("Please install GPU version of TF")
   
# Data preparation
df = pd.read_csv('Desktop/weekly_fuel_prices_all_data_from_2005_to_20210823.csv')
df.drop(columns=['PRODUCT_NAME', 'VAT', 'EXCISE', 'NET', 'CHANGE'], inplace = True)
df = df[df['PRODUCT_ID']== 1]
df.rename(columns={"SURVEY_DATE": "DATE"}, inplace = True)
df.drop(columns=['PRODUCT_ID'], inplace = True)

print(df)


plt.plot(df['DATE'], df['PRICE'])
plt.title('Euro super 95 price, Italy', fontsize=12)
plt.ylabel('€/1,000 liters', fontsize=12)
plt.show()
df.set_index("DATE", inplace = True)

print(df.describe())
df.info()
train_data = df['2005-01-03' : '2019-05-06']
test_data = df['2019-05-13':]

print('Observations: %d' % (len(df)))
print('Train Dataset: %d', train_data.shape)
print('Test Dataset: %d', test_data.shape)

ax = train_data.plot(figsize=(10, 5))
test_data.plot(ax=ax, color='r')
plt.legend(['train', 'test']);

scaler = MinMaxScaler(feature_range = (0,1))
train_data_scaled = scaler.fit_transform(train_data)
print(train_data_scaled); print(train_data_scaled.shape)

X_train = []
y_train = []
for i in range(60, len(train_data_scaled)):
    X_train.append(train_data_scaled[i-60:i,0])
    y_train.append(train_data_scaled[i,0])

X_train, y_train = np.array(X_train), np.array(y_train)
print(X_train); print(); print(y_train)

X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
print(X_train.shape); print(); print(X_train)

model = tf.keras.Sequential()

# adding 1st LSTM layer and some dropout regularization
model.add(tf.keras.layers.LSTM(units=50, input_shape=(X_train.shape[1], 1), return_sequences=True, activation = 'relu'))
model.add(tf.keras.layers.Dropout(0.2))

# adding 2nd LSTM layer and some dropout regularization
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(units=50, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

# adding 3rd LSTM layer and some dropout regularization
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(units=50, return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))

# adding 4th LSTM layer and some dropout regularization
model.add(tf.compat.v1.keras.layers.CuDNNLSTM(units=50))
model.add(tf.keras.layers.Dropout(0.2))

# adding output layer
model.add(tf.keras.layers.Dense(units=1))

#compiling RNN
model.compile(loss='mean_squared_error', optimizer='adam')

early_stopping = EarlyStopping(monitor='loss', patience=10)

# fitting RNN on training set
model.fit(X_train, y_train, epochs= 100, batch_size=32, 
          verbose=2, callbacks=[early_stopping])

model.save('Desktop/ORI_Projekat/Model')
# %% 

from sklearn.metrics import r2_score
model = tf.keras.models.load_model('Desktop/ORI_Projekat/Model')
 
dataset_total = pd.concat((train_data, test_data), axis=0)
print(dataset_total)
dataset_total = pd.concat((train_data, test_data), axis=0)  

inputs = dataset_total[len(dataset_total) - len(test_data)- 60:].values
inputs = inputs.reshape(-1,1)
inputs = scaler.transform(inputs) # transforming input data

X_test = []
y_test = []

for i in range (60, 177):
    X_test.append(inputs[i-60:i, 0])
    y_test.append(train_data_scaled[i,0])
      
X_test, y_test = np.array(X_test), np.array(y_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
pred_price = model.predict(X_test)
pred_price = scaler.inverse_transform(pred_price)
print(pred_price)

a = pd.DataFrame(pred_price)
a.rename(columns = {0: 'Predicted'}, inplace=True); 
a.index = test_data.index
compare = pd.concat([test_data, a],1)
compare

plt.figure(figsize= (15,5))
plt.plot(compare['PRICE'], color = 'red', label ="Actual Euro super 95 price")
plt.plot(compare.Predicted, color='blue', label = 'Predicted Price')
plt.title("Euro super 95 Price Prediction")
plt.xlabel('Time')
plt.ylabel('Euro super 95 price')
plt.legend(loc='best')
plt.show()

test_score = math.sqrt(mean_squared_error(compare['PRICE'], compare.Predicted))
print('Test Score: %.2f RMSE' % (test_score))
print("Price")
print(compare['PRICE'])
print("Predict")
print(compare.Predicted)
# Accurancy
print(r2_score(compare.Predicted, compare['PRICE']))






