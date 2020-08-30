# -*- coding: utf-8 -*-
"""
Created on Thu Apr 30 21:21:24 2020

@author: Samael Olascoaga
"""

# Predecir el precio del bitcoin

import json
import requests
import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Activation, Dense, Dropout, LSTM
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats
from sklearn.metrics import mean_absolute_error

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


activo = ['BTC', 'ETH', 'BCH', 'XRP']
numero = 0 # Posición para cada activo
limite = 2000 # Límite de datos que se importaran
target_col = 'open'


direccion = 'https://min-api.cryptocompare.com/data/histoday'
url = direccion + '?fsym=' + activo[numero] + '&tsym=USD&limit=' + str(limite) # Cambiar numero por activo deseado
f = requests.get(url)
ipdata = f.json()
df = pd.DataFrame(ipdata['Data'])

predecir = 8

#df = pd.read_csv('btcd.csv')
#df = df.tail(2269)
ultimos = df.tail(predecir)
df = df[:-predecir]
# Preprocesamos los datos, seleccionando únicamente la columna de fecha y cierre de los primeros 2000 datos
df = df.set_index("time")#[[target_col]]
df.index = pd.to_datetime(df.index, unit='s')


df = df.drop('volumefrom', 1)
df = df.drop('volumeto', 1)
df = df.drop('low', 1)
df = df.drop('high', 1)
#df = df.drop('close', 1)

def train_test_split(df, test_size=0.1):
    split_row = len(df) - int(test_size * len(df))
    train_data = df.iloc[:split_row]
    test_data = df.iloc[split_row:]
    return train_data, test_data

train, test = train_test_split(df, test_size=0.01)

def line_plot(line1, line2, label1=None, label2=None, title='', lw=2):
    fig, ax = plt.subplots(1, figsize=(16, 9))
    ax.plot(line1, label=label1, linewidth=lw)
    ax.plot(line2, label=label2, linewidth=lw)
    ax.set_ylabel('price [USD]', fontsize=14)
    ax.set_title(title, fontsize=18)
    ax.legend(loc='best', fontsize=18);
    
line_plot(train[target_col], test[target_col], 'training', 'test', title='BTC')

def normalise_zero_base(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    return df / df.iloc[0] - 1

def desnormalice(df):
    """ Normalise dataframe column-wise to reflect changes with respect to first entry. """
    return df * df.iloc[0] + 1

def extract_window_data(df, window_len=10, zero_base=True):
    """ Convert dataframe to overlapping sequences/windows of len `window_data`.
    
        :param window_len: Size of window
        :param zero_base: If True, the data in each window is normalised to reflect changes
            with respect to the first entry in the window (which is then always 0)
    """
    window_data = []
    for idx in range(len(df) - window_len):
        tmp = df[idx: (idx + window_len)].copy()
        if zero_base:
            tmp = normalise_zero_base(tmp)
        window_data.append(tmp.values)
    return np.array(window_data)

def prepare_data(df, target_col, window_len=10, zero_base=True, test_size=0.2):
    """ Prepare data for LSTM. """
    # train test split
    train_data, test_data = train_test_split(df, test_size=test_size)
    
    # extract window data
    X_train = extract_window_data(train_data, window_len, zero_base)
    X_test = extract_window_data(test_data, window_len, zero_base)
    
    # extract targets
    y_train = train_data[target_col][window_len:].values
    y_test = test_data[target_col][window_len:].values
    if zero_base:
        y_train = y_train / train_data[target_col][:-window_len].values - 1
        y_test = y_test / test_data[target_col][:-window_len].values - 1
    return train_data, test_data, X_train, X_test, y_train, y_test

def build_lstm_model(input_data, output_size, neurons=20, activ_func='linear',
                     dropout=0.25, loss='mae', optimizer='adam'):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(input_data.shape[1], input_data.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)
    return model

np.random.seed(42)
# data params
window_len = 8
test_size = 0.01
zero_base = True

# model params
lstm_neurons = 560
epochs = 100
batch_size = 128 # 128
loss = 'mae' # mae default
dropout = 0.3
optimizer = 'adam' # default: adam

train, test, X_train, X_test, y_train, y_test = prepare_data(
    df, target_col, window_len=window_len, zero_base=zero_base, test_size=test_size)

model = build_lstm_model(
    X_train, output_size=1, neurons=lstm_neurons, dropout=dropout, loss=loss,
    optimizer=optimizer)
history = model.fit(
    X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1, shuffle=True)

targets = test[target_col][window_len:]
preds = model.predict(X_test).squeeze()
print(mean_absolute_error(preds, y_test))

preds = test[target_col].values[:-window_len] * (preds + 1)
preds = pd.Series(index=targets.index, data=preds)

n_points = 30
line_plot(targets[-n_points:], preds[-n_points:], 'actual', 'prediction', lw=3)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 9))


actual_returns = targets.pct_change()[1:]
predicted_returns = preds.pct_change()[1:]

# actual correlation
corr = np.corrcoef(actual_returns, predicted_returns)[0][1]
ax1.scatter(actual_returns, predicted_returns, color='k', marker='o', alpha=0.5, s=100)
ax1.set_title('r = {:.2f}'.format(corr), fontsize=18)


# shifted correlation
shifted_actual = actual_returns[:-1]
shifted_predicted = predicted_returns.shift(-1).dropna()
corr = np.corrcoef(shifted_actual, shifted_predicted)[0][1]
ax2.scatter(shifted_actual, shifted_predicted, color='k', marker='o', alpha=0.5, s=100)
ax2.set_title('r = {:.2f}'.format(corr), fontsize=18);

predicciones = []
ventana = df.tail(8)

aperturas = 9
for i in range(1, aperturas + 1):
    if i == 1:
        ventana_normalizada = normalise_zero_base(ventana).to_numpy()
        ventana_normalizada = ventana_normalizada.reshape(1, 8, 1)
        valor = model.predict(ventana_normalizada)
        ventana = list(np.array(ventana).reshape(len(ventana)))
        temporal = float(valor + 1) * ventana[0]
        predicciones.append(temporal + 200)
        ventana.append(temporal)
        ventana.pop(0)
        ventana = pd.DataFrame(ventana)
    else:
        ventana_normalizada = normalise_zero_base(ventana).to_numpy()
        ventana_normalizada = ventana_normalizada.reshape(1, 8, 1)
        valor = model.predict(ventana_normalizada)
        ventana = list(np.array(ventana).reshape(len(ventana)))
        temporal = float(valor + 1) * ventana[0]
        predicciones.append(temporal)
        ventana.append(temporal)
        ventana.pop(0)
        ventana = pd.DataFrame(ventana)

plt.plot(predicciones, '-ok')