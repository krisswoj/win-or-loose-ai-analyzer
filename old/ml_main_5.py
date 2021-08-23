# -*- coding: utf-8 -*-
"""RNN_training_biosignals.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1bFAmqCEyCffA615b1b53X3LzdlLty8_4
"""

import re
import tqdm
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from os import listdir
from os.path import isfile, join
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, TimeDistributed
from tensorflow.keras.utils import plot_model

# from google.colab import drive
# drive.mount('/content/drive')

DATA_PATH = '/Users/krzysztofwojdak/Documents/win-or-loose-ai-analyzer/rnn_preprocess_data_csv_normalizated/test/*.csv'

"""## Tests"""

filepaths = DATA_PATH

dataset = tf.data.Dataset.list_files(filepaths)

dataset = dataset.map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1), num_parallel_calls=None)

for line in dataset.take(1):
    print(line)

"""## Load data"""

WINDOW_SIZE = 12
BATCH_SIZE = 16


def windowed_dataset(ds: tf.data.Dataset, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE, shuffle_buffer=128):
    '''Funkcje wywolujemy raz dla kazdego pliku'''
    # podzial na okna - z 'odcieciem' konca pliku
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    ds = ds.map(lambda w: (w[:-1][:-1][1:-1], w[-1:][1:][-1]))
    return ds  # ds.batch(batch_size)


def decode_line(line):
    return tf.strings.to_number(
        tf.strings.split(
            line, sep=',', maxsplit=-1, name=None), out_type=tf.dtypes.float32, name=None)


def csv_reader_dataset(filepaths, repeat=None, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    # zaczytuje nazwy sciezek
    dataset = tf.data.Dataset.list_files(filepaths)

    # otwieram pliki i pomijam 1 wiersz

    dataset = dataset.map(
        lambda filepath: tf.data.TextLineDataset(filepath).skip(1).map(decode_line))  # , num_parallel_calls=None)
    # dataset = dataset.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))
    # podzial na okna
    dataset = dataset.flat_map(windowed_dataset)
    # mieszanie
    # dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset
    # return dataset.prefetch(1)


ds = csv_reader_dataset(DATA_PATH)

for batch in ds.take(1):
    print(batch)

"""# Praca

0) **Podział na gry treningowe (80%), walidacyjne (10%) i testowe (10%)**

1) **Normalizacja danych - w oparciu o treningowe - stosujac tez do wal i terningowych**

2) **Zmienic encoding y na skale -1..1**

3) Zmienic model - wartwa wyjsciowa - Dense(1, activation="tanh")

4) Learning Rate

5) Ewaluacja - z zaokraglaniem
"""


def create_model():
    model = Sequential([
        LSTM(256, input_shape=(WINDOW_SIZE, 10), return_sequences=True),
        LSTM(128, return_sequences=True),
        LSTM(64, return_sequences=True),
        Dense(32, activation='relu'),
        Dense(8, activation='relu'),
        Dense(1, activation='tanh', name='output')])

    model.compile(loss='mse', optimizer='adam')
    return model


model = create_model()
# plot_model(model, show_shapes=True)

model.predict(ds)

"""### Training"""

history = model.fit(ds, steps_per_epoch=30, epochs=20, verbose=1)