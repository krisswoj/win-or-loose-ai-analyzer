

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
from tensorflow.keras.layers import Dense, Dropout, LSTM,TimeDistributed, Lambda
from tensorflow.keras.utils import plot_model

DATA_PATH = '/content/drive/MyDrive/rnn_preprocess_data_csv_global_standardization/train/*.csv'
VAL_DATA_PATH = '/content/drive/MyDrive/rnn_preprocess_data_csv_global_standardization/validation/*.csv'

# class RoundAccuracy(tf.keras.metrics.Metric):
#   def __init__(self, name='round_accuracy', **kwargs):
#     super(RoundAccuracy, self).__init__(name=name, **kwargs)
#     self.accuracy_metric = tf.keras.metrics.CategoricalAccuracy()
#     self.count = self.add_weight("count", initializer="zeros")

#   def update_state(self, y_true, y_pred, sample_weight=None):
#     y_pred = tf.keras.backend.round(y_pred)
#     self.accuracy_metric.update_state(y_true, y_pred, sample_weight=sample_weight)

#   def result(self):
#     return self.accuracy_metric.result()

class RoundAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name='round_accuracy', **kwargs):
        super(RoundAccuracy, self).__init__(name=name, **kwargs)
        self.state = self.add_weight(name='tp', initializer='zeros')
        self.count = self.add_weight("count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.keras.backend.round(y_pred)
        # print('y_pred.dtype', y_pred.dtype) # float32
        # print('y_true.dtype', y_true.dtype) # float32
        y_accuracy_same = tf.math.equal(y_true, y_pred)
        # size =  tf.size(y_true)
        y_accuracy_same = tf.cast(y_accuracy_same, dtype=tf.float32)
        #sumowac
        self.state.assign_add(tf.math.reduce_sum(y_accuracy_same))
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        #dzielic
        return self.state / self.count

# tf.math.divide(tf.cast(tf.math.equal(y_true, y_pred), tf.int32)

"""## Load data"""

WINDOW_SIZE = 2*256 # 2 seconds
BATCH_SIZE = 16
AUTOTUNE = tf.data.experimental.AUTOTUNE

def windowed_dataset(ds:tf.data.Dataset, window_size=WINDOW_SIZE, batch_size=BATCH_SIZE):
    '''Funkcje wywolujemy raz dla kazdego pliku'''
    #podzial na okna - z 'odcieciem' konca pliku
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True, )
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))

    ds = ds.map(lambda w: (tf.transpose(tf.transpose(w[:-1])[1:-1]), tf.transpose(tf.transpose(w[-1])[-1:])), num_parallel_calls=AUTOTUNE)
    #for consequences
    #ds = ds.map(lambda w: (tf.transpose(tf.transpose(w[1:])[1:-1]), tf.transpose(tf.transpose(w[0])[-1:])), num_parallel_calls=AUTOTUNE)
    return ds.batch(batch_size)


def decode_line(line):
    return tf.strings.to_number(
        tf.strings.split(
            line, sep=',', maxsplit=-1, name=None), out_type=tf.dtypes.float32, name=None)

def csv_reader_dataset(filepaths, repeat=None, n_readers=5,
                       n_read_threads=None, shuffle_buffer_size=10000,
                       n_parse_threads=5, batch_size=32):
    #zaczytuje nazwy sciezek
    dataset = tf.data.Dataset.list_files(filepaths)
    #otwieram pliki i pomijam 1 wiersz
    dataset = dataset.map(lambda filepath: tf.data.TextLineDataset(filepath).skip(1).map(decode_line), num_parallel_calls=AUTOTUNE)
    # dataset = dataset.map(lambda x: tf.io.decode_csv(x, record_defaults=font_column_types))
    #podzial na okna
    dataset = dataset.flat_map(windowed_dataset)
    #mieszanie
    dataset = dataset.shuffle(shuffle_buffer_size)
    return dataset.repeat().prefetch(AUTOTUNE)

ds = csv_reader_dataset(DATA_PATH)

val_ds = csv_reader_dataset(VAL_DATA_PATH)

# for batch in ds.take(1):
#   print(batch)

"""# Praca

-1) **Normalizacja danych - w oparciu o treningowe - stosujac tez do wal i terningowych**

 Wczytywanie zapisanego modelu

 Treningi

4) Learning Rate

### Regularized model
"""

def create_cnn_lstm_model_regularized():
    model = Sequential([
        # tf.keras.layers.Conv1D(128, (7,), input_shape=(WINDOW_SIZE,10), strides=(2,)),
        # tf.keras.layers.Conv1D(256, (7,)),
        # tf.keras.layers.LayerNormalization(axis=-1 , center=True , scale=True),
        LSTM(128, input_shape=(None, 10), return_sequences=True),
        # LSTM(256, return_sequences=True),
        # LSTM(256, return_sequences=True),
        # LSTM(64, return_sequences=True),
        LSTM(32, return_sequences=False),
        tf.keras.layers.LayerNormalization(axis=-1 , center=True , scale=True),
        # Dense(32, activation='relu'),
        Dense(8, activation='relu'),
        #
        Dense(1, activation='tanh', name='output')])

    model.compile(loss='mse', optimizer='adam', metrics = [RoundAccuracy()])
    return model


model_r = create_cnn_lstm_model_regularized()
plot_model(model_r, show_shapes = True)

def create_small():
    model = Sequential([
        tf.keras.layers.Input(shape=(WINDOW_SIZE,10)),
        #
        tf.keras.layers.Flatten(),
        #
        Dense(1, activation='tanh', name='output')])

    model.compile(loss='mse', optimizer='adam', metrics = [RoundAccuracy()])
    return model

"""### Training"""

results_dir = '/content/drive/MyDrive/rnn_preprocess_data_csv_global_standardization/real_training_kriss_23'
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(results_dir, save_weights_only=False, save_best_only=True),                                #saves network state
    tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=4),
    #tf.keras.callbacks.TensorBoard(log_dir=tensorboard_path, histogram_freq=1),
    tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True),
]

history_r = model_r.fit(ds, epochs=1500, verbose=1, callbacks = callbacks, validation_data = val_ds, steps_per_epoch = 5000, validation_steps=400, use_multiprocessing=True, workers=AUTOTUNE)

model_r.save(results_dir)

"""# Round predictions"""

pd.DataFrame(history_r.history).plot()

"""### Model wersja 1"""

# def create_model():
#   model = Sequential([
#                       LSTM(256, input_shape=(WINDOW_SIZE,10), return_sequences=True),
#                       LSTM(128, return_sequences=True),
#                       LSTM(64, return_sequences=False),
#                       Dense(32, activation='relu'),
#                       Dense(8, activation='relu'),
#                       Dense(1, activation='tanh', name='output')])

#   model.compile(loss='mse', optimizer='adam')
#   return model



# model_0 = create_model()
# plot_model(model_0, show_shapes = True)
# model_0.summary()

# def create_cnn_lstm_model():
#   model = Sequential([
#                       tf.keras.layers.Conv1D(128, (7,), input_shape=(WINDOW_SIZE,10), strides=(2,)),
#                       tf.keras.layers.Conv1D(256, (7,)),
#                       #LSTM(256, return_sequences=True),
#                       # LSTM(64, return_sequences=True),
#                       LSTM(32, return_sequences=False),
#                       # Dense(32, activation='relu'),
#                       Dense(8, activation='relu'),
#                       Dense(1, activation='tanh', name='output')])

#   model.compile(loss='mse', optimizer='adam')
#   return model


# model = create_cnn_lstm_model()
# plot_model(model, show_shapes = True)

model.summary()

#model.predict(ds)

output = Lambda(lambda x: tf.keras.backend.round(x)) (model.output)
m1 = tf.keras.Model(model.input, output)
m1.compile(loss='mae', metrics=[tf.keras.metrics.Accuracy()], optimizer='adam')
m1.evaluate(ds, steps=24000)

# output = Lambda(lambda x: tf.ones(shape=(BATCH_SIZE,))) (model.output)
# m2 = tf.keras.Model(model.input, output)
# m2.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()], optimizer='adam')
# m2.evaluate(ds)

# output = Lambda(lambda x: tf.zeros(shape=(BATCH_SIZE,))) (model.output)
# m3 = tf.keras.Model(model.input, output)
# m3.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()], optimizer='adam')
# m3.evaluate(ds)

# output = Lambda(lambda x: -tf.ones(shape=(BATCH_SIZE,))) (model.output)
# m4 = tf.keras.Model(model.input, output)
# m4.compile(loss='categorical_crossentropy', metrics=[tf.keras.metrics.Accuracy()], optimizer='adam')
# m4.evaluate(ds)

"""Save model

### Model wersja 2
"""