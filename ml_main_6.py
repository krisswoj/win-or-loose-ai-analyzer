import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM, TimeDistributed
from tensorflow.keras.models import Sequential
from keras.utils import to_categorical

from biosignal_and_tetris_result_service import get_player_results

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)
print(tf.__version__)


# def get_concatenated_data():
#     data_p1 = []
#     data_p2 = []
#     for value in get_player_results().values():
#         for ddd in value:
#             data_p1.append(ddd['p1'])
#             data_p2.append(ddd['p2'])
#     return {'data_p1': pd.concat(data_p1),
#             'data_p2': pd.concat(data_p2)}


def get_concatenated_data():
    data_p1 = []
    for value in get_player_results().values():
        for data in value:
            cols = ['Win', 'Lose', 'Draw']

            if not set(cols).issubset(data['p1'].columns):
                continue

            X = data['p1'].dropna().drop(['Czas', 'Draw', 'Lose', 'Win'], axis=1, errors='ignore').values
            y = data['p1'].dropna()[['Win', 'Lose', 'Draw']].values

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

            scaler = MinMaxScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

            X_train = X_train.reshape(-1, 1, 5)
            X_test = X_test.reshape(-1, 1, 5)
            y_train = y_train.reshape(-1, 1, 3)
            y_test = y_test.reshape(-1, 1, 3)

            yield X_train, y_train
    # return data_p1


# dataset = tf.data.Dataset.from_generator(generator=_generator,
#                                          output_types=(tf.float32, tf.float32),
#                                          output_shapes=shapes)

# dataset = tf.data.Dataset.from_generator(generator=get_concatenated_data(), output_types=(tf.float32, tf.float32))


model = Sequential()
model.add(LSTM(256, input_shape=(None, 5), return_sequences=True))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(TimeDistributed(Dense(3, activation='softmax', name='output')))
model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# X = concatenated_data['data_p1'].drop(['Czas', 'Draw', 'Lose', 'Win'], axis=1).values
# y = concatenated_data['data_p1'][['Win', 'Lose', 'Draw']].values
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
#
# scaler = MinMaxScaler()
# X_train = scaler.fit_transform(X_train)
# X_test = scaler.transform(X_test)
#
# X_train = X_train.reshape(-1, 1, 5)
# X_test = X_test.reshape(-1, 1, 5)
# y_train = y_train.reshape(-1, 1, 3)
# y_test = y_test.reshape(-1, 1, 3)

# history = model.fit(X_train, y_train, batch_size=512, epochs=100, validation_data=(X_test, y_test))
#
# results = model.evaluate(X_test, y_test)
# print('Final test set loss: {:4f}'.format(results[0]))
# print('Final test set accuracy: {:4f}'.format(results[1]))
#
# losses = pd.DataFrame(history.history)
#
# y_predicted = model.predict(X_test)
# confusion = confusion_matrix(y_test.reshape(-1, 3).argmax(axis=1), y_predicted.reshape(-1, 3).argmax(axis=1))
# print('Confusion Matrix\n')
# print(confusion)

model.fit(get_concatenated_data(), steps_per_epoch=30, epochs=20, verbose=1)
