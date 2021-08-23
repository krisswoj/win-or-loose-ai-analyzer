import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential

from biosignal_and_tetris_result_service import get_player_results

# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)
print(tf.__version__)

results = get_player_results()

df = [results['E03_R02_S01'][1]['p1']]

model = Sequential()

model.add(LSTM(64, input_shape=(None, 5), return_sequences=True))
model.add(LSTM(64, return_sequences=True))
model.add(LSTM(32, return_sequences=True))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation='softmax', name='output'))

for data in df:
    X = data.drop(['Czas', 'Draw', 'Lose', 'Win'], axis=1).values
    y = data[['Win', 'Lose', 'Draw']].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    X_train = X_train.reshape(-1, 1, 5)
    X_test = X_test.reshape(-1, 1, 5)
    y_train = y_train.reshape(-1, 1, 3)
    y_test = y_test.reshape(-1, 1, 3)

    model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')
    history = model.fit(X_train, y_train, batch_size=512, epochs=100, validation_data=(X_test, y_test))

    results = model.evaluate(X_test, y_test)

    print('Final test set loss: {:4f}'.format(results[0]))
    print('Final test set accuracy: {:4f}'.format(results[1]))

    losses = pd.DataFrame(history.history)

    y_predicted = model.predict(X_test)
    confusion = confusion_matrix(y_test.reshape(-1, 3).argmax(axis=1), y_predicted.reshape(-1, 3).argmax(axis=1))
    print('Confusion Matrix\n')
    print(confusion)
