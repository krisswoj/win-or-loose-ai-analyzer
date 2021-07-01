from biosignal_and_tetris_result_service import get_player_results

import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.metrics import classification_report,confusion_matrix


# Make numpy printouts easier to read.
np.set_printoptions(precision=3, suppress=True)
pd.set_option('display.max_columns', None)
print(tf.__version__)

results = get_player_results()

df = results['E03_R02_S01'][0]['p1']

X = df.drop(['Czas','Draw','Lose','Win'],axis=1).values
y = df[['Win', 'Lose', 'Draw']].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()

model.add(Dense(5,  activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(5, activation='relu'))
model.add(Dropout(0.2))

model.add(Dense(3, activation='softmax', name='output'))

model.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

model.fit(x=X_train,
          y=y_train,
          epochs=20,
          batch_size=512,
          validation_data=(X_test, y_test),
          )
hi
predictions = model.predict_classes(X_test)
print(classification_report(y_test,predictions))
confusion_matrix(y_test,predictions)

# results = model.evaluate(X_test, y_test)
#
# print('Final test set loss: {:4f}'.format(results[0]))
# print('Final test set accuracy: {:4f}'.format(results[1]))
