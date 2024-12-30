import tensorflow as tf
import keras
import keras_tuner as kt
import pandas as pd
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)

dataframe = pd.read_csv('./wine/wine.data', index_col=None)
dataframe.columns = [
    "Category",
    "Alcohol",
    "Malic acid",
    "Ash",
    "Alcalinity of ash",
    "Magnesium",
    "Total phenols",
    "Flavanoids",
    "Nonflavanoid phenols",
    "Proanthocyanins",
    "Color intensity",
    "Hue",
    "OD280/OD315 of diluted wines",
    "Proline"
]

hot_one_encoded_dataframe = pd.get_dummies(dataframe, columns=['Category'])
X = hot_one_encoded_dataframe.iloc[:, :13]
y = hot_one_encoded_dataframe.iloc[:, 13:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

def model_builder(hp):
  model = keras.Sequential()
  model.add(keras.layers.InputLayer(input_shape=(13,)))

  hp_units = hp.Int('units', min_value=10, max_value=100, step=10)
  # number of layers
  for i in range(1, hp.Int("num_layers", 2, 10)):
      model.add(keras.layers.Dense(units=hp_units, activation='relu'))

  model.add(tf.keras.layers.Dense(3, activation='softmax'))

  hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

  model.compile(optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
                loss="binary_crossentropy",
                metrics=['accuracy'])

  return model

stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

tuner = kt.Hyperband(model_builder,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='keras_tuner',
                     project_name='zjazd6')
tuner.search(X_train, y_train, epochs=50, validation_data=(X_train, y_train), callbacks=[stop_early])

best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]