import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split

tf.random.set_seed(42)
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)

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


# prove that training and data sets are always the same:
# from pandas.util import hash_pandas_object
#
# print(int(hashlib.sha256(pd.util.hash_pandas_object(X_train, index=True).values).hexdigest(), 16))
# print(int(hashlib.sha256(pd.util.hash_pandas_object(X_test, index=True).values).hexdigest(), 16))
# print(int(hashlib.sha256(pd.util.hash_pandas_object(y_train, index=True).values).hexdigest(), 16))
# print(int(hashlib.sha256(pd.util.hash_pandas_object(y_test, index=True).values).hexdigest(), 16))

def create_model_with_params(layers, units, learning_rate=0.001, batch_size=128):
    model = tf.keras.Sequential()
    model.add(keras.layers.InputLayer(input_shape=(13,)))
    for layer in range(int(layers)):
        model.add(keras.layers.Dense(int(units), activation='relu'))

    model.add(tf.keras.layers.Dense(3, activation='softmax'))

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                  loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X_train, y_train, validation_split=0.2, epochs=100,
              # callbacks=[tensorboard_callback],
              batch_size=int(batch_size))
    return model


# zjazd4
import sys

# example parameter: "5,20,0.0001,128;20,5,0.001,64;5,40,0.0001,128;5,40,0.0001,256;5,40,0.001,256;3,20,0.002,128;2,60,0.0001,128;2,30,0.001,128"
parameters = [x for x in sys.argv[1].split(';')]
best_accuracy = 0
best_model = ""
models_with_accuracies = {}
for parameter in parameters:
    parameter = np.array([float(x) for x in parameter.split(',')])
    model = create_model_with_params(parameter[0], parameter[1], parameter[2], parameter[3])
    model.save(str(parameter) + ".keras")
    loss, accuracy = model.evaluate(X_test, y_test)
    model_name = "model[layers:" + str(parameter[0]) + ",units:" + str(parameter[1]) + ",learning_rate:" + str(
        parameter[2]) + ",batch_size:" + str(parameter[3]) + "]"
    models_with_accuracies.update({model_name: accuracy})
    print(model_name + " accuracy: " + str(accuracy))
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model_name
print(" MODEL_NAME ", " ACCURACY ")

for key, value in models_with_accuracies.items():
    print(f"{key}: {value}")

print("Best model: " + best_model + " | accuracy: " + str(best_accuracy))