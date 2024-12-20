import numpy as np
import pandas as pd
import tensorflow as tf
import datetime
from sklearn.model_selection import train_test_split

tf.random.set_seed(5)
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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)

model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(13,)),
    tf.keras.layers.Dense(10, activation='relu', name='dense_1_model_1'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model2 = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(13,)),
    tf.keras.layers.Dense(10, activation='relu', name='dense_1_model_2'),
    tf.keras.layers.Dense(10, activation='relu', name='dense_2_model_2'),
    tf.keras.layers.Dense(10, activation='relu', name='dense_3_model_2'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#
model.fit(X_train, y_train, epochs=300, callbacks=[tensorboard_callback])
#
# model.save("model1.keras")

# loss2, accuracy2 = model2.evaluate(X_test, y_test)


import sys

model = tf.keras.models.load_model("model1.keras")

# example parameter: "13.2,2.5,2.8,18,100,2.7,3.1,0.4,1.5,4.3,1.0,3.0,1000"
wine_attributes = np.array([float(x) for x in sys.argv[1].split(',')]).reshape(1, -1)

prediction = model.predict(wine_attributes)

predicted_category_index = np.argmax(prediction, axis=1)[0]
predicted_category = predicted_category_index + 1
print(f"Predicted wine category: {predicted_category}")