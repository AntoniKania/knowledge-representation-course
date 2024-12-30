import os
import sys
from PIL import Image
import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np

(ds_train, ds_test), ds_info = tfds.load(
    'mnist',
    split=['train', 'test'],
    shuffle_files=True,
    as_supervised=True,
    with_info=True,
)


def normalize_img(image, label):
  """Normalizes images: `uint8` -> `float32`."""
  return tf.cast(image, tf.float32) / 255., label

ds_train = ds_train.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.shuffle(ds_info.splits['train'].num_examples)
ds_train = ds_train.batch(128)
ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

ds_test = ds_test.map(
    normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(128)
ds_test = ds_test.cache()
ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(10)
])
model.summary()
# model.compile(
#     optimizer=tf.keras.optimizers.Adam(0.001),
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#     metrics=[tf.keras.metrics.SparseCategoricalAccuracy()],
# )
#
# model.fit(
#     ds_train,
#     epochs=6,
#     validation_data=ds_test,
# )
# model.evaluate(ds_train)

# zadanie1
# import keras
from keras import datasets, layers, models
import matplotlib.pyplot as plt


def create_and_train_model():
    global model
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu', padding="same"))
    model.add(layers.Conv2D(32, (3, 3), activation='relu', padding="same"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.summary()
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(ds_train, epochs=10,
                        validation_data=ds_test)

    model.evaluate(ds_train)
    return model

# zadanie 2

if __name__ == '__main__':
    model_filename = 'number_predictions_convolutional_2.keras'
    if (os.path.isfile(model_filename)):
        model = tf.keras.models.load_model(model_filename)
    else:
        model = create_and_train_model()
        model.save(model_filename)  

    image_filename = sys.argv[1]
    size = 28, 28
    image = tf.keras.utils.load_img(image_filename, target_size=(28, 28), color_mode='grayscale')
    image.thumbnail(size, Image.Resampling.BOX)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.reshape(input_arr, (1, 28, 28))
    predictions = model.predict(input_arr)
    print(predictions)