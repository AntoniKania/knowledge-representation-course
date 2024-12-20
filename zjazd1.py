import tensorflow as tf
import numpy as np
import os.path
import sys
from PIL import Image
from tensorflow.python.ops.math_ops import negative

mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

def create_and_train_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=10)
    return model

if __name__ == '__main__':
    model_filename = 'first_model.keras'
    if(os.path.isfile(model_filename)):
        model = tf.keras.models.load_model(model_filename)
    else:
        model = create_and_train_model()
        model.save(model_filename)
    model.evaluate(x_test, y_test)

    # example param: "6.png"
    image_filename = sys.argv[1]
    size = 28, 28
    image = tf.keras.utils.load_img(image_filename, target_size=(28, 28), color_mode='grayscale')
    image.thumbnail(size, Image.Resampling.BOX)
    input_arr = tf.keras.utils.img_to_array(image)
    input_arr = np.reshape(input_arr, (1, 28, 28))
    print(input_arr[0])
    predictions = model.predict(input_arr)
    print(predictions)
