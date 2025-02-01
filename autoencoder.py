import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import layers, losses
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model


@keras.saving.register_keras_serializable()
class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim
        self.shape = shape
        self.encoder = tf.keras.Sequential([
            layers.Conv2D(18, 5, data_format="channels_last", activation='relu', padding='same', strides=2),
            # layers.Dropout(0.2),
            # layers.Conv2D(64, 3, data_format="channels_last", activation='relu'),
            # layers.Conv2D(128, 3, data_format="channels_last", activation='relu'),
            layers.Flatten(),
            # layers.Dropout(0.2),
            layers.Dense(128, activation='relu'),
            layers.Dense(64, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(16, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(latent_dim, activation='tanh'),
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(16, activation='relu'),
            layers.Dense(32, activation='relu'),
            # layers.Dropout(0.2),
            layers.Dense(128 * 128 * 3, activation='sigmoid'),
            layers.Reshape((128, 128, 3)),
            # layers.Dropout(0.2),
            # layers.Conv2DTranspose(64, 3, data_format="channels_last", activation='relu', padding="same"),
            layers.Conv2DTranspose(16, 3, data_format="channels_last", activation='relu', padding="same"),
            layers.BatchNormalization(),
            # layers.Conv2DTranspose(16, 3, data_format="channels_last", activation='relu', padding="same"),
            layers.Conv2DTranspose(3, 3, data_format="channels_last", activation='sigmoid', padding="same"),
            layers.Reshape(shape)
        ])

    def call(self, x):
        # x = tf.reshape(x,(28,28))
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def get_config(self):
        return {"latent_dim": self.latent_dim, "shape": self.shape}

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def create_and_learn_autoencoder(dataset, latent_dim):
    dataset = tf.concat(list(dataset), axis=0)
    # print(dataset)
    # print(x_test.shape)

    # x_train = tf.reshape(dataset, (len(dataset), 128, 128, 1))
    shape = (128, 128, 3)
    autoencoder = Autoencoder(latent_dim, shape)
    #
    autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

    autoencoder.fit(dataset, dataset,
                    epochs=1500,
                    shuffle=True,
                    # validation_data=(x_test, x_test),
                    batch_size=1000)

    return autoencoder

def generate_100_random_images_in_latent_space(autoencoder, latent_dim):
    encoded_imgs = np.random.normal(size=(100, latent_dim))
    decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()

    decoded_imgs = np.clip(decoded_imgs, 0, 1)

    fig, axs = plt.subplots(10, 10)

    for y in range(10):
        for x in range(10):
            axs[y, x].imshow(decoded_imgs[y * 10 + x].reshape(128, 128, 3))
            axs[y, x].axis('off')

    plt.show()

def reconstruct_image(autoencoder, original_image_path):
    img = tf.keras.preprocessing.image.load_img(original_image_path, target_size=(128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = tf.expand_dims(img_array, axis=0)

    encoded_img = autoencoder.encoder(img_array)

    decoded_img = autoencoder.decoder(encoded_img).numpy().squeeze()  # Remove batch dimension

    fig, ax = plt.subplots(1, 2, figsize=(8, 4))

    ax[0].imshow(img_array.numpy().squeeze())  # Original image
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(decoded_img)  # Reconstructed image
    ax[1].set_title("Reconstructed Image")
    ax[1].axis("off")

    plt.show()

if __name__ == '__main__':
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        "autoencoder-data-2",
        color_mode='rgb',
        batch_size=32,
        image_size=(128, 128),
        label_mode=None
    )

    dataset = dataset.map(lambda x: x / 255.0)
    latent_dim = 2
    autoencoder = create_and_learn_autoencoder(dataset, latent_dim)
    generate_100_random_images_in_latent_space(autoencoder, latent_dim)
    reconstruct_image(autoencoder, "autoencoder-data-2/Metro-NEw-York-3940178063.jpg")





