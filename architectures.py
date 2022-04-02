"""
Various model architectures for the generator and discriminator
"""


from typing import Tuple

import tensorflow as tf

from tensorflow.keras import layers


def generator_model_dcgan_paper() -> tf.keras.Sequential:
    """
    The generator used in the DCGAN paper.

    This is a very large model, it doesn't fit on a GTX980 :(
    """
    latent_dim = 100
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(4 * 4 * 1024, activation="relu"),
            layers.Reshape((4, 4, 1024)),
            layers.BatchNormalization(),
            # layers.leakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                512, kernel_size=5, strides=2, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=2, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                3, kernel_size=5, strides=2, padding="same", activation="tanh"
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(alpha=0.2),
        ]
    )
    model.summary()
    return model


def generator_model_dcgan_paper_lite() -> tf.keras.Sequential:
    """
    Based on the generator used in the DCGAN paper.

    A smaller model, to fit on a GTX980.
    """
    latent_dim = 100
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential(
        [
            # Latent input
            layers.Input(shape=(latent_dim,)),
            # Dense layer, shaped as (8, 8, 512) conv layer
            layers.Dense(8 * 8 * 512, kernel_initializer=init),
            layers.leakyReLU(alpha=0.2),
            layers.Reshape((8, 8, 512)),
            layers.BatchNormalization(),
            # 1st upscale by fractionally strided conv, (16, 16, 256)
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", kernel_initializer=init,
            ),
            layers.leakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            # 2nd upscale by fractionally strided conv, (32, 32, 128)
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=2, padding="same", kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.leakyReLU(alpha=0.2),
            # 3rd upscale by fractionally strided conv, (64, 64, 64)
            layers.Conv2DTranspose(
                3, kernel_size=5, strides=2, padding="same", kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.leakyReLU(alpha=0.2),
            # Output, (64, 64, 3)
            layers.Conv2D(3, kernel_size=7, padding="same", activation="tanh", kernel_initializer=init)
        ]
    )
    model.summary()
    return model


def generator_model_28x28() -> tf.keras.Sequential:
    """
    The generator used in the tensorflow tutorial, but with 3 channels.
    """
    model = tf.keras.Sequential()
    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))
    assert model.output_shape == (None, 7, 7, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(1, 1), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 7, 7, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 14, 14, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            3, (5, 5), strides=(2, 2), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 28, 28, 3)

    model.summary()

    return model


def discriminator_model_generic(
    input_shape: Tuple[int],
) -> tf.keras.Sequential:
    """
    A generic discriminator model
    """
    model = tf.keras.Sequential(
        [
            layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape
            ),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same"),
            layers.LeakyReLU(0.2),
            layers.Dropout(0.3),
            layers.Flatten(),
            layers.Dense(1),
        ]
    )

    model.summary()

    return model


def discriminator_model_generic_deeper(
    input_shape: Tuple[int],
) -> tf.keras.Sequential:
    """
    A generic discriminator model
    """
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential(
        [
            # Input size conv layer (e.g. 64, 64, 64)
            layers.Conv2D(
                64, (5, 5), strides=(2, 2), padding="same", input_shape=input_shape, kernel_initializer=init
            ),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            # Half input size conv layer (e.g. 32, 32, 128)
            layers.Conv2D(128, (5, 5), strides=(2, 2), padding="same", kernel_initializer=init),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            # Quarter input size conv layer (e.g. 16, 16, 256)
            layers.Conv2D(256, (5, 5), strides=(2, 2), padding="same", kernel_initializer=init),
            layers.LeakyReLU(alpha=0.2),
            layers.Dropout(0.3),
            # Output neuron
            layers.Flatten(),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    model.summary()

    return model
