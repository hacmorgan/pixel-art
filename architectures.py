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
            # layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                512, kernel_size=5, strides=2, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=2, padding="same", activation="relu"
            ),
            layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
            layers.Conv2DTranspose(
                3, kernel_size=5, strides=2, padding="same", activation="tanh"
            ),
            layers.BatchNormalization(),
            # layers.LeakyReLU(alpha=0.2),
        ]
    )
    model.summary()
    return model


def generator_model_dcgan_paper_lite() -> tf.keras.Sequential:
    """
    A smaller version of the DGCAN paper model, to fit on a GTX980.

    This version has some issues, such as using plain relu instead of leaky relu. Use
    generator_model_dcgan_paper_lite_relu instead. This is only maintained for
    future generation if desired.
    """
    latent_dim = 100
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(latent_dim,)),
            layers.Dense(8 * 8 * 512, use_bias=False, activation="relu"),
            layers.Reshape((8, 8, 512)),
            layers.BatchNormalization(),
            # layers.leakyReLU(0.2),
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", activation="relu", use_bias=False,
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(0.2),
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=2, padding="same", activation="relu", use_bias=False
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(0.2),
            layers.Conv2DTranspose(
                3, kernel_size=5, strides=2, padding="same", activation="tanh", use_bias=False
            ),
            layers.BatchNormalization(),
            # layers.leakyReLU(0.2),
        ]
    )
    model.summary()
    return model


def generator_model_dcgan_paper_lite_relu() -> tf.keras.Sequential:
    """
    A smaller version of the DGCAN paper model, to fit on a GTX980.

    This version is improved over the previous version, by using LeakyReLU activations
    and adding another conv layer.
    """
    latent_dim = 100
    init = tf.keras.initializers.RandomNormal(stddev=0.02)
    model = tf.keras.Sequential(
        [
            # Latent input
            layers.Input(shape=(latent_dim,)),
            # Dense layer, shaped as (8, 8, 512) conv layer
            layers.Dense(8 * 8 * 512, kernel_initializer=init),
            layers.LeakyReLU(alpha=0.2),
            layers.Reshape((8, 8, 512)),
            layers.BatchNormalization(),
            # 1st upscale by fractionally strided conv, (16, 16, 256)
            layers.Conv2DTranspose(
                256, kernel_size=5, strides=2, padding="same", kernel_initializer=init,
            ),
            layers.LeakyReLU(alpha=0.2),
            layers.BatchNormalization(),
            # 2nd upscale by fractionally strided conv, (32, 32, 128)
            layers.Conv2DTranspose(
                128, kernel_size=5, strides=2, padding="same", kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # 3rd upscale by fractionally strided conv, (64, 64, 64)
            layers.Conv2DTranspose(
                64, kernel_size=5, strides=2, padding="same", kernel_initializer=init,
            ),
            layers.BatchNormalization(),
            layers.LeakyReLU(alpha=0.2),
            # Output, (64, 64, 3)
            layers.Conv2D(3, kernel_size=7, padding="same", activation="tanh", kernel_initializer=init)
        ]
    )
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
            layers.Dense(1),
        ]
    )

    model.summary()

    return model
