#!/usr/bin/env python3


"""
Train a generative adversarial network to generate pixel art.
"""


__author__ = "Hamish Morgan"


from typing import Iterable

# import cv2
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf

from tensorflow.keras import layers


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def crops_from_full_size(
    image: np.ndarray, shape: tuple[int, int]
) -> Iterable[np.ndarray]:
    """
    From a full sized image, yield image crops of the desired size.

    Args:
        image: Image to make crops from
        shape: Desired size of crops (rows, cols)

    Yields:
        Image crops of desired size
    """
    image_height, image_width = image.shape
    crop_height, crop_width = shape
    for y in range(start=0, stop=image_height - crop_height, step=crop_height):
        for x in range(start=0, stop=image_width - crop_width, step=crop_width):
            yield image[y : y + crop_height, x : x + crop_width]


def permute_flips(
    image: np.ndarray, flip_x: bool = True, flip_y: bool = True
) -> Iterable[np.ndarray]:
    """
    From an input image, yield that image flipped in x and/or y
    """
    images = [image]
    if flip_x:
        images.append(np.fliplr(image))
    if flip_y:
        for image in images:
            images.append(np.flipud(image))
    return images


def normalise(image: np.ndarray) -> np.ndarray:
    """
    Normalise an image with pixel values on [0, 255] to [-1, 1]
    """
    return (image - 127.5) / 127.5


class PixelArtDataset:
    """
    Data serving class for pixel art images
    """

    def __init__(
        self, dataset_path: str, crop_shape: tuple[int, int]
    ) -> "PixelArtDataset":
        """
        Construct the data generator.

        Args:
            dataset_path: Path to directory tree containing training data
            crop_shape: Desired size of crops (rows, cols)
        """
        self.dataset_path_ = dataset_path
        self.crop_shape_ = crop_shape

    def __call__(self) -> Iterable[np.ndarray]:
        """
        Allow the data generator to be called (by tensorflow) to yield training examples.

        Yields:
            Training examples
        """
        for image_path in self.find_training_images():
            # image = cv2.imread(image_path).astype(np.float32)
            with PIL.Image.open(image_path) as image:
                for image_crop in crops_from_full_size(np.array(image), shape=self.crop_shape_):
                    yield from map(normalise, permute_flips(image_crop))

    def find_training_images(self) -> Iterable[str]:
        """
        Search the dataset.

        A single datum should consist of a directory containing one or more raster files. The
        directory structure above that is arbitrary.

        Returns:
            List
        """
        for root, _, filenames in os.walk(self.dataset_path_):
            for filename in filenames:
                yield os.path.join(root, filename)


def make_generator_model() -> tf.keras.Sequential:
    """
    Create the generator model
    """
    model = tf.keras.Sequential()

    model.add(layers.Dense(16 * 16 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((16, 16, 256)))
    assert model.output_shape == (None, 16, 16, 256)  # Note: None is the batch size

    model.add(
        layers.Conv2DTranspose(
            128, (5, 5), strides=(2, 2), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 32, 32, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            64, (5, 5), strides=(4, 4), padding="same", use_bias=False
        )
    )
    assert model.output_shape == (None, 128, 128, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(
        layers.Conv2DTranspose(
            3, (5, 5), strides=(4, 4), padding="same", use_bias=False, activation="tanh"
        )
    )
    assert model.output_shape == (None, 512, 512, 3)

    return model


def make_discriminator_model() -> tf.keras.Sequential:
    """
    Create the discriminator model
    """
    model = tf.keras.Sequential()
    model.add(
        layers.Conv2D(
            32, (5, 5), strides=(2, 2), padding="same", input_shape=[512, 512, 1]
        )
    )
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding="same"))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model


def discriminator_loss(
    real_output: tf.Tensor, fake_output: tf.Tensor
):
    """
    Define a custom loss function for the discriminator.

    Args:
        loss_fn: Function to compute loss on an image
        real_output: The discriminator's output for a real image
        fake_output: The discriminator's output for a fake image
    """
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


@tf.function
def train_step(
    images,
    generator: tf.keras.Sequential,
    generator_optimizer: "Optimizer",
    discriminator: tf.keras.Sequential,
    discriminator_optimizer: "Optimizer",
    batch_size: int,
    noise_size: int,
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        generator: Generator model
        generator_optimizer: Optimizer for generator model
        discriminator: Discriminator model
        discriminator_optimizer: Optimizer for discriminator model
        batch_size: Number of training examples in a batch
        noise_size: Length of input noise vector
    """
    noise = tf.random.normal([batch_size, noise_size])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables
    )
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )


def generate_and_save_images(model, epoch, test_input):
    """
    Generate and save images
    """
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
        plt.axis("off")

    plt.savefig("image_at_epoch_{:04d}.png".format(epoch))
    # plt.show()

    del(fig)



def do_train(
        dataset,
        epochs,
        generator,
        generator_optimizer,
        discriminator,
        discriminator_optimizer,
        seed,
        checkpoint,
        checkpoint_prefix
):
    for epoch in range(epochs):
        start = time.time()

        for image_batch in dataset:
            train_step(image_batch, generator, generator_optimizer, discriminator, discriminator_optimizer)

        # Produce images for the GIF as you go
        generate_and_save_images(generator, epoch + 1, seed)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

    # Generate after the final epoch
    generate_and_save_images(generator, epochs, seed)


def train(
    dataset_path: str,
    train_crop_shape: tuple[int, int] = (512, 512),
    buffer_size: int = 1000,
    batch_size: int = 2,
) -> None:
    """
    Train the networks.

    Args:
        dataset_path: Path to directory tree containing training data
        train_crop_shape: Desired shape of training crops from full images
        buffer_size: Number of images to randomly sample from at a time
        batch_size: Number of training examples in a batch
    """
    train_images = (
        tf.data.Dataset.from_generator(
            PixelArtDataset(dataset_path="", crop_shape=train_crop_shape),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
    )

    generator = make_generator_model()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4)

    discriminator = make_discriminator_model()
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = "./training_checkpoints"
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    EPOCHS = 200
    noise_dim = 100
    num_examples_to_generate = 16

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, noise_dim])

    do_train(
        dataset=train_images,
        epochs=EPOCHS,
        generator=generator,
        generator_optimizer=generator_optimizer,
        discriminator=discriminator,
        discriminator_optimizer=discriminator_optimizer,
        seed=seed,
        checkpoint=checkpoint,
        checkpoint_prefix=checkpoint_prefix
    )


def main() -> int:
    """
    Main CLI routine.
    """
    train_data_dir = sys.argv[1]
    train(dataset_path=train_data_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
