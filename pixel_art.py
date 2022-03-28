#!/usr/bin/env python3


"""
Train a generative adversarial network to generate pixel art.
"""


__author__ = "Hamish Morgan"


from typing import Iterable, Tuple

import argparse
import datetime
import os
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import tensorflow as tf

from architectures import discriminator_model_generic, generator_model_dcgan_paper


# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def crops_from_full_size(
    image: np.ndarray, shape: Tuple[int, int, int]
) -> Iterable[np.ndarray]:
    """
    From a full sized image, yield image crops of the desired size.

    Args:
        image: Image to make crops from
        shape: Desired size of crops (rows, cols)

    Yields:
        Image crops of desired size
    """
    (image_height, image_width, _) = image.shape
    crop_height, crop_width, _ = shape
    for y in (
        *range(0, image_height - crop_height, crop_height),
        image_height - crop_height,
    ):
        for x in (
            *range(0, image_width - crop_width, crop_width),
            image_width - crop_width,
        ):
            yield image[y : y + crop_height, x : x + crop_width, :]


def permute_flips(
    image: np.ndarray, flip_x: bool = True, flip_y: bool = True
) -> list[np.ndarray]:
    """
    From an input image, yield that image flipped in x and/or y
    """
    images = [image]
    if flip_x:
        images.append(np.fliplr(image))
    if flip_y:
        images += [*map(np.flipud, images)]
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
        self, dataset_path: str, crop_shape: Tuple[int, int]
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
            try:
                with PIL.Image.open(image_path) as image:
                    image_np = np.array(image)
            except PIL.UnidentifiedImageError:
                print(
                    f"Cannot open file: {image_path}, it will not be used for training"
                )
            if len(image_np.shape) != 3:
                print(f"Unusual image found: {image_path}, has shape {image_np.shape}")
                continue
            if image_np.shape[2] > 3:
                image_np = image_np[:, :, :3]
            for image_crop in crops_from_full_size(image_np, shape=self.crop_shape_):
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


def visualise_dataset(
    dataset_path: str = "./training-data",
    train_crop_shape: Tuple[int, int, int] = (28, 28, 3),
    as_tensorflow_dataset: bool = True,
) -> None:
    """
    View the dataset

    Args
        dataset_path: Directory containing training data
        train_crop_shape: Expected shape of crops for training
        as_tensorflow_dataset: Create tensorflow dataset and read from that if True,
                               show raw generator output otherwise.
    """
    if as_tensorflow_dataset:
        train_images = tf.data.Dataset.from_generator(
            PixelArtDataset(dataset_path=dataset_path, crop_shape=train_crop_shape),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        for image in iter(train_images):
            plt.imshow(image)
    else:
        for image in PixelArtDataset(
            dataset_path=dataset_path, crop_shape=train_crop_shape
        )():
            plt.imshow(image)
            plt.show()


def discriminator_loss(real_output: tf.Tensor, fake_output: tf.Tensor):
    """
    Custom loss function for the discriminator.

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
    """
    Custom loss function for the generator

    The generator wins if the discriminator thinks its output is real (i.e. all ones).
    """
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def generate_and_save_images(model, epoch, test_input, model_dir: str):
    """
    Generate and save images
    """
    predictions = model(test_input, training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow(np.array((predictions[i, :, :, :] * 127.5 + 127.5)).astype(int))
        plt.axis("off")

    os.makedirs(progress_dir := os.path.join(model_dir, "progress"), exist_ok=True)
    plt.savefig(os.path.join(progress_dir, f"image_at_epoch_{epoch:04d}.png"), dpi=250)

    del fig


@tf.function
def train_step(
    images,
    generator: tf.keras.Sequential,
    generator_optimizer: "Optimizer",
    generator_loss_metric: tf.keras.metrics,
    discriminator: tf.keras.Sequential,
    discriminator_optimizer: "Optimizer",
    discriminator_loss_metric: tf.keras.metrics,
    batch_size: int,
    noise_size: int,
) -> None:
    """
    Perform one step of training

    Args:
        images: Training batch
        generator: Generator model
        generator_optimizer: Optimizer for generator model
        generator_loss: Metric for logging generator loss
        discriminator: Discriminator model
        discriminator_optimizer: Optimizer for discriminator model
        discriminator_loss: Metric for logging discriminator loss
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

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables
    )

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables)
    )
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables)
    )

    generator_loss_metric(gen_loss)
    discriminator_loss_metric(disc_loss)


def train(
    model_dir: str,
    dataset_path: str,
    epochs: int = 200,
    train_crop_shape: Tuple[int, int, int] = (64, 64, 3),
    buffer_size: int = 1000,
    batch_size: int = 128,
    epochs_per_turn: int = 3,
) -> None:
    """
    Train the networks.

    Args:
        model_dir: Working dir for this experiment
        dataset_path: Path to directory tree containing training data
        epochs: How many full passes through the dataset to make
        train_crop_shape: Desired shape of training crops from full images
        buffer_size: Number of images to randomly sample from at a time
        batch_size: Number of training examples in a batch
        epochs_per_turn: How long to train one model before switching to the other
    """
    train_images = (
        tf.data.Dataset.from_generator(
            PixelArtDataset(dataset_path=dataset_path, crop_shape=train_crop_shape),
            output_signature=tf.TensorSpec(shape=train_crop_shape, dtype=tf.float32),
        )
        .shuffle(buffer_size)
        .batch(batch_size)
        .cache()
    )

    latent_dim = 100
    num_examples_to_generate = 16

    generator = generator_model_dcgan_paper()
    generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0.5)

    discriminator = discriminator_model_generic(input_shape=train_crop_shape)
    discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

    checkpoint_dir = os.path.join(model_dir, "training_checkpoints")
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator,
        discriminator=discriminator,
    )

    # Define our metrics
    generator_loss_metric = tf.keras.metrics.Mean("generator_loss", dtype=tf.float32)
    discriminator_loss_metric = tf.keras.metrics.Mean(
        "discriminator_loss", dtype=tf.float32
    )

    # Set up logs
    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    generator_log_dir = os.path.join(
        model_dir, "logs", "gradient_tape", current_time, "generator"
    )
    discriminator_log_dir = os.path.join(
        model_dir, "logs", "gradient_tape", current_time, "discriminator"
    )
    generator_summary_writer = tf.summary.create_file_writer(generator_log_dir)
    discriminator_summary_writer = tf.summary.create_file_writer(discriminator_log_dir)

    # You will reuse this seed overtime (so it's easier)
    # to visualize progress in the animated GIF)
    seed = tf.random.normal([num_examples_to_generate, latent_dim])

    # Start by freezing the generator, give the discriminator a head start
    generator.trainable = False
    discriminator.trainable = True
    
    for epoch in range(epochs):
        start = time.time()

        # Alternate who can train periodically
        if epoch % epochs_per_turn == 0:
            generator.trainable = not generator.trainable
            discriminator.trainable = not discriminator.trainable
            print(f"Switching who trains: {generator.trainable=}, {discriminator.trainable=}")

        for image_batch in train_images:
            train_step(
                images=image_batch,
                generator=generator,
                generator_optimizer=generator_optimizer,
                generator_loss_metric=generator_loss_metric,
                discriminator=discriminator,
                discriminator_optimizer=discriminator_optimizer,
                discriminator_loss_metric=discriminator_loss_metric,
                batch_size=batch_size,
                noise_size=latent_dim,
            )

        with generator_summary_writer.as_default(), discriminator_summary_writer.as_default():
            tf.summary.scalar(
                "generator_loss", generator_loss_metric.result(), step=epoch
            )
            tf.summary.scalar(
                "discriminator_loss", discriminator_loss_metric.result(), step=epoch
            )

        # Produce images for the GIF as you go
        generate_and_save_images(generator, epoch + 1, seed, model_dir)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print("Time for epoch {} is {} sec".format(epoch + 1, time.time() - start))

        template = """
        Epoch: {epoch}
        Generator loss: {generator_loss}
        Discriminator Loss: {discriminator_loss}
        """
        print(
            template.format(
                epoch=epoch + 1,
                generator_loss=generator_loss_metric.result(),
                discriminator_loss=discriminator_loss_metric.result(),
            )
        )

        # Reset metrics every epoch
        generator_loss_metric.reset_states()
        discriminator_loss_metric.reset_states()


def get_args() -> argparse.Namespace:
    """
    Define and parse command line arguments

    Returns:
        Argument values as argparse namespace
    """
    parser = argparse.ArgumentParser(
        "Generate pixel art", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(help="Modes of operation", dest="subcommand")

    train_parser = subparsers.add_parser(
        "train", help="Train the generator and discriminator models"
    )
    train_parser.add_argument(
        "--dataset",
        "-d",
        type=str,
        default="./training-data",
        help="Path to dataset directory, containing training images",
    )
    train_parser.add_argument(
        "--model-name",
        "-m",
        type=str,
        default=datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
        help="Name of this model/experiment for logging. Generated automatically if not given.",
    )

    return parser.parse_args()


def main(args: argparse.Namespace) -> int:
    """
    Main CLI routine

    Args:
        args: Command line arguments

    Returns:
        Exit status
    """
    if args.subcommand == "train":
        train(
            model_dir=os.path.join("models", args.model_name), dataset_path=args.dataset
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(get_args()))
