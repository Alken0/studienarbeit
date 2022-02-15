import datetime
import io
import matplotlib.pyplot as plt
import tensorflow as tf
from gan_model import HP_DIS_DROPOUT, HP_DIS_LR, HP_GAN_LR
from tensorflow.keras import preprocessing, models, callbacks
import os
from tqdm import tqdm
import numpy as np
import storage
from configuration import CONFIG
import gan_model
from metric_logger import Logger


### VARIABLES ###
EPOCHS = CONFIG.training.get("epochs")
BATCH_SIZE = CONFIG.training.get("batchSize")

LABEL_AMOUNT = 3
LABEL_NAMES = ["circle", "rectangle", "triangle"]

LATENT_DIM = 100
IMG_SIZE = CONFIG.image_size
IMG_CHANNELS = 1

EPOCHS_PER_IMAGE_SAMPLE = 5
IMAGES_PER_LABEL = 5

LOAD_MODEL = False
MODEL_NAME = "second_training"
SAVE_MODEL = False
### VARIABLES ###

#  some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# load dataset
dataset: tf.data.Dataset = preprocessing.image_dataset_from_directory(
    directory="data/training",
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
# normalize images (range [0, 255] to [-1.0, 1.0])
dataset = dataset.map(lambda img, label: ((img - 127.5)/127.0, label))


def generate_random_noise(batch_size: int, latent_dim: int):
    return np.random.randn(latent_dim * batch_size).reshape(batch_size, latent_dim)


def generate_random_input(batch_size: int, latent_dim: int, label_amount: int):
    noise = generate_random_noise(batch_size, latent_dim)
    labels = np.random.randint(low=0, high=label_amount, size=batch_size)
    return [noise, labels]


def generate_fake_data(generator: models.Model, batch_size: int, latent_dim: int, label_amount: int):
    noise, labels = generate_random_input(batch_size, latent_dim, label_amount)
    return [generator.predict([noise, labels]), labels]


def generate_fake_data_by_label(generator: models.Model, batch_size: int, latent_dim: int, label: int):
    noise = generate_random_noise(batch_size, latent_dim)
    labels = np.full(batch_size, label)
    return [generator.predict([noise, labels]), labels]


def image_grid(data, labels, label_names):
    # Data should be in (BATCH_SIZE, H, W, C)
    assert data.ndim == 4

    # invert image
    data = 1 - data

    figure = plt.figure(figsize=(10, 10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))

    for i in range(data.shape[0]):
        plt.subplot(size, size, i + 1, title=label_names[labels[i]])
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)

        # if grayscale
        if data.shape[3] == 1:
            plt.imshow(data[i], cmap=plt.cm.binary)

        else:
            plt.imshow(data[i])

    return figure

# Stolen from tensorflow official guide: https://www.tensorflow.org/tensorboard/image_summaries


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def log_epoch_metrics(discriminator, generator, gan, dis_logger, gan_logger, epoch, latent_dim, label_amount, hparams):
    gan_logger.write_log(gan, epoch, hparams)
    dis_logger.write_log(discriminator, epoch, hparams)
    gan.reset_metrics()
    discriminator.reset_metrics()

    """
    if epoch % EPOCHS_PER_IMAGE_SAMPLE == 0:
        images = np.empty((5, 64, 64, 1))
        labels = np.empty((5,))
        for label in range(label_amount):
            curr_images, curr_labels = generate_fake_data_by_label(
                generator, IMAGES_PER_LABEL, latent_dim, label)
            print(f"image shape: {curr_images.shape}")
            print(f"labels shape: {curr_labels.shape}")
            images = np.append(images, curr_images, axis=None)
            lables = np.append(labels, curr_labels, axis=None)
        print(f"image final: {images}")
        print(f"labels final: {labels}")
        figure = image_grid(images, labels, LABEL_NAMES)
        image = plot_to_image(figure)
        gan_logger.write_image("Composed Visualization", image, epoch)
    """
    # log Composed Visualization for each label
    if epoch % EPOCHS_PER_IMAGE_SAMPLE == 0:
        for label in range(label_amount):
            images, labels = generate_fake_data_by_label(
                generator, IMAGES_PER_LABEL, latent_dim, label)
            figure = image_grid(images, labels, LABEL_NAMES)
            image = plot_to_image(figure)
            gan_logger.write_image(
                f"Composed_Visualization_{LABEL_NAMES[label]}", image, epoch)

    # log Composed Visualization with random labels
    if epoch % EPOCHS_PER_IMAGE_SAMPLE == 0:
        images, labels = generate_fake_data(
            generator, (IMAGES_PER_LABEL*3), latent_dim, label_amount)
        figure = image_grid(images, labels, LABEL_NAMES)
        image = plot_to_image(figure)
        gan_logger.write_image(
            f"Composed_Visualization_Random_Labels", image, epoch)



def train_model(dataset: tf.data.Dataset, discriminator: models.Model, generator: models.Model, gan: models.Model, epochs: int, latent_dim: int, label_amount: int, hparams):
    gan_logger = Logger("gan", gan, hparams)
    dis_logger = Logger("discriminator", discriminator, hparams)

    for epoch in tqdm(range(epochs)):
        for batch in dataset:
            train_step(discriminator, generator, gan, batch, latent_dim, label_amount)
        log_epoch_metrics(discriminator, generator, gan, dis_logger, gan_logger, epoch, latent_dim, label_amount, hparams)
       
    if SAVE_MODEL:
        storage.save_model(MODEL_NAME, discriminator, generator)


def train_step(discriminator, generator, gan, batch, latent_dim, label_amount):
    # cannot be set statically -> e.g. if dataset is not dividable by batch-size the last batch is smaller
    batch_size = batch[0].shape[0]

    # train discriminator with real data
    discriminator.train_on_batch(
        batch,
        tf.ones((batch_size, 1)),
        reset_metrics=False
    )

    # generate fake data
    fake_data = generate_fake_data(
        generator, batch_size, latent_dim, label_amount)

    # train discriminator with fake data
    discriminator.train_on_batch(
        fake_data,
        tf.zeros((batch_size, 1)),
        reset_metrics=False
    )

    # train generator through gan (discriminator is not trained!)
    gan.train_on_batch(
        generate_random_input(batch_size, latent_dim, label_amount),
        tf.ones((batch_size, 1)),
        reset_metrics=False
    )

def train_all_hparams():
    for dis_lr in HP_DIS_LR.domain.values:
        for gan_lr in HP_GAN_LR.domain.values:
            for dis_drop in HP_DIS_DROPOUT.domain.values:
                hparams = {
                    HP_DIS_LR: dis_lr,
                    HP_GAN_LR: gan_lr,
                    HP_DIS_DROPOUT: dis_drop,
                }

                discriminator = gan_model.define_discriminator(
                    IMG_SIZE, IMG_CHANNELS, LABEL_AMOUNT, hparams)
                generator = gan_model.define_generator(LATENT_DIM, LABEL_AMOUNT, hparams)

                if LOAD_MODEL:
                    discriminator, generator = storage.load_model(MODEL_NAME)

                gan = gan_model.define_gan(generator, discriminator, hparams)
                print(gan.summary())

                train_model(dataset, discriminator, generator, gan, EPOCHS, LATENT_DIM, LABEL_AMOUNT, hparams)



train_all_hparams()
