import tensorflow as tf
from tensorflow.keras import preprocessing, models
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
LATENT_DIM = 100
IMG_SIZE = CONFIG.image_size
IMG_CHANNELS = 1

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


def random_generator_input(batch_size: int, latent_dim: int, label_amount: int):
    noise = np.random.randn(
        latent_dim * batch_size).reshape(batch_size, latent_dim)
    labels = np.random.randint(0, label_amount, batch_size)
    return [noise, labels]


def generate_fake_data(generator: models.Model, batch_size: int, latent_dim: int, label_amount: int):
    noise, labels = random_generator_input(
        batch_size, latent_dim, label_amount)
    return [generator.predict([noise, labels]), labels]


def train(dataset: tf.data.Dataset, discriminator: models.Model, generator: models.Model, gan: models.Model, epochs: int, latent_dim: int, label_amount: int):
    gan_logger = Logger("gan")
    dis_logger = Logger("dis")

    for epoch in tqdm(range(epochs)):
        for batch in dataset:
            train_step(batch, latent_dim, label_amount)

        gan_logger.write_log(gan, epoch)
        dis_logger.write_log(discriminator, epoch)
        gan.reset_metrics()
        discriminator.reset_metrics()

        # save some samples
        if epoch % 100 == 0:
            fake_data = generate_fake_data(
                generator, 10, latent_dim, label_amount)
            images, labels = fake_data
            gan_logger.write_images(labels, images, epoch)
            for index, image in enumerate(images):

                file = preprocessing.image.array_to_img(image)
                file.save(
                    f"data/results/epoch_{epoch}_label_{labels[index]}_index_{index}.png")

    if SAVE_MODEL:
        storage.save_model(MODEL_NAME, discriminator, generator)


def train_step(batch, latent_dim, label_amount):
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

    # train generator through gan
    gan.train_on_batch(
        random_generator_input(batch_size, latent_dim, label_amount),
        tf.ones((batch_size, 1)),
        reset_metrics=False
    )


discriminator = gan_model.define_discriminator(
    IMG_SIZE, IMG_CHANNELS, LABEL_AMOUNT)
generator = gan_model.define_generator(LATENT_DIM, LABEL_AMOUNT)

if LOAD_MODEL:
    discriminator, generator = storage.load_model(MODEL_NAME)

gan = gan_model.define_gan(generator, discriminator)
print(gan.summary())

train(dataset, discriminator, generator, gan, EPOCHS, LATENT_DIM, LABEL_AMOUNT)
