import numpy as np
from keras.models import Model
import tensorflow as tf

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import BATCH_SIZE, SMOOTH, LATENT_DIM, NUM_CLASSES
from dataset import Dataset

real = np.ones(shape=(BATCH_SIZE, 1))
fake = np.zeros(shape=(BATCH_SIZE, 1))

def train_step(generator: Model, discriminator: Model, gan: Model, dataset: Dataset, i: int):
    discriminator.trainable = False

    # Real Samples
    real_img_batch = dataset.x_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE]
    real_label_batch = dataset.y_train[i*BATCH_SIZE:(i+1)*BATCH_SIZE].reshape(-1, 1)

    amount_elements_in_batch = len(real_img_batch)

    real = np.ones(shape=(amount_elements_in_batch, 1))
    fake = np.zeros(shape=(amount_elements_in_batch, 1))

    d_loss_real = discriminator.train_on_batch(x=[real_img_batch, real_label_batch], y=real * (1 - SMOOTH))

    # Fake Samples
    latent = np.random.normal(loc=0, scale=1, size=(amount_elements_in_batch, LATENT_DIM))
    fake_label_batch = np.random.randint(0, NUM_CLASSES, amount_elements_in_batch).reshape(-1, 1)
    fake_img_batch = generator.predict_on_batch([latent, fake_label_batch])
    d_loss_fake = discriminator.train_on_batch(x=[fake_img_batch, fake_label_batch], y=fake)

    # Discriminator Loss
    d_loss_batch = 0.5 * (d_loss_real[0] + d_loss_fake[0])

    # Train Generator
    discriminator.trainable = False
    latent = np.random.normal(loc=0, scale=1, size=(amount_elements_in_batch, LATENT_DIM))
    random_labels = np.random.randint(0, NUM_CLASSES, amount_elements_in_batch).reshape(-1, 1)

    gan_loss_batch = gan.train_on_batch(x=[latent, random_labels], y=real)

    return (d_loss_batch, gan_loss_batch)
