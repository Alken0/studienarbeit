import tensorflow as tf
import numpy as np
from keras.models import Model
from .loss import generator_loss, discriminator_loss
from .optimizer import discriminator_optimizer, generator_optimizer

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import BATCH_SIZE, LATENT_DIM, NUM_CLASSES

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(generator: Model, discriminator: Model, labeled_img):
    noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
    labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE).reshape(-1, 1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator([noise, labels], training=True)

        real_output = discriminator(labeled_img, training=True)
        fake_output = discriminator([generated_images, labels], training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))
