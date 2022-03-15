from typing import Any
import tensorflow as tf
import numpy as np
from keras.models import Model
from .loss import generator_loss, discriminator_loss
from keras.optimizer_v2.adam import Adam

class HyperParams():
    def __init__(
        self,
        batch_size: int,
        optimizer_gen: Adam,
        optimizer_dis: Adam,
        latent_dim: int,
        num_classes: int) -> None:
        
        self.batch_size = batch_size
        self.optimizer_gen = optimizer_gen
        self.optimizer_dis = optimizer_dis
        self.latent_dim = latent_dim
        self.num_classes = num_classes
    
    def __str__(self) -> str:
        variables = self.__dict__.copy()
        variables["optimizer_gen"] = self.optimizer_gen._hyper["learning_rate"]
        variables["optimizer_dis"] = self.optimizer_dis._hyper["learning_rate"]
        return str(variables)
    

def train_step(generator: Model, discriminator: Model, labeled_img, hp: HyperParams):
    noise = tf.random.normal([hp.batch_size, hp.latent_dim])
    labels = np.random.randint(0, hp.num_classes, hp.batch_size).reshape(-1, 1)

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

    hp.optimizer_gen.apply_gradients(
        zip(gradients_of_generator, generator.trainable_variables))
    hp.optimizer_dis.apply_gradients(
        zip(gradients_of_discriminator, discriminator.trainable_variables))
