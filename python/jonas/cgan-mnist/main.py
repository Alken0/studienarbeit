import os
import tensorflow as tf

# https://github.com/mafda/generative_adversarial_networks_101/blob/master/src/mnist/03_CGAN_MNIST.ipynb

#  some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from model import make_generator_model, make_discriminator_model, make_gan_model
from training import train
from dataset import make_dataset

generator = make_generator_model()
discriminator = make_discriminator_model()
gan = make_gan_model(discriminator=discriminator, generator=generator)
mnist_datset = make_dataset()


train(generator, discriminator, gan, mnist_datset)
