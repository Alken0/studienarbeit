import tensorflow as tf
import os

# https://www.tensorflow.org/tutorials/generative/dcgan


#  some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from training import train
from dataset import make_dataset
from model import make_discriminator_model, make_generator_model

dataset = make_dataset()
generator = make_generator_model()
discriminator = make_discriminator_model()

train(generator, discriminator, dataset)
