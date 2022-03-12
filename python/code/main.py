import tensorflow as tf
import os
import shutil
import importlib

# https://www.tensorflow.org/tutorials/generative/dcgan

#  some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

import hyperparameters as hp

for params in hp.iterator():
    hp.update(params)

    from constants import MODEL_NAME
    from training import train
    from dataset import make_dataset

    models = importlib.import_module(MODEL_NAME)

    (dataset_train, dataset_test) = make_dataset()
    generator = models.make_generator_model()
    discriminator = models.make_discriminator_model()

    train(generator, discriminator, dataset_train, dataset_test)
