import numpy as np
from keras.models import Model
import sys
import tensorflow as tf

# Adds higher directory to python modules path.
sys.path.append(".")
from constants import LATENT_DIM, NUM_CLASSES

def generate_random_noise(amount: int):
    return tf.random.normal([amount, LATENT_DIM])

def generate_specific_labels(amount: int, label: int):
    return np.full((amount, 1), label, dtype=int)

def generate_labels_evenly(amount: int):
    function = lambda i, j: i % NUM_CLASSES
    return np.fromfunction(function, (amount, 1), dtype=int)

def generate_random_labels(amount: int):
    return np.random.randint(0, NUM_CLASSES, amount).reshape(-1, 1)

def generate_fake_data(generator: Model, noise, labels):
    return generator([noise, labels], training=False)

def generate_fake_data_by_label(generator: Model, amount: int, label: int):
    noise = generate_random_noise(amount)
    labels = generate_specific_labels(amount, label)
    return [generator.predict_on_batch([noise, labels]), labels]
