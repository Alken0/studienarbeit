from .base import MetricLogger
import tensorflow as tf
import sys
from keras.models import Model
import numpy as np


# Adds higher directory to python modules path.
sys.path.append(".")
from constants import BATCH_SIZE, LATENT_DIM, NUM_CLASSES

class FakeDataMetricLogger(MetricLogger):
    def __init__(self, discriminator: Model, generator: Model):
        name = "fake-data"
        super().__init__(name, discriminator, generator)

    def log(self, dataset: tf.data.Dataset, epoch):
        self.discriminator.reset_metrics()

        # the following is returning 14 instead of the correct number...
        # amount = len(list(dataset)) # not recommended!
        # amount = dataset.cardinality()
        # amount = dataset.cardinality().numpy()
        amount = 892
        
        noise = tf.random.normal([amount, LATENT_DIM])
        labels = np.random.randint(0, NUM_CLASSES, amount).reshape(-1, 1)

        fake_data = self.generator([noise, labels], training=False)

        self.discriminator.evaluate(
            x=[fake_data, labels], 
            y=tf.zeros(amount),
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        self._write_log(self.discriminator, epoch)
