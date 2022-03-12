from .base import MetricLogger
import tensorflow as tf
from keras.models import Model

class RealDataMetricLogger(MetricLogger):
    def __init__(self, discriminator: Model, generator: Model):
        name = "real-data"
        super().__init__(name, discriminator, generator)

    def log(self, dataset: tf.data.Dataset, epoch):
        self.discriminator.reset_metrics()

        for image_batch in dataset:
            self.discriminator.evaluate(
                x=image_batch, 
                y=tf.ones(len(list(image_batch[0]))), 
                verbose=0
            )
        
        self._write_log(self.discriminator, epoch)