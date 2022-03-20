from .base import MetricLogger
import tensorflow as tf
import sys
from keras.models import Model
from .util.log_image import image_grid, plot_to_image
from .util.generate import generate_fake_data, generate_labels_evenly, generate_random_labels, generate_random_noise
from tensorflow import summary
import matplotlib.pyplot as plt

# Adds higher directory to python modules path.
sys.path.append(".")
from constants import BATCH_SIZE, NUM_CLASSES, CLASS_NAMES, LOG_IMG_PER_LABEL_TO_TB

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

        noise = generate_random_noise(amount)
        labels = generate_random_labels(amount)
        fake_data = generate_fake_data(self.generator, noise, labels)

        self.discriminator.evaluate(
            x=[fake_data, labels], 
            y=tf.zeros(amount),
            batch_size=BATCH_SIZE,
            verbose=0
        )
        
        self._write_image(epoch)
        self._write_log(self.discriminator, epoch)

    def _write_image(self, epoch):
        img_amount = LOG_IMG_PER_LABEL_TO_TB * NUM_CLASSES

        noise = generate_random_noise(img_amount)
        labels = generate_labels_evenly(img_amount)
        images = generate_fake_data(self.generator, noise, labels)

        figure = image_grid(images, labels, CLASS_NAMES)
        image = plot_to_image(figure)

        with self.summary_writer.as_default():
            summary.image(f"Generated_Images", image, max_outputs=len(image), step=epoch)
        
        plt.close('all') # prevent overflow of figure instances
