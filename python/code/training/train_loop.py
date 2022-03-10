from keras.models import Model
import sys
import time
import tensorflow as tf

from .train_step import train_step
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import EPOCHS
from logger import MetricLogger

def train(generator: Model, discriminator: Model, dataset_train: tf.data.Dataset, dataset_test: tf.data.Dataset):
    logger_real = MetricLogger("real-data", discriminator)
    logger_fake = MetricLogger("fake-data", discriminator)

    for epoch in range(EPOCHS):
        start = time.time()
        for image_batch in dataset_train:
            train_step(generator, discriminator, image_batch)
        
        # logging
        logger_real.log(discriminator, dataset_test, epoch)
        print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')
