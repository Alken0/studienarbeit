from keras.models import Model
import sys
import time
import tensorflow as tf
from keras.optimizer_v2.adam import Adam

from .train_step import HyperParams, train_step
# Adds higher directory to python modules path.
sys.path.append(".")

from logger import RealDataMetricLogger, FakeDataMetricLogger

def train(generator: Model, discriminator: Model, dataset_train: tf.data.Dataset, dataset_test: tf.data.Dataset):
    from constants import EPOCHS, LEARNING_RATE_GENERATOR, LEARNING_RATE_DISCRIMINATOR, NUM_CLASSES, BATCH_SIZE, LATENT_DIM

    logger_real = RealDataMetricLogger(discriminator, generator)
    logger_fake = FakeDataMetricLogger(discriminator, generator)

    hyperparams = HyperParams(
        num_classes=NUM_CLASSES,
        batch_size=BATCH_SIZE,
        latent_dim=LATENT_DIM,
        optimizer_dis=Adam(LEARNING_RATE_DISCRIMINATOR),
        optimizer_gen=Adam(LEARNING_RATE_GENERATOR)
    )

    compiled_trainstep = tf.function(train_step)

    for epoch in range(EPOCHS):
        start = time.time()
        for image_batch in dataset_train:
            compiled_trainstep(generator, discriminator, image_batch, hyperparams)
        
        # logging
        logger_real.log(dataset_test, epoch)
        logger_fake.log(dataset_test, epoch)
        print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')
