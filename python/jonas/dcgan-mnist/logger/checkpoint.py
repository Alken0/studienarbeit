import os
import tensorflow as tf

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from training.optimizer import generator_optimizer, discriminator_optimizer
from constants import CHECKPOINT_DIR

try:
    os.makedirs(CHECKPOINT_DIR)
except FileExistsError:
    pass

def make_checkpoint(generator, discriminator) -> tf.train.Checkpoint:
    return tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                               discriminator_optimizer=discriminator_optimizer,
                               generator=generator,
                               discriminator=discriminator)
