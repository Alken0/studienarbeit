from keras.optimizer_v2.adam import Adam

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import LEARNING_RATE_DISCRIMINATOR, LEARNING_RATE_GENERATOR

generator_optimizer = Adam(LEARNING_RATE_DISCRIMINATOR)
discriminator_optimizer = Adam(LEARNING_RATE_GENERATOR)
