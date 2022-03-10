from keras.optimizer_v2.adam import Adam

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import LEARNING_RATE

generator_optimizer = Adam(LEARNING_RATE)
discriminator_optimizer = Adam(LEARNING_RATE)
