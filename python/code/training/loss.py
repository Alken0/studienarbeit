import tensorflow as tf
import sys

# Adds higher directory to python modules path.
sys.path.append(".")
# This method returns a helper function to compute cross entropy loss
from constants import LOSS_FUNCTION as cross_entropy
from constants import SMOOTH

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output) * (1 - SMOOTH), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)
