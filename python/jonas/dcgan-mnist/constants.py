import tensorflow as tf
import os

BUFFER_SIZE = 60000
BATCH_SIZE = 256
EPOCHS = 50
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
SEED = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

LOGGING_DIR = './logs'
LOG_IMG_DIR = os.path.join(LOGGING_DIR, "img")
CHECKPOINT_DIR = os.path.join(LOGGING_DIR, 'checkpoints')
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt")


