from keras.losses import BinaryCrossentropy
from tensorboard.plugins.hparams import api as hp 


MODEL_NAME="models.densegan"

CLASS_NAMES = ["circle", "rectangle", "triangle"]
NUM_CLASSES = len(CLASS_NAMES)

LATENT_DIM = 100
IMG_SIZE = 28
IMG_DIM = IMG_SIZE * IMG_SIZE
EMBEDDING_SIZE = "set by hyperparameters"
LEARNING_RATE_DISCRIMINATOR = "set by hyperparameters"
LEARNING_RATE_GENERATOR = "set by hyperparameters"
DROPOUT = "set by hyperparameters"
EPOCHS = 2
BATCH_SIZE = 64
SMOOTH = "set by hyperparameters"
LOSS_FUNCTION = BinaryCrossentropy(from_logits=True)

IMG_SEED = 123456789
IMG_DIR = "data/training"

LOG_PATH = "data/logs"
LOG_IMG_PER_LABEL = 3

SHAPE_RECTANGLE_NAME = "rectangle"
SHAPE_RECTANGLE_AMOUNT = 500
SHAPE_RECTANGLE_MIN_SIZE = 5
SHAPE_RECTANGLE_MAX_SIZE = 16

SHAPE_TRIANGLE_NAME = "triangle"
SHAPE_TRIANGLE_AMOUNT = 500
SHAPE_TRIANGLE_MIN_SIZE = 5
SHAPE_TRIANGLE_MAX_SIZE = 16

SHAPE_CIRCLE_NAME = "circle"
SHAPE_CIRCLE_AMOUNT = 500
SHAPE_CIRCLE_MIN_SIZE = 5
SHAPE_CIRCLE_MAX_SIZE = 10

