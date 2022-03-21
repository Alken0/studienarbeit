from keras import Sequential
from keras.layers import *
from keras.models import Model
from keras.initializers.initializers_v2 import GlorotNormal
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_discriminator_model() -> Model:
    from constants import NUM_CLASSES, IMG_DIM, IMG_SIZE
    initializer = GlorotNormal()

    input_label = Input(shape=(1,), dtype='int32')
    in_label = Embedding(NUM_CLASSES, IMG_DIM, embeddings_initializer=initializer)(input_label)
    in_label = Flatten()(in_label)

    input_img = Input(shape=(IMG_SIZE, IMG_SIZE))
    in_img = Flatten()(input_img)

    input_discriminator = multiply([in_img, in_label])

    validity = discriminator()(input_discriminator)

    model = Model([input_img, input_label], validity, name="Discriminator_Input")

    return model


def discriminator() -> Model:
    from constants import IMG_DIM, DROPOUT, EMBEDDING_SIZE
    initializer = GlorotNormal()

    model = Sequential([
        # input
        Dense(128, input_shape=(IMG_DIM,), kernel_initializer=initializer, name="input"),
        LeakyReLU(alpha=0.2),

        # hidden layer 1
        Dense(256, name="hidden1", kernel_initializer=initializer),
        LeakyReLU(alpha=0.2),

        # hidden layer 2
        Dense(512, name="hidden2", kernel_initializer=initializer),
        LeakyReLU(alpha=0.2),

        Flatten(),
        Dropout(DROPOUT),

        # output
        Dense(1, name="output", kernel_initializer=initializer),
    ], name="Discriminator_Layers")

    return model