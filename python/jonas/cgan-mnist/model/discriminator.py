from keras import Sequential
from keras.layers import *
from keras.optimizer_v2.adam import Adam
from keras.models import Model
from keras.initializers.initializers_v2 import RandomNormal
from keras.losses import BinaryCrossentropy
from keras.metrics import BinaryAccuracy

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import NUM_CLASSES, IMG_DIM, LEARNING_RATE

def make_discriminator_model() -> Model:
    input_label = Input(shape=(1,), dtype='int32')
    in_label = Embedding(NUM_CLASSES, IMG_DIM)(input_label)
    in_label = Flatten()(in_label)

    input_img = Input(shape=(IMG_DIM,))

    input_discriminator = multiply([input_img, in_label])

    validity = discriminator()(input_discriminator)

    model = Model([input_img, input_label], validity, name="Discriminator_Input")

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE, beta_1=0.5),
        loss=BinaryCrossentropy(),
        metrics= [BinaryAccuracy()]
    )

    model.summary()



    return model


def discriminator() -> Model:
    model = Sequential([
        # input
        Dense(128, input_shape=(IMG_DIM,), kernel_initializer=RandomNormal(stddev=0.02), name="input"),
        LeakyReLU(alpha=0.2),

        # hidden layer 1
        Dense(256, name="hidden1"),
        LeakyReLU(alpha=0.2),

        # hidden layer 2
        Dense(512, name="hidden2"),
        LeakyReLU(alpha=0.2),

        # output
        Dense(1, activation='sigmoid', name="output"),
    ], name="Discriminator_Layers")

    model.summary()

    return model