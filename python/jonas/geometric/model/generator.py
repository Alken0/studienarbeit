from keras import Sequential, layers, models
from keras.layers import *
from keras.models import Model
from keras.initializers.initializers_v2 import RandomNormal

import sys

# Adds higher directory to python modules path.
sys.path.append(".")
from constants import NUM_CLASSES, IMG_DIM, LATENT_DIM, IMG_SIZE

def make_generator_model() -> Model:
    input_label = layers.Input(shape=(1,), dtype='int32')
    in_label = layers.Embedding(NUM_CLASSES, LATENT_DIM)(input_label)
    in_label = layers.Flatten()(in_label)

    input_latent = layers.Input(shape=(LATENT_DIM,))

    input_generator = layers.multiply([input_latent, in_label])

    image = generator()(input_generator)
    
    model = models.Model([input_latent, input_label], image, name="Generator_Inputs")

    model.summary()

    return model

def generator() -> Model:
    model = Sequential([
        # input
        Dense(128, input_shape=(LATENT_DIM,), kernel_initializer=RandomNormal(stddev=0.02), name="input"),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        # hidden layer 1
        Dense(256, name="hidden1"),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        # hidden layer 2
        Dense(256, name="hidden2"),
        LeakyReLU(alpha=0.2),
        BatchNormalization(momentum=0.8),

        Dense(IMG_DIM, activation='tanh', name="output"),
        Reshape((IMG_SIZE, IMG_SIZE)),
    ], name="Generator_Layers")

    model.summary()

    return model
