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
from constants import NUM_CLASSES, IMG_DIM, LEARNING_RATE, LATENT_DIM

def make_gan_model(discriminator: Model, generator: Model) -> Model:
    discriminator.trainable = False

    input_latent, input_label = generator.input
    d_g = discriminator([generator([input_latent, input_label]), input_label])

    model = Model([input_latent, input_label], d_g, name="GAN")

    model.compile(
        optimizer=Adam(lr=LEARNING_RATE, beta_1=0.5),
        loss=BinaryCrossentropy(),
        metrics= [BinaryAccuracy()]
    )

    model.summary()
    
    return model
