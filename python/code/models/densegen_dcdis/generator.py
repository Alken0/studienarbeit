from keras import models
from keras.layers import *
from keras.models import Model
from keras.initializers.initializers_v2 import RandomNormal
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_generator_model() -> Model:
    from constants import NUM_CLASSES, LATENT_DIM, IMG_DIM, LATENT_DIM, IMG_SIZE

    input_label = Input(shape=(1,), dtype='int32')
    in_label = Embedding(NUM_CLASSES, LATENT_DIM)(input_label)
    in_label = Flatten()(in_label)

    input_latent = Input(shape=(LATENT_DIM,))

    merge = multiply([input_latent, in_label])

    merge = Dense(128, input_shape=(LATENT_DIM,), kernel_initializer=RandomNormal(stddev=0.02))(merge)
    LeakyReLU(alpha=0.2)
    BatchNormalization(momentum=0.8)

    merge = Dense(256)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)
    merge = BatchNormalization(momentum=0.8)(merge)

    merge = Dense(256)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)
    merge = BatchNormalization(momentum=0.8)(merge)

    merge = Dense(IMG_DIM, activation='tanh')(merge)
    merge = Reshape((IMG_SIZE, IMG_SIZE))(merge)
    
    model = models.Model([input_latent, input_label], merge, name="Generator_Inputs")

    return model
