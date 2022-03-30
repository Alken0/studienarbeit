from keras.layers import *
from keras.models import Model, Sequential
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
    
    # dense 1
    merge = Dense(128, input_shape=(LATENT_DIM,), kernel_initializer=RandomNormal(stddev=0.02))(merge)
    merge = LeakyReLU(alpha=0.2)(merge)
    merge = BatchNormalization(momentum=0.8)(merge)

    # hidden layer 1
    merge = Dense(256)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)
    merge = BatchNormalization(momentum=0.8)(merge)

    # hidden layer 2
    merge = Dense(256)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)
    merge = BatchNormalization(momentum=0.8)(merge)

    out = Dense(IMG_DIM, activation='tanh', name="output")(merge)
    out = Reshape((IMG_SIZE, IMG_SIZE))(out)
    
    model = Model([input_latent, input_label], out, name="Generator")

    return model
