from unicodedata import name
from keras.layers import *
from keras.models import Model
from keras.activations import tanh
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_generator_model() -> Model:
    from constants import NUM_CLASSES, LATENT_DIM, EMBEDDING_SIZE, IMG_SIZE, IMG_CHANNELS

    input_label = Input(shape=(1), name="label_input")
    in_label = Dense(49, name="label_dense")(input_label)
    in_label = Reshape((7,7,1), name="label_reshape")(in_label)

    input_latent = Input(shape=(LATENT_DIM), name="noise_input")
    in_latent = Dense((7*7*256), name="noise_dense")(input_latent)
    in_latent = LeakyReLU(alpha=0.2)(in_latent)
    in_latent = Reshape((7,7,256), name="noise_reshape")(in_latent)

    merge = Concatenate()([in_latent, in_label])

    conv = Conv2DTranspose(filters=512, kernel_size=(4,4), strides=(2,2))(merge)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2DTranspose(filters=256, kernel_size=(4,4), dilation_rate=(2,2))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2DTranspose(filters=128, kernel_size=(5,5), strides=(3,3))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2DTranspose(filters=64, kernel_size=(5,5), dilation_rate=(2,2) )(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2DTranspose(filters=32, kernel_size=(5,5), strides=(3,3))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    out = Conv2D(filters=IMG_CHANNELS, kernel_size=(7,7), activation="tanh")(conv)

    model = Model([input_latent, input_label], out, name = "Generator")
    model.summary()
    return model
    