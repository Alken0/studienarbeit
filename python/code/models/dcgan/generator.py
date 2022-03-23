from keras.layers import *
from keras.models import Model
from keras.activations import tanh
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_generator_model() -> Model:
    from constants import NUM_CLASSES, LATENT_DIM, EMBEDDING_SIZE, IMG_SIZE

    # input label 
    input_label = Input(shape=(1,), name="label_input")
    in_label = Embedding(
        input_dim=NUM_CLASSES, output_dim=EMBEDDING_SIZE,  name="label_embedding")(input_label)
    in_label = Flatten()(in_label)    
    in_label = Dense(7 * 7 * 1,  name="label_dense")(in_label)
    in_label = Reshape((7, 7, 1),  name="label_reshape")(in_label)

    # input random noise 
    input_latent = Input(shape=(LATENT_DIM,), name="noise_input")
    in_latent = Dense(7*7*255, name="noise_dense", use_bias=False)(input_latent)
    in_latent = BatchNormalization()(in_latent)
    in_latent = LeakyReLU(name="noise_reLu")(in_latent)
    in_latent = Reshape((7, 7, 255), name="noise_reshape")(in_latent)

    merge = Concatenate()([in_latent, in_label])

    merge = Conv2DTranspose(
        128, (5, 5), strides=(1, 1), padding='same', use_bias=False)(merge)
    merge = BatchNormalization()(merge)
    merge = LeakyReLU()(merge)

    merge = Conv2DTranspose(
        64, (5, 5), strides=(2, 2), padding='same', use_bias=False)(merge)
    merge = BatchNormalization()(merge)
    merge = LeakyReLU()(merge)

    merge = Conv2DTranspose(
        1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation="tanh")(merge)

    out = Reshape((IMG_SIZE, IMG_SIZE))(merge)

    model = Model([input_latent, input_label], out, name = "Generator")
    return model
    