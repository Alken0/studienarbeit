import imp
from keras import Sequential
from keras.layers import *
from keras.models import Model
from keras.initializers.initializers_v2 import GlorotNormal
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_discriminator_model() -> Model:
    from constants import NUM_CLASSES, EMBEDDING_SIZE, DROPOUT, IMG_SIZE, IMG_CHANNELS
    img_dimension = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

    input_label = Input(shape=(1), name="label_input")
    in_label = Dense(IMG_SIZE * IMG_SIZE * IMG_CHANNELS, name="label_dense")(input_label)
    in_label = Reshape(img_dimension, name="label_reshape")(in_label)
    
    input_img = Input(shape=img_dimension, name="img_input")

    merge = Concatenate()([input_img, in_label])

    conv = Conv2D(32, (4,4), strides=(3,3))(merge)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2D(64, (4,4), dilation_rate=(2,2))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2D(128, (4,4), strides=(3,3))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2D(256, (4,4), dilation_rate=(2,2))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    conv = Conv2D(512, (4,4), strides=(2,2))(conv)
    conv = LeakyReLU(alpha=0.2)(conv)

    out = Flatten()(conv)
    out = Dropout(rate=DROPOUT)(out)
    out = Dense(1, activation="tanh")(out)
   
    model = Model([input_img, input_label], out, name = "discriminator")
    model.summary()
    return model
