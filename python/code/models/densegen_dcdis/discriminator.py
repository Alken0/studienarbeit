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

    # input for label -> scale input for label to match dimensions of image
    # first layer is called differently because it's needed for model description
    input_label = Input(shape=(1,), name="label_input")
    in_label = Embedding(1, EMBEDDING_SIZE)(input_label)
    in_label = Dense(4*4*1)(in_label)
    in_label = Reshape((4,4, 1), name="label_reshape")(in_label)

    # input for image
    # layers for feature extraction
    input_img = Input(shape=(IMG_SIZE, IMG_SIZE, 1), name="image_input")

    in_img = Conv2D(64, (5, 5), strides=(2, 2), padding='same')(input_img)
    in_img = LeakyReLU()(in_img)
    in_img = Dropout(DROPOUT)(in_img)

    in_img = Conv2D(128, (5, 5), strides=(2, 2), padding='same')(in_img)
    in_img = LeakyReLU()(in_img)
    in_img = Dropout(DROPOUT)(in_img)

    in_img = Conv2D(256, (5, 5), strides=(2, 2), padding='same')(in_img)
    in_img = LeakyReLU()(in_img)
    in_img = Dropout(DROPOUT)(in_img)

    # combine image and label input
    merge = Multiply()([in_img, in_label])

     # output including downsizing model
    out = Flatten()(merge)
    out = Dense(1)(out)

    # model
    model = Model([input_img, input_label], out, name = "discriminator")
    return model
