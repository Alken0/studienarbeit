from keras import Sequential
from keras.layers import *
from keras.models import Model
from keras.initializers.initializers_v2 import GlorotNormal
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_discriminator_model() -> Model:
    from constants import NUM_CLASSES, IMG_DIM, IMG_SIZE, DROPOUT
    initializer = GlorotNormal()

    input_label = Input(shape=(1,), dtype='int32')
    in_label = Embedding(NUM_CLASSES, IMG_DIM, embeddings_initializer=initializer)(input_label)
    in_label = Flatten()(in_label)

    input_img = Input(shape=(IMG_SIZE, IMG_SIZE))
    in_img = Flatten()(input_img)

    merge = multiply([in_img, in_label])

    # hidden layer 0
    merge = Dense(128, input_shape=(IMG_DIM,), kernel_initializer=initializer, name="input")(merge)
    merge = LeakyReLU(alpha=0.2)(merge)

    # hidden layer 1
    merge = Dense(256, name="hidden1", kernel_initializer=initializer)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)

    # hidden layer 2
    merge = Dense(512, name="hidden2", kernel_initializer=initializer)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)

    # output
    out = Flatten()(merge)
    out = Dropout(DROPOUT)(out)
    out = Dense(1, name="output", kernel_initializer=initializer)(out)

    model = Model([input_img, input_label], out, name="Discriminator_Input")

    return model
