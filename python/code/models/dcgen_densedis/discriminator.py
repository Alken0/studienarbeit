from keras.layers import *
from keras.models import Model
import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def make_discriminator_model() -> Model:
    from constants import NUM_CLASSES, IMG_DIM, IMG_SIZE, DROPOUT

    input_label = Input(shape=(1,), dtype='int32')
    in_label = Embedding(NUM_CLASSES, IMG_DIM)(input_label)
    in_label = Flatten()(in_label)

    input_img = Input(shape=(IMG_SIZE, IMG_SIZE))
    in_img = Flatten()(input_img)

    merge = multiply([in_img, in_label])

    merge = Dense(128, input_shape=(IMG_DIM,))(merge)
    merge = LeakyReLU(alpha=0.2)(merge)

    # hidden layer 1
    merge = Dense(256)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)

    # hidden layer 2
    merge = Dense(512)(merge)
    merge = LeakyReLU(alpha=0.2)(merge)

    merge = Flatten()(merge)
    merge = Dropout(DROPOUT)(merge)

    # output
    merge = Dense(1)(merge)

    model = Model([input_img, input_label], merge, name="Discriminator_Input")

    return model
