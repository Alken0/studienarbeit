import importlib
from constants import MODEL_NAME
import constants
from keras.utils.vis_utils import plot_model

# to run:
# 1. install + add to path https://graphviz.gitlab.io/download/
# pip install (for pydot-ng)
# 2. restart ide

constants.EMBEDDING_SIZE = 50

models = importlib.import_module(MODEL_NAME)

generator = models.make_generator_model()
discriminator = models.make_discriminator_model()

model_name = MODEL_NAME.removeprefix('models.')

plot_model(
    generator,
    to_file=f"{model_name}_generator.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
    show_layer_activations=True
)

plot_model(
    discriminator,
    to_file=f"{model_name}_discriminator.png",
    show_shapes=True,
    show_dtype=False,
    show_layer_names=False,
    rankdir="TB",
    expand_nested=True,
    dpi=96,
    show_layer_activations=True
)
