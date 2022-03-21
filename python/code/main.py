import matplotlib
import tensorflow as tf
import os
import importlib

# https://www.tensorflow.org/tutorials/generative/dcgan

#  some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

matplotlib.use('Agg')

'''
ATTENTION

hyperparameter set the constants!
to get the current values of the constants, the constants.py are imported inside functions!

train_step gets compiled in train - once compiled it cannot be changed during python-runtime
that's why hyperparameters need to be passed as arguments to the function (multiline lambda/... is not supported by python)
'''

from generator import Generator
import metrics

GENERATE_IMAGES = False

if GENERATE_IMAGES:
    Generator().generate()
    metrics.precalc()
    print("successfully generated")


import hyperparameters as hp

print(f"hp combinations: {len(list(hp.iterator()))}")

for params in hp.iterator():
    print("\n\n#####   new training   #####")

    hp.update(params)

    from constants import MODEL_NAME
    from training import train
    from dataset import make_dataset

    models = importlib.import_module(MODEL_NAME)

    (dataset_train, dataset_test) = make_dataset()
    generator = models.make_generator_model()
    discriminator = models.make_discriminator_model()

    train(generator, discriminator, dataset_train, dataset_test)

