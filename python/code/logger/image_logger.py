from .metric_logger.util.generate import generate_fake_data, generate_labels_evenly, generate_random_noise
import sys
from keras.models import Model
from PIL import Image
import hyperparameters
import numpy as np
import os
import shutil

# Adds higher directory to python modules path.
sys.path.append(".")

def _clear_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)        

def write_image(generator: Model):
    from constants import NUM_CLASSES, CLASS_NAMES, LOG_IMG_PER_LABEL_TO_OS, IMG_RESULT_DIR 

    dir = f"{IMG_RESULT_DIR}/{hyperparameters.to_string()}"
    for c in CLASS_NAMES:
        _clear_dir(f"{dir}/{c}")

    noise = generate_random_noise(LOG_IMG_PER_LABEL_TO_OS)
    labels = generate_labels_evenly(LOG_IMG_PER_LABEL_TO_OS)
    images = generate_fake_data(generator, noise, labels)

    for i, label in enumerate(labels):
        path = f"{dir}/{CLASS_NAMES[int(label)]}/{i}.png"
        image = images[i].numpy()
        image = (image * 127.0) + 127.5
        Image.fromarray(image).convert('RGB').save(path, "PNG")    
        #.reshape((1, -1))
