from keras.models import Model
import sys
import time
from tensorflow.python.data import Dataset

from .train_step import train_step
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import EPOCHS
from logger import generate_and_save_images

def train(generator: Model, discriminator: Model, dataset: Dataset):
    for epoch in range(EPOCHS):
        start = time.time()
        for image_batch in dataset:
            train_step(generator, discriminator, image_batch)
        
        # logging
        generate_and_save_images(generator, epoch+1)
        print(f'Time for epoch {epoch + 1} is {time.time()-start} sec')
