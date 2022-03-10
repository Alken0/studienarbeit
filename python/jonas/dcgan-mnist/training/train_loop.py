from .train_step import train_step

import time
from IPython import display

import sys
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import SEED, CHECKPOINT_PREFIX, EPOCHS
from logger import generate_and_save_images, make_checkpoint

def train(generator, discriminator, dataset):
    checkpoint = make_checkpoint(generator, discriminator)

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in dataset:
            train_step(generator, discriminator, image_batch)

        # Produce images for the GIF as you go
        display.clear_output(wait=True)
        generate_and_save_images(generator,
                                 epoch + 1,
                                 SEED)

        # Save the model every 15 epochs
        if (epoch + 1) % 15 == 0:
            checkpoint.save(file_prefix=CHECKPOINT_PREFIX)

        print('Time for epoch {} is {} sec'.format(
            epoch + 1, time.time()-start))

    # Generate after the final epoch
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             EPOCHS,
                             SEED)

