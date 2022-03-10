import matplotlib.pyplot as plt
import sys
import numpy as np

# Adds higher directory to python modules path.
sys.path.append(".")
from constants import LOG_IMG_DIR, LOG_SEED, NUM_CLASSES, NUM_EXAMPLES_PER_CLASSES, NUM_EXAMPLES

def generate_and_save_images(generator, epoch):
    labels = np.fromfunction(lambda x,_: x%3, (NUM_EXAMPLES,1), dtype=int)
    # Notice `training` is set to False.
    # This is so all layers run in inference mode (batchnorm).
    predictions = generator([LOG_SEED, labels], training=False)

    fig = plt.figure(figsize=(4, 4))

    for i in range(predictions.shape[0]):
        plt.subplot(NUM_EXAMPLES_PER_CLASSES, NUM_CLASSES, i+1)
        plt.imshow(predictions[i] * 127.5 + 127.5, cmap='gray')
        plt.axis('off')
    
    plt.savefig(f'{LOG_IMG_DIR}/image_at_epoch_{epoch:04d}.png')
    plt.close('all')