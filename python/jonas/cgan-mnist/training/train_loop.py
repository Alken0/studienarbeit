import numpy as np
from keras.models import Model
import matplotlib.pyplot as plt
import tensorflow as tf

import sys

from tqdm import tqdm
# Adds higher directory to python modules path.
sys.path.append(".")
from constants import BATCH_SIZE, EPOCHS, SMOOTH, LATENT_DIM, NUM_CLASSES
from dataset import Dataset

from .train_step import train_step

def generate_and_save_images(generator: Model, epoch: int):
    samples = 10
    z = np.random.normal(loc=0, scale=1, size=(samples, LATENT_DIM))
    labels = np.arange(0, NUM_CLASSES).reshape(-1, 1)
    
    x_fake = generator.predict([z, labels])

    for k in range(samples):
        plt.subplot(2, 5, k+1)
        plt.imshow(x_fake[k].reshape(28, 28), cmap='gray')
        plt.xticks([])
        plt.yticks([])

    plt.tight_layout()
    plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
    plt.close('all')
    

def train(generator: Model, discriminator: Model, gan: Model, dataset: Dataset):
    num_batches = len(dataset.x_train) // BATCH_SIZE

    d_loss = []
    d_g_loss = []

    for e in range(EPOCHS):
        for i in range(num_batches):
            (d_loss_batch, gan_loss_batch) = train_step(generator, discriminator, gan, dataset, i)
            
            print(
                'epoch = %d/%d, batch = %d/%d, d_loss=%.3f, g_loss=%.3f' % (e, EPOCHS, i, num_batches, d_loss_batch, gan_loss_batch[0]),
                100*' ',
                end='\r'
            )

            d_loss.append(d_loss_batch)
            d_g_loss.append(gan_loss_batch)
        
        generate_and_save_images(generator, e)

