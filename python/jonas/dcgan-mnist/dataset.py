import tensorflow as tf
from constants import BUFFER_SIZE, BATCH_SIZE


def make_dataset() -> tf.data.Dataset:
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(
        train_images.shape[0], 28, 28, 1).astype('float32')
    # Normalize the images to [-1, 1]
    train_images = (train_images - 127.5) / 127.5

    # Batch and shuffle the data
    return tf.data.Dataset.from_tensor_slices(
        train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
