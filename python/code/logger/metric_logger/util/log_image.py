import matplotlib.pyplot as plt
import numpy as np
import io
import tensorflow as tf

def image_grid(data, labels, label_names, gray_scale = True):
    # Data should be in (BATCH_SIZE, H, W, C)
    # assert data.ndim == 4

    # invert image
    data = 1 - data

    figure = plt.figure(figsize=(10, 10))
    num_images = data.shape[0]
    size = int(np.ceil(np.sqrt(num_images)))

    for i in range(data.shape[0]):
        title = label_names[labels[i][0]]
        plt.subplot(size, size, i + 1, title=title)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        
        # if grayscale
        if gray_scale:
            plt.imshow(data[i], cmap=plt.cm.binary)
        else:
            plt.imshow(data[i])

    return figure

# Stolen from tensorflow official guide: https://www.tensorflow.org/tensorboard/image_summaries
def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""

    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format="png")

    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)

    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)

    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image
