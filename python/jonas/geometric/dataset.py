from tensorflow.python.data import Dataset
from keras.preprocessing.image_dataset import image_dataset_from_directory
from constants import IMG_SIZE, BATCH_SIZE, IMG_DIR

def make_dataset() -> Dataset:
    dataset = image_dataset_from_directory(
        directory=IMG_DIR,
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    dataset = dataset.map(lambda img, label: ((img - 127.5)/127.0, label))
    return dataset