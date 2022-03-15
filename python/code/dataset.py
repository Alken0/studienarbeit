from keras.preprocessing.image_dataset import image_dataset_from_directory
import tensorflow as tf

def make_dataset() -> tuple[tf.data.Dataset, tf.data.Dataset]:
    from constants import IMG_SIZE, BATCH_SIZE, IMG_DIR

    # https://keras.io/examples/vision/image_classification_from_scratch/
    
    dataset_train = image_dataset_from_directory(
        directory=IMG_DIR,
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        seed=1337,
        subset="training"
    ).map(lambda img, label: ((img - 127.5)/127.0, label))

    dataset_test = image_dataset_from_directory(
        directory=IMG_DIR,
        labels='inferred',
        label_mode='int',
        color_mode='grayscale',
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        shuffle=True,
        validation_split=0.2,
        seed=1337,
        subset="validation"
    ).map(lambda img, label: ((img - 127.5)/127.0, label))
    
    return dataset_train, dataset_test
