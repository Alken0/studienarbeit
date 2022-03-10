from keras.datasets import mnist

class Dataset:
    def __init__(self, x_train, y_train) -> None:
        self.x_train = x_train
        self.y_train = y_train

def make_dataset() -> Dataset:
    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    # reshaping the inputs
    X_train = X_train.reshape(60000, 28*28)
    # normalizing the inputs (-1, 1)
    X_train = (X_train.astype('float32') / 255 - 0.5) * 2

    return Dataset(X_train, y_train)

