import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input, optimizers, losses, preprocessing, models
import os
from tqdm import tqdm
import numpy as np
from storage import Storage


# largely inspired by these tutorials:
# https://www.youtube.com/watch?v=eR5ZnFWekNQ
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/


#  some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

### VARIABLES ###
EPOCHS = 10000
BATCH_SIZE = 28
LABEL_AMOUNT = 2
LATENT_DIM = 100
IMG_SIZE = 64
IMG_CHANNELS = 1

LOAD_CHECKPOINT = True
CHECKPOINT_DATE = "12-11-2021"
SAVE_CHECKPOINT = True
### VARIABLES ### 

# load dataset
dataset: tf.data.Dataset = preprocessing.image_dataset_from_directory(
    directory="data/training",
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    shuffle=True,
)

# normalize images (range [0, 255] to [-1.0, 1.0])
dataset = dataset.map(lambda img, label: ((img - 127.5)/127.0, label))


def define_discriminator(img_size: int, img_channels: int, label_amount: int) -> models.Model:
    # helpers
    img_dimension = (img_size, img_size, img_channels)
    img_as_nodes = img_size * img_size * img_channels

    # input for label -> scale input for label to match dimensions of image
    # first layer is called differently because it's needed for model description
    input_label = Input(shape=(1,))
    in_label = layers.Embedding(label_amount, 50)(input_label)
    in_label = layers.Dense(img_as_nodes)(in_label)
    in_label = layers.Reshape(img_dimension)(in_label)

    input_img = Input(shape=img_dimension)

    # combine image and label input + extra layers
    merge = layers.Concatenate()([input_img, in_label])
    merge = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)
    merge = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)
    merge = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)

    # output including downsizing model
    out = layers.Flatten()(merge)
    out = layers.Dropout(0.4)(out)
    out = layers.Dense(1, activation='sigmoid')(out)

    # model
    model = models.Model([input_img, input_label], out)
    optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    loss_function = losses.BinaryCrossentropy()
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=['accuracy']
    )
    return model


def define_generator(latent_dim: int, label_amount: int) -> models.Model:
    # input label -> convert to 16*16*1
    input_label = Input(shape=(1,), name="label")
    in_label = layers.Embedding(
        input_dim=label_amount, output_dim=50)(input_label)
    in_label = layers.Dense(16 * 16 * 1)(in_label)
    in_label = layers.Reshape((16, 16, 1))(in_label)

    # input random noise -> convert to 16x16x127
    input_latent = Input(shape=(latent_dim,), name="random noise")
    in_latent = layers.Dense(16*16*127)(input_latent)
    in_latent = layers.LeakyReLU(alpha=0.2)(in_latent)
    in_latent = layers.Reshape((16, 16, 127))(in_latent)

    # merge to shape with dimensions: 16x16x128
    merge = layers.Concatenate()([in_latent, in_label])
    # convert to 32x32x128
    merge = layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)
    # convert to 64x64x128
    merge = layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)

    # out -> downsize to 64x64x1
    out = layers.Conv2D(1, (16, 16), activation='tanh', padding='same')(merge)

    # model
    model = models.Model([input_latent, input_label], out)
    return model


def define_gan(generator: models.Model, discriminator: models.Model) -> models.Model:
    # deactivate training of discriminator -> purpose of this network is to train generator
    discriminator.trainable = False

    # connect generator output with discriminator input
    # discriminator output = gan output
    gen_noise, gen_label = generator.input
    gen_output = generator.output
    gan_output = discriminator([gen_output, gen_label])

    model = models.Model([gen_noise, gen_label], gan_output)
    optimizer = optimizers.Adam(learning_rate=2e-4, beta_1=0.5)
    loss_function = losses.BinaryCrossentropy()
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
    )
    return model


def random_generator_input(batch_size: int, latent_dim: int, label_amount: int):
    noise = np.random.randn(
        latent_dim * batch_size).reshape(batch_size, latent_dim)
    labels = np.random.randint(0, label_amount, batch_size)
    return [noise, labels]


def generate_fake_data(generator: models.Model, batch_size: int, latent_dim: int, label_amount: int):
    noise, labels = random_generator_input(
        batch_size, latent_dim, label_amount)
    return [generator.predict([noise, labels]), labels]


def train(dataset: tf.data.Dataset, discriminator: models.Model, generator: models.Model, gan: models.Model, epochs: int, latent_dim: int, label_amount: int):
    for epoch in tqdm(range(epochs)):
        for batch in dataset:
            # cannot be set statically -> e.g. if dataset is not dividable by batch-size the last batch is smaller
            batch_size = batch[0].shape[0]

            # train discriminator
            discriminator.train_on_batch(
                batch,
                tf.ones((batch_size, 1))
            )
            fake_data = generate_fake_data(
                generator, batch_size, latent_dim, label_amount)
            discriminator.train_on_batch(
                fake_data,
                tf.zeros((batch_size, 1))
            )

            # train generator through gan
            gan.train_on_batch(
                random_generator_input(batch_size, latent_dim, label_amount),
                tf.ones((batch_size, 1))
            )

            # save some samples
            if epoch % 100 == 0:
                images, labels = fake_data
                img = preprocessing.image.array_to_img(images[0])
                img.save(
                    f"data/results/epoch_{epoch}_label_{labels[0]}.png")

    if SAVE_CHECKPOINT:
        storage.save_checkpoint(discriminator, generator)            


discriminator = define_discriminator(IMG_SIZE, IMG_CHANNELS, LABEL_AMOUNT)
generator = define_generator(LATENT_DIM, LABEL_AMOUNT)

storage = Storage("data/neural_networks", "first_training")
if LOAD_CHECKPOINT:
    discriminator, generator = storage.load_checkpoint(discriminator, generator, CHECKPOINT_DATE)

gan = define_gan(generator, discriminator)
print(gan.summary())

train(dataset, discriminator, generator, gan, EPOCHS, LATENT_DIM, LABEL_AMOUNT)
