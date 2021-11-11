import tensorflow as tf
from tensorflow.keras import layers, Sequential, Input, optimizers, losses, preprocessing
import os
from tqdm import tqdm


# largely inspired by this video: https://www.youtube.com/watch?v=eR5ZnFWekNQ


# some settings
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
physical_devices = tf.config.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(physical_devices[0], True)

BATCH_SIZE = 28 # increase this to maximum your gpu can handle

dataset = preprocessing.image_dataset_from_directory(
    directory="data/training",
    labels='inferred',
    label_mode='int',
    color_mode='grayscale',
    image_size=(64, 64),
    batch_size=BATCH_SIZE,
    shuffle=True,
).map(lambda x: x / 255.0)

discriminator = Sequential(
    layers=[
        Input(shape=(64, 64, 1)),
        layers.Conv2D(64, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Flatten(),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ],
    name="discriminator"
)
print(discriminator.summary())


latent_dim = 128
generator = Sequential(
    layers=[
        layers.Input(shape=(latent_dim)),
        layers.Dense(8*8*128),
        layers.Reshape((8, 8, 128)),
        layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding="same"),
        layers.LeakyReLU(0.2),
        layers.Conv2D(1, kernel_size=3, padding="same", activation="sigmoid")
    ],
    name="generator"
)
print(generator.summary())

optimizer_generator = optimizers.Adam(1e-4)
optimizer_discriminator = optimizers.Adam(1e-4)
loss_function = losses.BinaryCrossentropy()


for epoch in tqdm(range(100000)):
    for index, real in enumerate(dataset):
        batch_size = real.shape[0]
        random_latent_vectors = tf.random.normal(
            shape=(batch_size, latent_dim))

        fake = generator(random_latent_vectors)

        if epoch % 100 == 0 and index % 100 == 0:
            img = preprocessing.image.array_to_img(fake[0])
            img.save(f"data/results/generated_image_{epoch}_{index}.png")

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        with tf.GradientTape() as disc_tape:
            loss_discriminator_real = loss_function(
                tf.ones((batch_size, 1)),
                discriminator(real)
            )
            loss_discriminator_fake = loss_function(
                tf.zeros((batch_size, 1)),
                discriminator(fake)
            )
            loss_discriminator = (loss_discriminator_real +
                                  loss_discriminator_fake) / 2.0

        gradients = disc_tape.gradient(
            loss_discriminator, discriminator.trainable_weights)
        optimizer_discriminator.apply_gradients(
            zip(gradients, discriminator.trainable_weights))

        # Train Generator min log(1 - D(G(z))) <-> max log (D(G(z)))
        with tf.GradientTape() as gen_tape:
            fake = generator(random_latent_vectors)
            output = discriminator(fake)
            loss_generator = loss_function(tf.ones(batch_size, 1), output)

        gradients = gen_tape.gradient(
            loss_generator, generator.trainable_weights)
        optimizer_generator.apply_gradients(
            zip(gradients, generator.trainable_weights))
