from tensorflow.keras import layers, Input, optimizers, losses, models

# largely inspired by these tutorials:
# https://www.youtube.com/watch?v=eR5ZnFWekNQ
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/


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
    input_latent = Input(shape=(latent_dim,), name="random_noise")
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
    print("Generator:")
    model.summary()
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
