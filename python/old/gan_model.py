from tensorflow.keras import layers, Input, metrics, optimizers, losses, models, activations
from tensorboard.plugins.hparams import api as hp 

# largely inspired by these tutorials:
# https://www.youtube.com/watch?v=eR5ZnFWekNQ
# https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/

### HYPER PARAMS ###
HP_EMBEDDING_SIZE = hp.HParam("Embedding Size", hp.Discrete([30,50]))

# HP_DIS_NUM_UNITS = hp.HParam("DIS: num units", hp.Discrete([32, 64, 128]))
HP_DIS_DROPOUT = hp.HParam("DIS: dropout", hp.Discrete([0.4]))
HP_DIS_LR = hp.HParam("DIS: learning_rate", hp.Discrete([2e-4, 2e-3]))

# HP_GAN_NUM_UNITS = hp.HParam("GAN: num units", hp.Discrete([32, 64, 128]))
HP_GAN_LR = hp.HParam("GAN: learning_rate", hp.Discrete([2e-4, 2e-3]))
HP_GAN_LABEL_DENSE =  hp.HParam("GAN: label_dense", hp.Discrete([1]))
### HYPER PARAMS ###

def define_discriminator(img_size: int, img_channels: int, label_amount: int, hparams) -> models.Model:
    # helpers
    img_dimension = (img_size, img_size, img_channels)
    img_as_nodes = img_size * img_size * img_channels

    # input for label -> scale input for label to match dimensions of image
    # first layer is called differently because it's needed for model description
    input_label = Input(shape=(3,), name="label_input")
    in_label = layers.Embedding(label_amount, hparams[HP_EMBEDDING_SIZE])(input_label)
    in_label = layers.Flatten()(in_label)
    in_label = layers.Dense(4*4*1)(in_label)
    in_label = layers.Reshape((4,4, 1), name="label_reshape")(in_label)

    # input for image
    # layers for feature extraction
    input_img = Input(shape=img_dimension, name="image_input")
    in_img = layers.Conv2D(32, (3, 3), strides=(2, 2), padding='same')(input_img)
    in_img = layers.LeakyReLU(alpha=0.2)(in_img)
    in_img = layers.Conv2D(64, (3, 3), strides=(2, 2), padding='same')(in_img)
    in_img = layers.LeakyReLU(alpha=0.2)(in_img)
    in_img = layers.Conv2D(128, (3, 3), strides=(2, 2), padding='same')(in_img)
    in_img = layers.LeakyReLU(alpha=0.2)(in_img)

    # combine image and label input
    merge = layers.Concatenate()([in_img, in_label])

    # output including downsizing model
    out = layers.Flatten()(merge)
    out = layers.Dropout(hparams[HP_DIS_DROPOUT])(out)
    out = layers.Dense(1, activation='sigmoid')(out)

    # model
    model = models.Model([input_img, input_label], out, name = "discriminator")
    optimizer = optimizers.Adam(learning_rate=hparams[HP_DIS_LR], beta_1=0.5)
    loss_function = losses.BinaryCrossentropy()
    discriminator_metrics = [metrics.BinaryAccuracy(name="dis_acc")]
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=discriminator_metrics
    )
    print(model.summary())
    return model


def define_generator(latent_dim: int, label_amount: int, hparams) -> models.Model:
    # input label 
    input_label = Input(shape=(3,), name="label_input")
    in_label = layers.Embedding(
        input_dim=label_amount, output_dim=hparams[HP_EMBEDDING_SIZE],  name="label_embedding")(input_label)
    in_label = layers.Flatten()(in_label)    
    in_label = layers.Dense(4 * 4 * hparams[HP_GAN_LABEL_DENSE],  name="label_dense")(in_label)
    in_label = layers.Reshape((4, 4, hparams[HP_GAN_LABEL_DENSE]),  name="label_reshape")(in_label)

    # input random noise 
    input_latent = Input(shape=(latent_dim,), name="noise_input")
    in_latent = layers.Dense(4*4*(128-hparams[HP_GAN_LABEL_DENSE]), name="noise_dense")(input_latent)
    in_latent = layers.LeakyReLU(alpha=0.2, name="noise_reLu")(in_latent)
    in_latent = layers.Reshape((4, 4, (128-hparams[HP_GAN_LABEL_DENSE])), name="noise_reshape")(in_latent)


    merge = layers.Concatenate()([in_latent, in_label])

    merge = layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(merge)
    merge = layers.BatchNormalization()(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)

    merge = layers.Conv2DTranspose(
        128, (4, 4), strides=(2, 2), padding='same')(merge)
    merge = layers.BatchNormalization()(merge)
    merge = layers.LeakyReLU(alpha=0.2)(merge)

    merge = layers.Conv2DTranspose(
        1, (4, 4), strides=(2, 2), padding='same')(merge)
    merge = layers.BatchNormalization()(merge)

    out = activations.tanh(merge)

    model = models.Model([input_latent, input_label], out, name = "Generator")
    model.summary()
    return model

def define_gan(generator: models.Model, discriminator: models.Model, hparams) -> models.Model:
    # deactivate training of discriminator -> purpose of this network is to train generator
    discriminator.trainable = False

    # connect generator output with discriminator input
    # discriminator output = gan output
    gen_noise, gen_label = generator.input
    gen_output = generator.output
    gan_output = discriminator([gen_output, gen_label])

    model = models.Model([gen_noise, gen_label], gan_output, name="GAN")
    optimizer = optimizers.Adam(learning_rate=hparams[HP_GAN_LR], beta_1=0.5)
    loss_function = losses.BinaryCrossentropy()
    gan_metrics = [metrics.BinaryAccuracy(name="gan_acc")]
    model.compile(
        loss=loss_function,
        optimizer=optimizer,
        metrics=gan_metrics
    )
    return model
