

def log_epoch_metrics(discriminator, generator, gan, dis_logger, gan_logger, epoch, latent_dim, label_amount, hparams):
    gan_logger.write_log(gan, epoch, hparams)
    dis_logger.write_log(discriminator, epoch, hparams)
    gan.reset_metrics()
    discriminator.reset_metrics()

    """
    if epoch % EPOCHS_PER_IMAGE_SAMPLE == 0:
        images = np.empty((5, 64, 64, 1))
        labels = np.empty((5,))
        for label in range(label_amount):
            curr_images, curr_labels = generate_fake_data_by_label(
                generator, IMAGES_PER_LABEL, latent_dim, label)
            print(f"image shape: {curr_images.shape}")
            print(f"labels shape: {curr_labels.shape}")
            images = np.append(images, curr_images, axis=None)
            lables = np.append(labels, curr_labels, axis=None)
        print(f"image final: {images}")
        print(f"labels final: {labels}")
        figure = image_grid(images, labels, LABEL_NAMES)
        image = plot_to_image(figure)
        gan_logger.write_image("Composed Visualization", image, epoch)
    """
    # log Composed Visualization for each label
    if epoch % EPOCHS_PER_IMAGE_SAMPLE == 0:
        for label in range(label_amount):
            images, labels = generate_fake_data_by_label(
                generator, IMAGES_PER_LABEL, latent_dim, label, label_amount)
            figure = image_grid(images, labels, LABEL_NAMES)
            image = plot_to_image(figure)
            gan_logger.write_image(
                f"Composed_Visualization_{LABEL_NAMES[label]}", image, epoch)

    # log Composed Visualization with random labels
    if epoch % EPOCHS_PER_IMAGE_SAMPLE == 0:
        images, labels = generate_fake_data(
            generator, (IMAGES_PER_LABEL*3), latent_dim, label_amount)
        figure = image_grid(images, labels, LABEL_NAMES)
        image = plot_to_image(figure)
        gan_logger.write_image(
            f"Composed_Visualization_Random_Labels", image, epoch)

