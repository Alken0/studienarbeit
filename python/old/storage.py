from datetime import date
from typing import Tuple
from tensorflow.python.keras.engine.training import Model
from tensorflow import keras


def load_checkpoint(mode_name: str, discriminator: Model, generator: Model, date: str) -> Tuple[Model, Model]:
    discriminator.load_weights(
        f"data/neural_networks/{mode_name}_{date}_discriminator")
    generator.load_weights(
        f"data/neural_networks/{mode_name}_{date}_generator")
    return (discriminator, generator)


def save_checkpoint(mode_name: str, discriminator: Model, generator: Model):
    current_date = date.today().strftime("%d-%m-%Y")
    discriminator.save
    discriminator.save_weights(
        f"data/neural_networks/{mode_name}_{current_date}_discriminator")
    generator.save_weights(
        f"data/neural_networks/{mode_name}_{current_date}_generator")
    print("checkpoint saved")


def load_model(mode_name: str) -> Tuple[Model, Model]:
    discriminator = keras.models.load_model(
        f"data/neural_networks/{mode_name}_discriminator")
    generator = keras.models.load_model(
        f"data/neural_networks/{mode_name}_generator")
    return (discriminator, generator)


def save_model(mode_name: str, discriminator: Model, generator: Model):
    discriminator.save(f"data/neural_networks/{mode_name}_discriminator")
    generator.save(f"data/neural_networks/{mode_name}_generator")
