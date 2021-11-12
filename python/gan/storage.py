from datetime import date
from typing import Tuple
from tensorflow.python.keras.engine.training import Model

class Storage:

    def __init__(self, dir_path: str, identifier: str) -> None:
        self.dir_path = dir_path
        self.identifier = identifier

    def load_checkpoint(self, discriminator: Model, generator: Model, date: str) -> Tuple[Model, Model]:
        discriminator.load_weights(f"data/neural_networks/{self.identifier}_{date}_discriminator")
        generator.load_weights(f"data/neural_networks/{self.identifier}_{date}_generator")
        return (discriminator, generator)

    def save_checkpoint(self, discriminator: Model, generator: Model):
        current_date = date.today().strftime("%d-%m-%Y")
        discriminator.save_weights(f"data/neural_networks/{self.identifier}_{current_date}_discriminator")
        generator.save_weights(f"data/neural_networks/{self.identifier}_{current_date}_generator")
        print("checkpoint saved")

    # checkpoint format is only for savepoints (only weights are saved)
    # the final models should be saved to Protobuffer or HD5F
