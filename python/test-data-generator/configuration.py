import json

FILE_PATH = "test-data-generator/config.json"

class Configuration:

    seed: int
    image_size: int
    image_amount: int

    def __init__(self, seed, imageSize, imageAmount) -> None:
        self.seed = seed
        self.image_size = imageSize
        self.image_amount = imageAmount

def load() -> Configuration:
    with open(FILE_PATH) as config_file:
        content = json.load(config_file)
        return Configuration(**content)