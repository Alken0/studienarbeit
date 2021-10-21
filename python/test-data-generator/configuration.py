import json

FILE_PATH = "config.json"

class Configuration:

    seed: int

    def __init__(self, seed) -> None:
        self.seed = seed

def load() -> Configuration:
    with open(FILE_PATH) as config_file:
        content = json.load(config_file)
        return Configuration(**content)