import json

FILE_PATH = "test-data-generator/config.json"

class Shape:

    generate: bool
    amount: int

class Shapes:

    triangle: Shape
    rectangle: Shape
    circle: Shape

    def __init__(self, triangle: Shape, rectangle: Shape, circle: Shape) -> None:
        self.triangle = triangle
        self.rectangle = rectangle
        self.circle = circle

class Configuration:

    seed: int
    image_size: int
    min_figur_size: int
    shapes: Shapes

    def __init__(self, seed, imageSize, minFigurSize, shapes: Shapes) -> None:
        self.seed = seed
        self.image_size = imageSize
        self.min_figur_size = minFigurSize
        self.shapes = shapes

def load() -> Configuration:
    with open(FILE_PATH) as config_file:
        content = json.load(config_file)
        return Configuration(**content)  