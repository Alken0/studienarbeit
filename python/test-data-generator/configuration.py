import json

FILE_PATH = "config.json"


class Shape:
    generate: bool
    amount: int
    min_size: int
    max_size: int

    def __init__(self,  generate: bool, amount: int, minSize: int, maxSize: int) -> None:
        self.generate = generate
        self.amount = amount
        self.min_size = minSize
        self.max_size = maxSize


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
    shapes: Shapes

    def __init__(self, seed, imageSize, shapes: Shapes) -> None:
        self.seed = seed
        self.image_size = imageSize
        self.shapes = shapes


def load() -> Configuration:
    with open(FILE_PATH) as config_file:
        content = json.load(config_file)
        return Configuration(**content)
