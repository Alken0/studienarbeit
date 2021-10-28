import configuration as config
import os
import shutil
from PIL import Image, ImageDraw
from skimage.draw import random_shapes
from enum import Enum


class Shape(Enum):
    # value has to match 'random_shapes' shape input value!
    TRIANGLE = 'triangle'
    RECTANGLE = 'rectangle'
    CIRCLE = 'circle'


class Generator:
    CONFIG = config.load()
    EXPORT_DIR = "data/training"

    def __generate_shape(self, shape: Shape, amount: int) -> None:
        digit_format = len(str(self.CONFIG.image_amount - 1))

        for i in range(amount):
            image, _ = random_shapes(
                (self.CONFIG.image_size, self.CONFIG.image_size),
                max_shapes=1,
                shape=shape.value,
                random_seed=(self.CONFIG.seed+i),
                multichannel=False,
                min_size=self.CONFIG.min_figur_size,
                intensity_range=((0, 0))
            )

            path = f"{self.EXPORT_DIR}/{shape.value}/{i:0{digit_format}d}.png"
            Image.fromarray(image).save(path, "PNG")

    def __cleanExportDir(self) -> None:
        if os.path.exists(self.EXPORT_DIR):
            shutil.rmtree(self.EXPORT_DIR)

        for shape in Shape:
            os.makedirs(f"{self.EXPORT_DIR}/{shape.value}")

    def generate(self) -> None:
        self.__cleanExportDir()

        for shape in Shape:
            self.__generate_shape(shape, self.CONFIG.image_amount)


Generator().generate()
