import configuration as config
import os
import shutil
from PIL import Image
from skimage.draw import rectangle, disk, polygon
import numpy as np
import random
import math


class Generator:
    CONFIG = config.load()
    EXPORT_DIR = "data/training"

    def _generate_rectangle(self, amount: int, min_size: int, max_size: int) -> None:
        digit_format = len(str(amount - 1))
        random.seed(self.CONFIG.seed + 1000000)
        amount = round(amount / (max_size - min_size + 1))
        for size in range(min_size, max_size + 1):
            for i in range(amount):
                x = math.floor(random.random() *
                               (self.CONFIG.image_size - size))
                y = math.floor(random.random() *
                               (self.CONFIG.image_size - size))
                image = np.zeros(
                    [self.CONFIG.image_size, self.CONFIG.image_size, 3], dtype=np.uint8)
                image.fill(255)
                row, col = rectangle(start=(x, y), end=(x+size, y+size))
                image[row, col] = 0

                self._export_image(
                    'rectangle', image, digit_format, (size - min_size) * amount + i)

    def _generate_circle(self, amount: int, min_size: int, max_size: int) -> None:
        digit_format = len(str(amount - 1))
        random.seed(self.CONFIG.seed + 2000000)
        amount = round(amount / (max_size - min_size + 1))
        for size in range(min_size, max_size + 1):
            for i in range(amount):
                x = round(random.random() *
                          (self.CONFIG.image_size - 2 * size) + size)
                y = round(random.random() *
                          (self.CONFIG.image_size - 2 * size) + size)
                image = np.zeros(
                    [self.CONFIG.image_size, self.CONFIG.image_size, 3], dtype=np.uint8)
                image.fill(255)
                row, col = disk((x, y), size)
                image[row, col] = 0

                self._export_image('circle', image, digit_format,
                                   (size - min_size) * amount + i)

    def _generate_triangle(self, amount: int, min_size: int, max_size: int) -> None:
        digit_format = len(str(amount - 1))
        random.seed(self.CONFIG.seed + 3000000)
        amount = int(amount / (max_size - min_size + 1))
        for size in range(min_size, max_size + 1):
            h = round((size / 2) * np.sqrt(3))
            for i in range(amount):
                left_bottom_x = int(
                    random.random() * (self.CONFIG.image_size - size))
                left_bottom_y = int(
                    random.random() * (self.CONFIG.image_size - h) + h)
                image = np.zeros(
                    [self.CONFIG.image_size, self.CONFIG.image_size, 3], dtype=np.uint8)
                image.fill(255)
                row, col = polygon((left_bottom_y, left_bottom_y, left_bottom_y - h),
                                   (left_bottom_x, left_bottom_x + size, left_bottom_x + int(size / 2)))
                image[row, col] = 0

                self._export_image(
                    'triangle', image, digit_format, (size - min_size) * amount + i)

    def _export_image(self,  shape, image, digit_format, image_count) -> None:
        path = f"{self.EXPORT_DIR}/{shape}/{image_count:0{digit_format}d}.png"
        Image.fromarray(image).save(path, "PNG")

    def _cleanExportDir(self) -> None:
        if os.path.exists(self.EXPORT_DIR):
            shutil.rmtree(self.EXPORT_DIR)

    def generate(self) -> None:
        shapes = self.CONFIG.shapes

        self._cleanExportDir()

        for key in shapes:
            shape = shapes.get(key)
            if shape.get('generate'):
                os.makedirs(f"{self.EXPORT_DIR}/{key}")
                # todo make this more beautiful
                if key == 'rectangle':
                    self._generate_rectangle(shape.get('amount'), int(
                        shape.get('minSize')), int(shape.get('maxSize')))
                if key == 'circle':
                    self._generate_circle(shape.get('amount'), int(
                        shape.get('minSize')), int(shape.get('maxSize')))
                if key == 'triangle':
                    self._generate_triangle(shape.get('amount'), int(
                        shape.get('minSize')), int(shape.get('maxSize')))


Generator().generate()
