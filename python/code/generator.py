import os
import shutil
from PIL import Image
from skimage.draw import rectangle, disk, polygon
import numpy as np
import random
import math
from constants import *

class Generator:
    def _generate_rectangle(self, name: str, amount: int, min_size: int, max_size: int) -> None:
        os.makedirs(f"{IMG_DIR}/{name}")

        digit_format = len(str(amount - 1))
        random.seed(IMG_SEED + 1000000)
        amount = round(amount / (max_size - min_size + 1))
        for size in range(min_size, max_size + 1):
            for i in range(amount):
                x = math.floor(random.random() * (IMG_SIZE - size))
                y = math.floor(random.random() * (IMG_SIZE - size))
                image = np.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
                image.fill(255)
                row, col = rectangle(start=(x, y), end=(x+size, y+size))
                image[row, col] = 0

                self._export_image(name, image, digit_format, (size - min_size) * amount + i)

    def _generate_circle(self, name: str, amount: int, min_size: int, max_size: int) -> None:
        os.makedirs(f"{IMG_DIR}/{name}")

        digit_format = len(str(amount - 1))
        random.seed(IMG_SEED + 2000000)
        amount = round(amount / (max_size - min_size + 1))
        for size in range(min_size, max_size + 1):
            for i in range(amount):
                x = round(random.random() * (IMG_SIZE - 2 * size) + size)
                y = round(random.random() * (IMG_SIZE - 2 * size) + size)
                image = np.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
                image.fill(255)
                row, col = disk((x, y), size)
                image[row, col] = 0

                self._export_image(name, image, digit_format, (size - min_size) * amount + i)

    def _generate_triangle(self, name: str, amount: int, min_size: int, max_size: int) -> None:
        os.makedirs(f"{IMG_DIR}/{name}")

        digit_format = len(str(amount - 1))
        random.seed(IMG_SEED + 3000000)
        amount = int(amount / (max_size - min_size + 1))
        for size in range(min_size, max_size + 1):
            h = round((size / 2) * np.sqrt(3))
            for i in range(amount):
                left_bottom_x = int(random.random() * (IMG_SIZE - size))
                left_bottom_y = int(random.random() * (IMG_SIZE - h) + h)
                image = np.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=np.uint8)
                image.fill(255)
                row, col = polygon((left_bottom_y, left_bottom_y, left_bottom_y - h),
                                   (left_bottom_x, left_bottom_x + size, left_bottom_x + int(size / 2)))
                image[row, col] = 0

                self._export_image(name, image, digit_format, (size - min_size) * amount + i)

    def _export_image(self,  shape, image, digit_format, image_count) -> None:
        path = f"{IMG_DIR}/{shape}/{image_count:0{digit_format}d}.png"
        Image.fromarray(image).save(path, "PNG")

    def _cleanExportDir(self) -> None:
        if os.path.exists(IMG_DIR):
            shutil.rmtree(IMG_DIR)

    def generate(self) -> None:
        self._cleanExportDir()

        self._generate_rectangle(
            SHAPE_RECTANGLE_NAME,
            SHAPE_RECTANGLE_AMOUNT,
            SHAPE_RECTANGLE_MIN_SIZE,
            SHAPE_RECTANGLE_MAX_SIZE
        )

        self._generate_circle(
            SHAPE_CIRCLE_NAME,
            SHAPE_CIRCLE_AMOUNT,
            SHAPE_CIRCLE_MIN_SIZE,
            SHAPE_CIRCLE_MAX_SIZE
        )

        self._generate_triangle(
            SHAPE_TRIANGLE_NAME,
            SHAPE_TRIANGLE_AMOUNT,
            SHAPE_TRIANGLE_MIN_SIZE,
            SHAPE_TRIANGLE_MAX_SIZE
        )
