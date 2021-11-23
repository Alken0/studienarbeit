from typing import List

from skimage.draw.draw import circle
import configuration as config
import os
import shutil
from PIL import Image
from skimage.draw import rectangle, disk, polygon
import numpy as np


class Generator:
    CONFIG = config.load()
    EXPORT_DIR = "data/training"

    def __generate_rectangle(self, amount: int, min_size: int, max_size: int) -> None:
         digit_format = len(str(amount - 1))

         for i in range(amount):
        
            image = np.zeros([self.CONFIG.image_size,self.CONFIG.image_size,3],dtype=np.uint8)
            image.fill(255)
            row, col = rectangle(start=(10,5), end=(20, 39))
            image[row, col] = 0

            self.__export_image('rectangle', image, digit_format, i)

    def __generate_circle(self, amount: int, min_size: int, max_size: int) -> None:
         digit_format = len(str(amount - 1))

         for i in range(amount):
        
            image = np.zeros([self.CONFIG.image_size,self.CONFIG.image_size,3],dtype=np.uint8)
            image.fill(255)
            row, col = disk((32,32), 10)
            image[row, col] = 0

            self.__export_image('circle', image, digit_format, i)

    def __generate_triangle(self, amount: int, min_size: int, max_size: int) -> None:
         digit_format = len(str(amount - 1))

         for i in range(amount):
        
            image = np.zeros([self.CONFIG.image_size,self.CONFIG.image_size,3],dtype=np.uint8)
            image.fill(255)
            row, col = polygon((10,10,40), (10,45,25))
            image[row, col] = 0

            self.__export_image('triangle', image, digit_format, i)

    def __export_image(self,  shape, image, digit_format, image_count) -> None:
        path = f"{self.EXPORT_DIR}/{shape}/{image_count:0{digit_format}d}.png"
        Image.fromarray(image).save(path, "PNG")
    
    def __cleanExportDir(self) -> None:
        if os.path.exists(self.EXPORT_DIR):
            shutil.rmtree(self.EXPORT_DIR)

    def generate(self) -> None:
        shapes = self.CONFIG.shapes

        self.__cleanExportDir()

        for key in shapes:
            shape = shapes.get(key)
            if shape.get('generate'):
                os.makedirs(f"{self.EXPORT_DIR}/{key}")
                #todo make this more beautiful
                if key == 'rectangle':
                    self.__generate_rectangle(shape.get('amount'), shape.get('min_size'), shape.get('max_size'))
                if key == 'circle':
                    self.__generate_circle(shape.get('amount'), shape.get('min_size'), shape.get('max_size'))
                if key == 'triangle':
                     self.__generate_triangle(shape.get('amount'), shape.get('min_size'), shape.get('max_size'))      


Generator().generate()
