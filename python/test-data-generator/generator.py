from typing import List
import configuration as config
import os
import shutil
import inspect
from PIL import Image
from skimage.draw import random_shapes
from configuration import Shape

class Generator:
    CONFIG = config.load()
    EXPORT_DIR = "data/training"

    def __generate_shape(self, shape: str, amount: int) -> None:
        digit_format = len(str(amount - 1))

        for i in range(amount):
            image, _ = random_shapes(
                (self.CONFIG.image_size, self.CONFIG.image_size),
                max_shapes=1,
                shape=shape,
                random_seed=(self.CONFIG.seed+i),
                multichannel=False,
                min_size=self.CONFIG.min_figur_size,
                intensity_range=((0, 0))
            )

            path = f"{self.EXPORT_DIR}/{shape}/{i:0{digit_format}d}.png"
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
                self.__generate_shape(key, shape.get('amount'))
        

       
                
       
     

Generator().generate()
