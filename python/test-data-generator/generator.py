import configuration as config
import os, shutil
from PIL import Image, ImageDraw 
from skimage.draw import random_shapes

class Generator:
    CONFIG = config.load()
    EXPORT_DIR = "data/training"

    def cleanExportDir(self) -> None:
        shutil.rmtree(self.EXPORT_DIR)
        os.mkdir(self.EXPORT_DIR)

    def generate(self) -> None:
        self.cleanExportDir()
        digit = len(str(self.CONFIG.image_amount - 1))
        for i in range(0, self.CONFIG.image_amount):
            image, labels = random_shapes((self.CONFIG.image_size, self.CONFIG.image_size), max_shapes=1, shape='triangle', random_seed=(self.CONFIG.seed+i), multichannel=False, min_size=self.CONFIG.min_figur_size, intensity_range=((0, 0)))
            Image.fromarray(image).save(f"{self.EXPORT_DIR}/sample_circle_{i:0{digit}d}.png", "PNG")

Generator().generate()
