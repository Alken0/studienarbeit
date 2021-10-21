import configuration as config
import os, shutil
from PIL import Image, ImageDraw 

#todo: clear export dic before saving

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
            image = Image.new("L", (self.CONFIG.image_size, self.CONFIG.image_size), 255)
            draw = ImageDraw.Draw(image)
            draw.ellipse([(0, 0), (27, 27)], fill="black")
            image.save(f"{self.EXPORT_DIR}/sample_circle_{i:0{digit}d}.png", "PNG")

Generator().generate()
