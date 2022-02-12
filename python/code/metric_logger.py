import shutil
import os
from tensorflow import summary
from tensorflow.keras import models
import tensorflow as tf


LOG_PATH = "./data/logs"


class Logger:

    def __init__(self, name: str, model: models.Model):
        if os.path.exists(f"{LOG_PATH}/{name}"):
            shutil.rmtree(f"{LOG_PATH}/{name}")
        self.summary_writer = summary.create_file_writer(f"{LOG_PATH}/{name}")
        tb_callback = tf.keras.callbacks.TensorBoard(f"{LOG_PATH}/{name}")
        tb_callback.set_model(model)

    def write_log(self, model: models.Model, epoch_no):
        with self.summary_writer.as_default():
            for metric in model.metrics:
                summary.scalar(
                    metric.name, metric.result(), epoch_no)

    def write_image(self, tag, data, step):
        with self.summary_writer.as_default():
            summary.image(f"{tag}", data,
                          max_outputs=len(data), step=step)
