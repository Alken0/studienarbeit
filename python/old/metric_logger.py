import shutil
import os
from tensorflow import summary
from gan_model import HP_DIS_DROPOUT, HP_DIS_LR, HP_GAN_LR
from keras import models
import tensorflow as tf
from tensorboard.plugins.hparams import api as hp 


LOG_PATH = "./data/logs"


class Logger:

    def __init__(self, name: str, model: models.Model, hparams):
        self.log_dir = f"{LOG_PATH}/dis_lr_{hparams[HP_DIS_LR]}-gan_lr_{hparams[HP_GAN_LR]}-dis_drop_{hparams[HP_DIS_DROPOUT]}/{name}"
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        self.summary_writer = summary.create_file_writer(self.log_dir)
    
        # todo: callback liest nur den konzept graph, nicht den echten
        tb_callback = tf.keras.callbacks.TensorBoard(self.log_dir)
        tb_callback.set_model(model)

    def write_log(self, model: models.Model, epoch_no, hparams):
        with self.summary_writer.as_default():
            for metric in model.metrics:
                hp.hparams(hparams)
                summary.scalar(
                    metric.name, metric.result(), epoch_no)

    def write_image(self, tag, data, step):
        with self.summary_writer.as_default():
            summary.image(f"{tag}", data,
                          max_outputs=len(data), step=step)
