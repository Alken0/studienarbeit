import shutil
import os
from tensorflow import summary
from keras.models import Model
from keras.callbacks import TensorBoard
from tensorboard.plugins.hparams import api as hp 
import sys
from datetime import datetime
import tensorflow as tf
from keras import metrics
import tensorflow as tf

# Adds higher directory to python modules path.
sys.path.append(".")
from constants import LOG_PATH, MODEL_NAME, LOSS_FUNCTION
import hyperparameters

def get_log_dir(name: str) -> str:
    import hyperparameters
    path = f"{LOG_PATH}/{MODEL_NAME}"
    return f"{path}/{hyperparameters.to_string()}/{name}"


class MetricLogger:
    log_dir: str
    name: str
    discriminator: Model
    generator: Model

    def __init__(self, name: str, discriminator: Model, generator: Model):
        self.name = name
        self.log_dir = get_log_dir(name)

        discriminator.compile(
            loss=LOSS_FUNCTION,
            metrics = [
                metrics.BinaryAccuracy()
            ]
        )

        self.discriminator = discriminator
        self.generator = generator
        
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        
        self.summary_writer = summary.create_file_writer(self.log_dir)
    
        # todo: callback liest nur den konzept graph, nicht den echten
        #tb_callback = TensorBoard(self.log_dir)
        #tb_callback.set_model(discriminator)

    def log(self, dataset: tf.data.Dataset, epoch):
        raise NotImplementedError
        
    def _write_log(self, model: Model, epoch_no):
        with self.summary_writer.as_default():
            for metric in model.metrics:
                hp.hparams(hyperparameters.to_tf_hp())
                summary.scalar(metric.name, metric.result(), epoch_no)
