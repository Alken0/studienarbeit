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
from keras.losses import BinaryCrossentropy
import tensorflow as tf
import numpy as np

# Adds higher directory to python modules path.
sys.path.append(".")
from constants import LOG_PATH, MODEL_NAME, BATCH_SIZE, LATENT_DIM, NUM_CLASSES

def get_log_dir(name: str) -> str:
    timestamp = datetime.now().strftime(f"%Y-%m-%d_%H-%M-%S")
    path = f"{LOG_PATH}/{MODEL_NAME}"
    return f"{path}/{timestamp}/{name}"


class MetricLogger:
    log_dir: str
    name: str

    def __init__(self, name: str, model: Model):
        self.name = name
        self.log_dir = get_log_dir(name)
        
        if os.path.exists(self.log_dir):
            shutil.rmtree(self.log_dir)
        
        self.summary_writer = summary.create_file_writer(self.log_dir)

        
    
        # todo: callback liest nur den konzept graph, nicht den echten
        tb_callback = TensorBoard(self.log_dir)
        tb_callback.set_model(model)

    def log(self, model: Model, dataset: tf.data.Dataset, epoch):
        model.reset_metrics()
        model.compile(
            loss=BinaryCrossentropy(),
            metrics = [
                metrics.BinaryAccuracy(name=f"{self.name}-accuracy")
            ]
        )
        for image_batch in dataset:
            model.evaluate(
                x=image_batch, 
                y=tf.ones(len(list(image_batch[0]))), 
                verbose=0
            )
        self._write_log(model, epoch)
    
    def log_with_generated_data(self, discriminator: Model, generator: Model, dataset: tf.data.Dataset, epoch):
        discriminator.reset_metrics()
        discriminator.compile(
            loss=BinaryCrossentropy(),
            metrics = [
                metrics.BinaryAccuracy(name=f"{self.name}-accuracy")
            ]
        )
        amount = len(list(dataset))

        noise = tf.random.normal([BATCH_SIZE, LATENT_DIM])
        labels = np.random.randint(0, NUM_CLASSES, BATCH_SIZE).reshape(-1, 1)
        fake_data = generator([noise, labels], training=False)
        discriminator.evaluate(
            x=[fake_data, labels], 
            y=tf.zeros(len(list(fake_data))),
            batch_size=BATCH_SIZE,
            verbose=0
        )
        self._write_log(discriminator, epoch)
        
    def _write_log(self, model: Model, epoch_no):
        with self.summary_writer.as_default():
            for metric in model.metrics:
                summary.scalar(metric.name, metric.result(), epoch_no)

    def write_log_original(self, model: Model, epoch_no, hparams):
        with self.summary_writer.as_default():
            for metric in model.metrics:
                hp.hparams(hparams)
                summary.scalar(
                    metric.name, metric.result(), epoch_no)

    def write_image(self, tag, data, step):
        with self.summary_writer.as_default():
            summary.image(f"{tag}", data,
                          max_outputs=len(data), step=step)


