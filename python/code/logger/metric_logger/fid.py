from logger.metric_logger.base import MetricLogger
from tensorboard.plugins.hparams import api as hp 
import sys
from keras.models import Model
from tensorflow import summary
import hyperparameters

# Adds higher directory to python modules path.
sys.path.append(".")

class FIDMetricLogger(MetricLogger):

    def __init__(self, discriminator: Model, generator: Model):
        name = "fid"
        super().__init__(name, discriminator, generator)

    def log(self, class_name, fid_value, epoch):
        with self.summary_writer.as_default():
            hp.hparams(hyperparameters.to_tf_hp())
            summary.scalar(f"fid_{class_name}", fid_value, epoch)