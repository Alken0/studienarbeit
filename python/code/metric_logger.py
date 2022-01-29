from tensorflow import summary
from tensorflow.keras import models, preprocessing

LOG_PATH = "./data/logs"


class Logger:

    def __init__(self, name: str):
        self.summary_writer = summary.create_file_writer(f"{LOG_PATH}/{name}")

    def write_log(self, model: models.Model, epoch_no):
        with self.summary_writer.as_default():
            for metric in model.metrics:
                summary.scalar(
                    metric.name, metric.result(), epoch_no)

    def write_images(self, label, data, epoch_no):
        with self.summary_writer.as_default():
            summary.image(f"{label}", data,
                          max_outputs=len(data), step=epoch_no)
