from __future__ import absolute_import, division, print_function
import os
import glob
#os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np
import metrics.fid_metric.script as fid
from imageio import imread
import tensorflow as tf
from logger.metric_logger.fid import FIDMetricLogger

from constants import CLASS_NAMES, IMG_DIR, IMG_RESULT_DIR, EPOCHS

def _calc_single_fid(class_name, logger: FIDMetricLogger):
    print(f"calc fid for {class_name}")
    import hyperparameters

    image_path = f"{IMG_RESULT_DIR}/{hyperparameters.to_string()}/{class_name}"
    stats_path = f"{IMG_DIR}/{class_name}/fid_stats.npz"
    inception_path = fid.check_or_download_inception(None) # download inception network

    # loads all images into memory (this might require a lot of RAM!)
    image_list = glob.glob(os.path.join(image_path, '*.png'))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])

    # load precalculated training set statistics
    f = np.load(stats_path)
    mu_real, sigma_real = f['mu'][:], f['sigma'][:]
    f.close()

    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    with tf.compat.v1.Session() as sess:
        #sess.run(tf.compat.v1.global_variables_initializer())
        mu_gen, sigma_gen = fid.calculate_activation_statistics(images, sess, batch_size=100)

    fid_value = fid.calculate_frechet_distance(mu_gen, sigma_gen, mu_real, sigma_real)
    logger.log(class_name, fid_value, EPOCHS - 1)


def calc_all_fid(logger: FIDMetricLogger):
    for c in CLASS_NAMES:
        _calc_single_fid(c, logger)

