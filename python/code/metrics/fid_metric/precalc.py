import os
import glob
import numpy as np
import metrics.fid_metric.script as fid
from imageio import imread
import tensorflow as tf

import sys

# Adds higher directory to python modules path.
sys.path.append(".")

def _calc_and_save_for_path(data_path: str) -> None:
    output_path = f"${data_path}/fid_stats.npz"

    # if you have downloaded and extracted
    #   http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    # set this path to the directory where the extracted files are, otherwise
    # just set it to None and the script will later download the files for you
    inception_path = None
    print("check for inception model..", end=" ", flush=True)
    inception_path = fid.check_or_download_inception(inception_path) # download inception if necessary
    print("ok")

    # loads all images into memory (this might require a lot of RAM!)
    print("load images..", end=" " , flush=True)
    image_list = glob.glob(os.path.join(data_path, '*.png'))
    images = np.array([imread(str(fn)).astype(np.float32) for fn in image_list])
    print("%d images found and loaded" % len(images))

    print("create inception graph..", end=" ", flush=True)
    fid.create_inception_graph(inception_path)  # load the graph into the current TF graph
    print("ok")

    print("calculte FID stats..", end=" ", flush=True)
    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        mu, sigma = fid.calculate_activation_statistics(images, sess, batch_size=100)
        np.savez_compressed(output_path, mu=mu, sigma=sigma)
    print("finished")

def precalc() -> None:
    from constants import IMG_DIR, CLASS_NAMES
    data_path = IMG_DIR

    for c in CLASS_NAMES:
        class_path = f"{data_path}/{c}"
        print(class_path)
        _calc_and_save_for_path(class_path)