# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import tensorflow as tf
import numpy as np
from dace.frontend.tensorflow import TFSession
import matplotlib.pyplot as plt
import sys


def data_input_fn(filenames, batch_size=2, shuffle=False):
    def _parser(record):
        features = {
            "label": tf.FixedLenFeature([], tf.int64),
            "image_raw": tf.FixedLenFeature([], tf.string),
        }
        parsed_record = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_record["image_raw"], tf.float32)
        image = tf.reshape(image, [28, 28])

        label = tf.cast(parsed_record["label"], tf.int32)
        label = tf.one_hot(indices=label, depth=10, on_value=1, off_value=0)
        return image, tf.one_hot(label, depth=10)

    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames).map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10000)

        dataset = dataset.batch(batch_size, drop_remainder=True)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels

    return _input_fn


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('USAGE: dataset_reader.py <FILENAME> [FILENAMES...]')
        exit(1)

    filenames = list(sys.argv[1:])

    with tf.Session() as sess:
        output_tf = sess.run(data_input_fn(filenames)())[0]
        for _out in output_tf:
            _out = np.multiply(255.0, _out)
            _out = _out.astype(np.uint8)
            plt.imshow(_out)
            plt.show()

    with TFSession() as sess:
        output_dace = sess.run(data_input_fn(filenames)())[0]
        for _out in output_dace:
            _out = np.multiply(255.0, _out)
            _out = _out.astype(np.uint8)
            plt.imshow(_out)
            plt.show()
