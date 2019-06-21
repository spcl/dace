import tensorflow as tf
import dace
from dace.frontend.tensorflow import TFSession

def data_input_fn(filenames, batch_size=1000, shuffle=False):
    def _parser(record):
        features = {
            "label": tf.FixedLenFeature([], tf.int64),
            "image_raw": tf.FixedLenFeature([], tf.string),
        }
        parsed_record = tf.parse_single_example(record, features)
        image = tf.decode_raw(parsed_record["image_raw"], tf.float32)

        label = tf.cast(parsed_record["label"], tf.int32)

        return image, tf.one_hot(label, depth=10)

    def _input_fn():
        dataset = tf.data.TFRecordDataset(filenames).map(_parser)
        if shuffle:
            dataset = dataset.shuffle(buffer_size=10_000)

        dataset = dataset.repeat(
            None
        )
        dataset = dataset.batch(batch_size)

        iterator = dataset.make_one_shot_iterator()
        features, labels = iterator.get_next()

        return features, labels

    return _input_fn

filenames = ["/home/saurabh/data/mnist/validation.tfrecords"]

with TFSession() as sess:
    try:
        while True:
            print(sess.run(data_input_fn(filenames)()))
    except tf.errors.OutOfRangeError:
        pass

with tf.Session() as sess:
    try:
        while True:
            print(sess.run(data_input_fn(filenames)()))
    except tf.errors.OutOfRangeError:
        pass
