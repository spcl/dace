# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
try:
    import tensorflow as tf
except ImportError:
    print("WARNING: Tensorflow not found, skipping test")
    exit(0)

from dace.frontend.tensorflow import TFSession

if __name__ == '__main__':
    t = tf.placeholder(tf.int32, [3, 2, 3])
    b = tf.placeholder(tf.int32, [3])
    s = tf.placeholder(tf.int32, [3])
    output = tf.placeholder(tf.int32, [1, 1, 3])
    output = tf.slice(t, b, s)
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]],
                                [[5, 5, 5], [6, 6, 6]]])

    sess_tf = tf.Session()
    sess_dace = TFSession()

    begin_tensor = tf.constant([1, 0, 0])
    size_tensor_1 = tf.constant([1, 2, 2])
    size_tensor_2 = tf.constant([1, 2, 3])
    size_tensor_3 = tf.constant([2, 1, 3])
    tf_out = sess_tf.run(tf.slice(input_tensor, begin_tensor, size_tensor_3))
    dace_out = sess_dace.run(
        tf.slice(input_tensor, begin_tensor, size_tensor_3))
    print(tf_out)
    print(dace_out)
    assert (tf_out == dace_out).all()
