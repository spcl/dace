# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np


@pytest.mark.tensorflow
def test_callback():
    import tensorflow as tf
    from dace.frontend.tensorflow import TFSession

    input_image = tf.constant(0.69, tf.float64, [2, 2, 5, 5, 2])
    conv_filter = tf.constant(0.01, tf.float64, [1, 1, 1, 2, 2])
    tests = []
    tests.append(tf.nn.conv3d(input_image, conv_filter, strides=[1, 1, 1, 1, 1], padding="VALID"))

    myinput = tf.constant(0.69, tf.float64, [2, 2])
    tests.append(tf.sigmoid(myinput))

    myinput = np.random.rand(2, 3, 4).astype(np.float64)
    tests.append(tf.reduce_max(myinput))

    myinput = np.random.rand(10).astype(np.float64)
    tests.append(tf.nn.top_k(myinput, 4)[0])
    tests.append(tf.nn.top_k(myinput, 4)[1])

    sess_tf = tf.Session()
    sess_dace = TFSession()

    for test in tests:
        output_tf = sess_tf.run(test)
        output_dace = sess_dace.run(test)
        print(output_dace)
        print(output_tf)
        assert np.linalg.norm(output_dace - output_tf) < 1e-8


if __name__ == '__main__':
    try:
        import tensorflow
        test_callback()
    except ImportError:
        pass
