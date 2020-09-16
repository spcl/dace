# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
try:
    import tensorflow as tf
except ImportError:
    print("WARNING: Tensorflow not found, skipping test")
    exit(0)

import numpy as np
import dace
from dace.frontend.tensorflow import TFSession

if __name__ == '__main__':
    shape = [10, 11, 12, 13]

    inp = tf.placeholder(tf.float64, shape)
    outp_1 = tf.reduce_mean(inp, keepdims=True)
    outp_3 = tf.reduce_mean(inp, axis=[0, 2], keepdims=True)
    outp_0 = tf.reduce_mean(inp, axis=[0, 2])
    outp_2 = tf.reduce_mean(inp, axis=[-2, -1])
    outp_4 = tf.reduce_mean(inp, axis=[0, -1], keepdims=True)

    sess_tf = tf.Session()
    sess_dace = TFSession()
    real_inp = np.random.rand(*shape)
    for index, op in enumerate([outp_0, outp_1, outp_2, outp_3, outp_4]):
        output_tf = sess_tf.run(op, feed_dict={inp: real_inp})
        output_dace = sess_dace.run(op, feed_dict={inp: real_inp})
        try:
            assert tf.norm(output_dace -
                           output_tf).eval(session=sess_tf) < 1e-10
        except:
            print(output_dace)
            print(output_tf)
            print(tf.norm(output_dace - output_tf).eval(session=sess_tf))
            raise AssertionError("mean test {i} failed".format(i=index))

    print("mean tests passed!")
    inputs = [np.random.rand(*shape) for _ in range(10)]
    addn_test_0 = tf.add_n(inputs)
    output_tf = sess_tf.run(addn_test_0)
    output_dace = sess_dace.run(addn_test_0)
    try:
        assert tf.norm(output_dace - output_tf).eval(session=sess_tf) < 1e-10
    except:
        print(output_dace)
        print(output_tf)
        print(tf.norm(output_dace - output_tf).eval(session=sess_tf))
        raise AssertionError("AddN test failed")
    print("AddN test passed!")
