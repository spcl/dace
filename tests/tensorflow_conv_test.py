import tensorflow as tf
import numpy as np
import dace
from dace.frontend.tensorflow import TFSession

dace.Config.append('compiler', 'cpu', 'args', value=' -faligned-new')
inp_shape = [10, 10, 10, 10]
filter_shape = [3, 3, 10, 3]
strides = [1, 2, 2, 1]

inp = tf.placeholder(tf.float64, inp_shape)
filter = tf.placeholder(tf.float64, filter_shape)
outp = tf.nn.conv2d(inp, filter, strides, padding="SAME", data_format="NHWC")

test_in = np.random.uniform(size=tuple(inp_shape)).astype(np.float64)
test_filter = np.random.uniform(size=tuple(filter_shape)).astype(np.float64)

sess_dace = TFSession()
sess_tf = tf.Session()

output_dace = sess_dace.run(
    outp, feed_dict={
        inp: test_in,
        filter: test_filter
    })
output_tf = sess_tf.run(outp, feed_dict={inp: test_in, filter: test_filter})
try:
    assert tf.norm(output_dace - output_tf).eval(session=sess_tf) < 1e-10
except:
    print(output_tf)
    print(output_dace)
    print(tf.linalg.norm(output_tf - output_dace).eval(session=sess_tf))
    raise AssertionError("Convolution test failed")
