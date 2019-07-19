import tensorflow as tf
from tensorflow.python.ops import gen_nn_ops
import numpy as np
import dace
from dace.frontend.tensorflow import TFSession
from dace.transformation.interstate import GPUTransformState
from dace.transformation.dataflow import (
    GPUTransformLocalStorage,
    GPUTransformMap,
    RedundantArray,
    RedundantArrayCopying,
    RedundantArrayCopying2,
    RedundantArrayCopying3,
)

dace.Config.append("compiler", "cpu", "args", value=" -faligned-new")
inp_shape = [10, 56, 56, 64]
filter_shape = [3, 3, 64, 64]
strides = [1, 1, 1, 1]

inp = tf.placeholder(tf.float32, inp_shape)
filter = tf.placeholder(tf.float32, filter_shape)
outp = tf.nn.conv2d(
    inp, filter, strides, padding="SAME", data_format="NHWC", use_cudnn_on_gpu=True
)

test_in = np.random.uniform(size=tuple(inp_shape)).astype(np.float32)
test_filter = np.random.uniform(size=tuple(filter_shape)).astype(np.float32)

sess_dace = TFSession()
sess_tf = tf.Session()

output_dace = sess_dace.run(
    outp,
    transformations=[
        [GPUTransformLocalStorage],
        [
            RedundantArray,
            RedundantArrayCopying3,
            RedundantArrayCopying2,
            RedundantArrayCopying,
        ],
    ],
    feed_dict={inp: test_in, filter: test_filter},
)
output_tf = sess_tf.run(outp, feed_dict={inp: test_in, filter: test_filter})
print(output_tf)
print(output_dace)
print(tf.linalg.norm(output_tf - output_dace).eval(session=sess_tf))
