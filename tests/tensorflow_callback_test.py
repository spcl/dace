import numpy as np
import tensorflow as tf
import dace
from dace.frontend.tensorflow import TFSession

input_image = tf.constant(0.69, tf.float64, [2, 2, 2, 2, 2])
conv_filter = tf.constant(0.01, tf.float64, [1, 1, 1, 2, 2])
tests = []
tests.append(
    tf.nn.conv3d(input_image, conv_filter, strides=[1, 1, 1, 1, 1], padding="VALID")
)

myinput = tf.constant(0.69, tf.float64, [2, 2])
tests.append(tf.sigmoid(myinput))

sess_tf = tf.Session()
sess_dace = TFSession()

for test in tests:
    output_tf = sess_tf.run(test)
    output_dace = sess_dace.run(test)
    print(output_dace)
    print(output_tf)
    assert (output_dace == output_tf).all()
