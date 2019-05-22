import tensorflow as tf
import dace
import numpy as np
from dace.frontend.tensorflow import TFSession

size_in = [1, 112, 112, 3]
# size_in = [4, 4, 4, 4]
input_tensor = np.random.uniform(size=size_in).astype(np.float64)
input_placeholder = tf.placeholder(tf.float64, size_in)
ksize = [1, 3, 3, 1]
stride = [1, 2, 2, 1]
# need to fix bug in padding SAME
max_pool_outp = tf.nn.max_pool(input_placeholder, ksize, stride, "VALID")
avg_pool_outp = tf.nn.avg_pool(input_placeholder, ksize, stride, "VALID")
sess_tf = tf.Session()
sess_dace = TFSession()
# MAX pool test
tf_output = sess_tf.run(max_pool_outp, feed_dict={input_placeholder: input_tensor})
dace_output = sess_dace.run(max_pool_outp, feed_dict={input_placeholder: input_tensor})
try:
    assert tf.norm(dace_output - tf_output).eval(session=sess_tf) < 1e-10
except:
    print(dace_output.shape)
    print(tf_output.shape)
    print(tf.norm(dace_output - tf_output).eval(session=sess_tf))
    raise AssertionError("max pool test failed")
# AVG pool test
tf_output = sess_tf.run(avg_pool_outp, feed_dict={input_placeholder: input_tensor})
dace_output = sess_dace.run(avg_pool_outp, feed_dict={input_placeholder: input_tensor})
try:
    assert tf.norm(dace_output - tf_output).eval(session=sess_tf) < 1e-10
except:
    print(dace_output)
    print(tf_output)
    print(tf.norm(dace_output - tf_output).eval(session=sess_tf))
    raise AssertionError("avg pool test failed")

# AVG pool gradient test
loss_placeholder = tf.placeholder(tf.float64, avg_pool_outp.shape)
loss_tensor = np.random.uniform(size=avg_pool_outp.shape)
grads_avg = tf.gradients(avg_pool_outp, input_placeholder, grad_ys=loss_placeholder)
dace_output = sess_dace.run(grads_avg, feed_dict={loss_placeholder: loss_tensor})
tf_output = sess_tf.run(grads_avg, feed_dict={loss_placeholder: loss_tensor})
try:
    assert tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf) < 1e-10
except:
    print(dace_output)
    print(tf_output)
    print(tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf))
    raise AssertionError("avg pool gradient test failed")

# Max pool gradient test
loss_placeholder = tf.placeholder(tf.float64, max_pool_outp.shape)
loss_tensor = np.random.uniform(size=max_pool_outp.shape)
grads_max = tf.gradients(max_pool_outp, input_placeholder, grad_ys=loss_placeholder)
dace_output = sess_dace.run(
    grads_max,
    feed_dict={input_placeholder: input_tensor, loss_placeholder: loss_tensor},
)
tf_output = sess_tf.run(
    grads_max,
    feed_dict={input_placeholder: input_tensor, loss_placeholder: loss_tensor},
)
try:
    assert tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf) < 1e-10
except:
    print(dace_output)
    print(tf_output)
    print(tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf))
    raise AssertionError("max pool gradient test failed")
