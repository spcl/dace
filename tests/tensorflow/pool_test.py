# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np


@pytest.mark.tensorflow
def test_pooling():
    import tensorflow as tf
    from dace.frontend.tensorflow import TFSession
    size_in = [1, 112, 112, 3]
    # size_in = [4, 4, 4, 4]
    np.random.seed(0)
    input_tensor = np.random.uniform(size=size_in).astype(np.float32)
    input_placeholder = tf.placeholder(tf.float32, size_in)
    ksize = [1, 3, 3, 1]
    stride = [1, 2, 2, 1]
    # need to fix bug in padding SAME
    max_pool_outp = tf.nn.max_pool(input_placeholder, ksize, stride, "VALID", data_format="NHWC")
    avg_pool_outp = tf.nn.avg_pool(input_placeholder, ksize, stride, "VALID", data_format="NHWC")
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
    print("Max pool test passed")

    # AVG pool test
    tf_output = sess_tf.run(avg_pool_outp, feed_dict={input_placeholder: input_tensor})
    dace_output = sess_dace.run(avg_pool_outp, feed_dict={input_placeholder: input_tensor})
    try:
        assert tf.norm(dace_output - tf_output).eval(session=sess_tf) < 1e-5
    except:
        print(dace_output.shape)
        print(tf_output.shape)
        print(tf.norm(dace_output - tf_output).eval(session=sess_tf))
        raise AssertionError("avg pool test failed")
    print("Average pool test passed")

    # AVG pool gradient test
    np.random.seed(0)
    loss_placeholder = tf.placeholder(tf.float32, avg_pool_outp.shape)
    loss_tensor = np.random.uniform(size=avg_pool_outp.shape).astype(np.float32)
    grads_avg = tf.gradients(avg_pool_outp, input_placeholder, grad_ys=loss_placeholder)
    dace_output = sess_dace.run(grads_avg, feed_dict={loss_placeholder: loss_tensor})
    tf_output = sess_tf.run(grads_avg, feed_dict={loss_placeholder: loss_tensor})
    try:
        assert tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf) < 1e-5
    except:
        print(dace_output)
        print(tf_output)
        print(tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf))
        raise AssertionError("avg pool gradient test failed")

    # Max pool gradient test
    loss_placeholder = tf.placeholder(tf.float32, max_pool_outp.shape)
    np.random.seed(0)
    loss_tensor = np.random.uniform(size=max_pool_outp.shape).astype(np.float32)
    grads_max = tf.gradients(max_pool_outp, input_placeholder, grad_ys=loss_placeholder)
    dace_output = sess_dace.run(
        grads_max,
        feed_dict={
            input_placeholder: input_tensor,
            loss_placeholder: loss_tensor
        },
    )
    tf_output = sess_tf.run(
        grads_max,
        feed_dict={
            input_placeholder: input_tensor,
            loss_placeholder: loss_tensor
        },
    )
    try:
        assert tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf) < 1e-5
    except:
        print(dace_output)
        print(tf_output)
        print(tf.norm(dace_output[0] - tf_output[0]).eval(session=sess_tf))
        raise AssertionError("max pool gradient test failed")


if __name__ == '__main__':
    try:
        import tensorflow
        test_pooling()
    except ImportError:
        pass
