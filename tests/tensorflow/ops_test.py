# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np


@pytest.mark.tensorflow
def test_shapen():
    import tensorflow as tf
    from dace.frontend.tensorflow import TFSession
    myshape = [69, 96, 666]
    num_inputs = 5

    inpList = [tf.ones(myshape) for _ in range(num_inputs)]

    sess_tf = tf.Session()
    sess_dace = TFSession()

    shapes_tf = sess_tf.run(tf.shape_n(inpList))
    shapes_dace = sess_dace.run(tf.shape_n(inpList))
    for dc, tf in zip(shapes_dace, shapes_tf):
        try:
            assert (dc == tf).all()
        except (AssertionError):
            print(dc)
            print(tf)


@pytest.mark.tensorflow
def test_mean():
    import tensorflow as tf
    from dace.frontend.tensorflow import TFSession
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
            assert tf.norm(output_dace - output_tf).eval(session=sess_tf) < 1e-10
        except:
            print(output_dace)
            print(output_tf)
            print(tf.norm(output_dace - output_tf).eval(session=sess_tf))
            raise AssertionError("mean test {i} failed".format(i=index))

    print("mean tests passed!")


@pytest.mark.tensorflow
def test_addn():
    import tensorflow as tf
    from dace.frontend.tensorflow import TFSession
    shape = [10, 11, 12, 13]
    inputs = [np.random.rand(*shape) for _ in range(10)]
    addn_test_0 = tf.add_n(inputs)

    sess_tf = tf.Session()
    sess_dace = TFSession()

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


@pytest.mark.tensorflow
def test_slice():
    import tensorflow as tf
    from dace.frontend.tensorflow import TFSession
    t = tf.placeholder(tf.int32, [3, 2, 3])
    b = tf.placeholder(tf.int32, [3])
    s = tf.placeholder(tf.int32, [3])
    output = tf.placeholder(tf.int32, [1, 1, 3])
    output = tf.slice(t, b, s)
    input_tensor = tf.constant([[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]]])

    sess_tf = tf.Session()
    sess_dace = TFSession()

    begin_tensor = tf.constant([1, 0, 0])
    size_tensor_1 = tf.constant([1, 2, 2])
    size_tensor_2 = tf.constant([1, 2, 3])
    size_tensor_3 = tf.constant([2, 1, 3])
    tf_out = sess_tf.run(tf.slice(input_tensor, begin_tensor, size_tensor_3))
    dace_out = sess_dace.run(tf.slice(input_tensor, begin_tensor, size_tensor_3))
    print(tf_out)
    print(dace_out)
    assert (tf_out == dace_out).all()


if __name__ == '__main__':
    try:
        import tensorflow
        test_shapen()
        test_mean()
        test_addn()
        test_slice()
    except ImportError:
        pass
