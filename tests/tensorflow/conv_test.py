try:
    import tensorflow as tf
except ImportError:
    print("WARNING: Tensorflow not found, skipping test")
    exit(0)

from tensorflow.python.ops import gen_nn_ops
import numpy as np
import dace
from dace.frontend.tensorflow import TFSession
K = 2
C = 2
R = 5
S = 5
inp_shape = [2, 8, 8, 2]
filters = [[R, S, C, K]]
strides = [[1, 1, 1, 1]]
dilations = [[1, 1, 1, 1]]
paddings = ["VALID"]
for p in paddings:
    for f in filters:
        for s in strides:
            for d in dilations:
                print(p, f, s, d)
                inp = tf.placeholder(tf.float32, inp_shape)
                filter = tf.placeholder(tf.float32, f)
                outp = tf.nn.conv2d(inp, filter, s, padding=p, data_format="NHWC", dilations=dilations[0])

                #test_in = np.array([[[[1],[0],[0],[0],[1]],
                #            [[0],[1],[0],[0],[0]],
                #            [[0],[0],[1],[0],[0]],
                #            [[0],[0],[0],[1],[0]],
                #            [[0],[0],[0],[0],[1]]]]).astype(np.float32)
                #test_in = np.full(shape=inp_shape, fill_value=2, dtype=np.float32)
                #test_filter = np.full(shape=f, fill_value=np.random.uniform(), dtype=np.float32)
                #test_filter = np.array([[[[0]]],
                #                       [[[1]]]]).astype(np.float32)
                #test_filter3 = np.array([[[[1],[1]],
                #                          [[-1],[1]]]]).astype(np.float32)                     
                #print(test_filter.shape)
                #test_filter3 = np.transpose(test_filter, [3, 2, 0, 1])[:]
                test_filter = np.random.uniform(size=tuple(f)).astype(np.float32)
                test_in = np.random.uniform(size=tuple(inp_shape)).astype(np.float32)

                
                config = tf.ConfigProto(device_count={'GPU':0})
                config.gpu_options.allow_growth = True

                sess_dace = TFSession()
                sess_tf = tf.Session(config=config)


                output_dace = sess_dace.run(
                    outp, feed_dict={
                        inp: test_in,
                        filter: test_filter[:]
                    }, gpu=True)

                output_tf = sess_tf.run(outp, feed_dict={inp: test_in, filter: test_filter})
                
                try:
                    assert tf.norm(output_dace - output_tf).eval(session=sess_tf) < 1e-2
                    print('\nConvolution test passed\n')
                except:
                    print('filter: ', test_filter)
                    #print('filter3: ', test_filter)
                    print('tf ', output_tf)
                    print('dace ', output_dace)
                    print(tf.linalg.norm(output_tf - output_dace).eval(session=sess_tf))
                    print(output_tf - output_dace)
                    raise AssertionError("Convolution test failed")

##### Conv backprop grad ######
#inp_shape = [10, 10, 10, 10]
#filters = [[2, 2, 10, 3]]
#strides = [[1, 1, 1, 1]]
#paddings = ["VALID"]
for p in paddings:
    for f in filters:
        for s in strides:
            for d in dilations:
                print(p, f, s, d)
                filter = tf.placeholder(tf.float32, f)
                outp = tf.nn.conv2d(inp, filter, s, padding=p, data_format="NHWC")
                out_backprop = tf.placeholder(tf.float32, outp.shape)
                inp_gradients = gen_nn_ops.conv2d_backprop_input(
                    inp_shape, filter, out_backprop, s, padding=p)
                test_grads = np.random.uniform(size=outp.shape).astype(np.float32)
                test_filter = np.random.uniform(size=tuple(f)).astype(np.float32)

                output_tf = sess_tf.run(
                    inp_gradients,
                    feed_dict={
                        filter: test_filter,
                        out_backprop: test_grads
                    })
                output_dace = sess_dace.run(
                    inp_gradients,
                    feed_dict={
                        filter: test_filter,
                        out_backprop: test_grads
                    }, gpu=True)

                try:
                    assert tf.norm(output_dace -
                                   output_tf).eval(session=sess_tf) < 1e-2
                    print('Convolution grad test passed\n')
                except:
                    print(output_tf)
                    print(output_dace)
                    print(
                        tf.linalg.norm(output_tf -
                                       output_dace).eval(session=sess_tf))
                    print(output_tf - output_dace)
                    raise AssertionError("Convolution grad test failed")

##### Conv filter backprop ##################
#inp_shape = [10, 10, 10, 10]
#filters = [[4, 4, 10, 3]]
#strides = [[1, 1, 1, 1]]
#paddings = ["VALID"]
for p in paddings:
    for f in filters:
        for s in strides:
            for d in dilations:
                print(p, f, s, d)
                input_placeholder = tf.placeholder(tf.float32, inp_shape)
                filter = tf.placeholder(tf.float32, f)
                outp = tf.nn.conv2d(inp, filter, s, padding=p, data_format="NHWC")
                out_backprop = tf.placeholder(tf.float32, outp.shape)
                filter_gradients = gen_nn_ops.conv2d_backprop_filter(
                    input_placeholder, f, out_backprop, s, padding=p)
                test_grads = np.random.uniform(size=outp.shape).astype(np.float32)
                test_input = np.random.uniform(size=tuple(inp_shape)).astype(
                    np.float32)

                output_tf = sess_tf.run(
                    filter_gradients,
                    feed_dict={
                        input_placeholder: test_input,
                        out_backprop: test_grads
                    })
                output_dace = sess_dace.run(
                    filter_gradients,
                    feed_dict={
                        input_placeholder: test_input,
                        out_backprop: test_grads
                    }, gpu=True)

                try:
                    assert tf.norm(output_dace -
                                   output_tf).eval(session=sess_tf) < 1e-2
                    print('Conv filter grad test passed\n')
                except:
                    print(output_tf)
                    print(output_dace)
                    print(
                        tf.linalg.norm(output_tf -
                                       output_dace).eval(session=sess_tf))
                    raise AssertionError("Convolution filter grad test failed")
