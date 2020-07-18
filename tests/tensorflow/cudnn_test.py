try:
    import tensorflow as tf
except ImportError:
    print("WARNING: Tensorflow not found, skipping test")
    exit(0)

from tensorflow.python.ops import gen_nn_ops
import numpy as np
import sys
from collections import namedtuple
from dace.frontend.tensorflow import TFSession
import time
from tensorflow.contrib.compiler import xla
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

ConvBench = namedtuple('ConvBench', ['w','h','c','n','k','s','r','pad_w','pad_h','wstride','hstride'])
conv_training_set = [
    ConvBench(700, 161, 1, 4, 32, 20, 5, 0, 0, 2, 2),
    ConvBench(700, 161, 1, 8, 32, 20, 5, 0, 0, 2, 2),
    ConvBench(700, 161, 1, 16, 32, 20, 5, 0, 0, 2, 2),
    ConvBench(700, 161, 1, 32, 32, 20, 5, 0, 0, 2, 2),
    ConvBench(341, 79, 32, 4, 32, 10, 5, 0, 0, 2, 2),
    ConvBench(341, 79, 32, 8, 32, 10, 5, 0, 0, 2, 2),
    ConvBench(341, 79, 32, 16, 32, 10, 5, 0, 0, 2, 2),
    ConvBench(341, 79, 32, 32, 32, 10, 5, 0, 0, 2, 2),
    ConvBench(480, 48, 1, 16, 16, 3, 3, 1, 1, 1, 1),
    ConvBench(240, 24, 16, 16, 32, 3, 3, 1, 1, 1, 1),
    ConvBench(120, 12, 32, 16, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(60, 6, 64, 16, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(108, 108, 3, 8, 64, 3, 3, 1, 1, 2, 2),
    ConvBench(54, 54, 64, 8, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(27, 27, 128, 8, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(14, 14, 128, 8, 256, 3, 3, 1, 1, 1, 1),
    ConvBench(7, 7, 256, 8, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(224, 224, 3, 8, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(112, 112, 64, 8, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(56, 56, 128, 8, 256, 3, 3, 1, 1, 1, 1),
    ConvBench(28, 28, 256, 8, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(14, 14, 512, 8, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(7, 7, 512, 8, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(224, 224, 3, 16, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(112, 112, 64, 16, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(56, 56, 128, 16, 256, 3, 3, 1, 1, 1, 1),
    ConvBench(28, 28, 256, 16, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(14, 14, 512, 16, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(7, 7, 512, 16, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(224, 224, 3, 16, 64, 7, 7, 3, 3, 2, 2),
    ConvBench(28, 28, 192, 16, 32, 5, 5, 2, 2, 1, 1),
    ConvBench(28, 28, 192, 16, 64, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 512, 16, 48, 5, 5, 2, 2, 1, 1),
    ConvBench(14, 14, 512, 16, 192, 1, 1, 0, 0, 1, 1),
    ConvBench(7, 7, 832, 16, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(7, 7, 832, 16, 128, 5, 5, 2, 2, 1, 1),
    ConvBench(56, 56, 64, 8, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(56, 56, 64, 8, 256, 1, 1, 0, 0, 2, 2),
    ConvBench(28, 28, 128, 8, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(28, 28, 128, 8, 512, 1, 1, 0, 0, 2, 2),
    ConvBench(14, 14, 256, 8, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 256, 8, 256, 3, 3, 1, 1, 1, 1),
    ConvBench(14, 14, 256, 8, 1024, 1, 1, 0, 0, 2, 2),
    ConvBench(7, 7, 512, 8, 512, 1, 1, 0, 0, 1, 1),
    ConvBench(7, 7, 2048, 8, 512, 1, 1, 3, 3, 2, 2),
    ConvBench(56, 56, 64, 16, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(56, 56, 64, 16, 256, 1, 1, 0, 0, 2, 2),
    ConvBench(28, 28, 128, 16, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(28, 28, 128, 16, 512, 1, 1, 0, 0, 2, 2),
    ConvBench(14, 14, 256, 16, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 256, 16, 256, 3, 3, 1, 1, 1, 1),
    ConvBench(14, 14, 256, 16, 1024, 1, 1, 0, 0, 2, 2),
    ConvBench(7, 7, 512, 16, 512, 1, 1, 0, 0, 1, 1),
    ConvBench(7, 7, 2048, 16, 512, 1, 1, 3, 3, 2, 2),
    ConvBench(700, 161, 1, 16, 64, 5, 5, 1, 1, 2, 2),
    ConvBench(350, 80, 64, 16, 64, 3, 3, 1, 1, 1, 1),
    ConvBench(350, 80, 64, 16, 128, 5, 5, 1, 1, 2, 2),
    ConvBench(175, 40, 128, 16, 128, 3, 3, 1, 1, 1, 1),
    ConvBench(175, 40, 128, 16, 256, 5, 5, 1, 1, 2, 2),
    ConvBench(84, 20, 256, 16, 256, 3, 3, 1, 1, 1, 1),
    ConvBench(84, 20, 256, 16, 512, 5, 5, 1, 1, 2, 2),
    ConvBench(42, 10, 512, 16, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(112, 112, 64, 8, 64, 1, 1, 0, 0, 1, 1),
    ConvBench(56, 56, 64, 8, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(56, 56, 256, 8, 64, 1, 1, 0, 0, 1, 1),
    ConvBench(56, 56, 256, 8, 128, 1, 1, 0, 0, 2, 2),
    ConvBench(28, 28, 128, 8, 512, 1, 1, 0, 0, 1, 1),
    ConvBench(28, 28, 512, 8, 128, 1, 1, 0, 0, 1, 1),
    ConvBench(28, 28, 512, 8, 256, 1, 1, 0, 0, 2, 2),
    ConvBench(14, 14, 256, 8, 1024, 1, 1, 0, 0, 1, 1),
    ConvBench(28, 28, 512, 8, 1024, 1, 1, 0, 0, 2, 2),
    ConvBench(14, 14, 1024, 8, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 256, 8, 1024, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 1024, 8, 512, 1, 1, 0, 0, 2, 2),
    ConvBench(7, 7, 512, 8, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(7, 7, 512, 8, 2048, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 1024, 8, 2048, 1, 1, 0, 0, 2, 2),
    ConvBench(7, 7, 2048, 8, 512, 1, 1, 0, 0, 1, 1),
    ConvBench(112, 112, 64, 16, 64, 1, 1, 0, 0, 1, 1),
    ConvBench(56, 56, 64, 16, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(56, 56, 256, 16, 64, 1, 1, 0, 0, 1, 1),
    ConvBench(56, 56, 256, 16, 128, 1, 1, 0, 0, 2, 2),
    ConvBench(28, 28, 128, 16, 512, 1, 1, 0, 0, 1, 1),
    ConvBench(28, 28, 512, 16, 128, 1, 1, 0, 0, 1, 1),
    ConvBench(28, 28, 512, 16, 256, 1, 1, 0, 0, 2, 2),
    ConvBench(14, 14, 256, 16, 1024, 1, 1, 0, 0, 1, 1),
    ConvBench(28, 28, 512, 16, 1024, 1, 1, 0, 0, 2, 2),
    ConvBench(14, 14, 1024, 16, 256, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 256, 16, 1024, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 1024, 16, 512, 1, 1, 0, 0, 2, 2),
    ConvBench(7, 7, 512, 16, 512, 3, 3, 1, 1, 1, 1),
    ConvBench(7, 7, 512, 16, 2048, 1, 1, 0, 0, 1, 1),
    ConvBench(14, 14, 1024, 16, 2048, 1, 1, 0, 0, 2, 2),
    ConvBench(7, 7, 2048, 16, 512, 1, 1, 0, 0, 1, 1)
]


if __name__ == '__main__':

    # Conv2d
    rows_list = []
    for num in [1,2,3,4]:
        sample = conv_training_set[num*10]
        input_shape = [sample.n, sample.h, sample.w, sample.c]
        input = tf.placeholder(tf.float64, input_shape)
        filter_shape = [sample.r, sample.s, sample.c, sample.k]
        filter = tf.placeholder(tf.float64, filter_shape)
        output = tf.nn.conv2d(input, filter, [1, sample.hstride, sample.wstride, 1],
                              padding=[(0, 0), (0, sample.pad_h), (0, sample.pad_w), (0, 0)], data_format="NHWC")

        test_input = np.random.uniform(size=tuple(input_shape)).astype(np.float64)
        test_filter = np.random.uniform(size=tuple(filter_shape)).astype(np.float64)

        sess_dace = TFSession()
        sess_tf = tf.Session()

        sess_run = sess_dace.compile(output, gpu=True, cudnn=True)
        start = time.time()
        times = [0.0] * 10
        for i in range(10):
            if i == 0:
                sess_run(feed_dict={input: test_input, filter: test_filter})
            else:
                times[i] = time.time()
                sess_run(feed_dict={input: test_input, filter: test_filter})
                times[i] = time.time() - times[i]
                dict1 = {"values": times[i], "version": "cudnn", "sample": num}
                rows_list.append(dict1)

        start = time.time()
        times = [0.0] * 10
        for i in range(10):
            if i == 0:
                sess_tf.run(output, feed_dict={input: test_input, filter: test_filter})
            else:
                times[i] = time.time()
                sess_tf.run(output, feed_dict={input: test_input, filter: test_filter})
                times[i] = time.time() - times[i]
                dict1 = {"values": times[i], "version": "tf", "sample": num}
                rows_list.append(dict1)

    panda_set_1 = pd.DataFrame(rows_list)
    ax = sns.barplot(x="sample", y="values", hue='version', data=panda_set_1)
    plt.show()
    ax.figure.savefig("convolution.png")

    # Conv2d Backprop Input
    rows_list = []
    for num in [1,2,3,4]:
        sample = conv_training_set[num*10]
        strides = [1, sample.hstride, sample.wstride, 1]
        padding = [(0, 0), (0, sample.pad_h), (0, sample.pad_w), (0, 0)]
        padding_backprop = [0, 0, 0, sample.pad_h, 0, sample.pad_w, 0, 0]
        input_shape = [sample.n, sample.h, sample.w, sample.c]
        input = tf.placeholder(tf.float64, input_shape)
        filter_shape = [sample.r, sample.s, sample.c, sample.k]
        filter = tf.placeholder(tf.float64, filter_shape)
        output = tf.nn.conv2d(input, filter, strides=strides, padding=padding, data_format="NHWC")

        output_backprop = tf.placeholder(tf.float64, output.shape)
        input_gradients = gen_nn_ops.conv2d_backprop_input(input_shape,filter, output_backprop, strides=strides
                                                           ,padding='EXPLICIT', explicit_paddings=padding_backprop)

        test_grads = np.random.uniform(size=output.shape).astype(np.float64)
        test_filter = np.random.uniform(size=tuple(filter_shape)).astype(np.float64)

        sess_dace = TFSession()
        sess_tf = tf.Session()

        sess_run = sess_dace.compile(input_gradients, gpu=True, cudnn=True)
        start = time.time()
        times = [0.0] * 10
        for i in range(10):
            if i == 0:
                sess_run(feed_dict={output_backprop: test_grads, filter: test_filter})
            else:
                times[i] = time.time()
                sess_run(feed_dict={output_backprop: test_grads, filter: test_filter})
                times[i] = time.time() - times[i]
                dict1 = {"values": times[i], "version": "cudnn", "sample": num}
                rows_list.append(dict1)

        start = time.time()
        times = [0.0] * 10
        for i in range(10):
            if i == 0:
                sess_tf.run(input_gradients, feed_dict={output_backprop: test_grads, filter: test_filter})
            else:
                times[i] = time.time()
                sess_tf.run(input_gradients, feed_dict={output_backprop: test_grads, filter: test_filter})
                times[i] = time.time() - times[i]
                dict1 = {"values": times[i], "version": "tf", "sample": num}
                rows_list.append(dict1)

    panda_set_1 = pd.DataFrame(rows_list)
    ax = sns.barplot(x="sample", y="values", hue='version', data=panda_set_1)
    plt.show()
    ax.figure.savefig("backpropimg.png")

    # Conv2d Backprop Filter
    rows_list = []
    for num in [1,2,3,4]:
        sample = conv_training_set[num*10]
        strides = [1, sample.hstride, sample.wstride, 1]
        padding = [(0, 0), (0, sample.pad_h), (0, sample.pad_w), (0, 0)]
        padding_backprop = [0, 0, 0, sample.pad_h, 0, sample.pad_w, 0, 0]
        input_shape = [sample.n, sample.h, sample.w, sample.c]
        input = tf.placeholder(tf.float64, input_shape)
        filter_shape = [sample.r, sample.s, sample.c, sample.k]
        filter = tf.placeholder(tf.float64, filter_shape)
        output = tf.nn.conv2d(input, filter, strides=strides, padding=padding, data_format="NHWC")

        output_backprop = tf.placeholder(tf.float64, output.shape)
        filter_gradients = gen_nn_ops.conv2d_backprop_filter(input, filter_shape, output_backprop, strides=strides
                                                           , padding='EXPLICIT', explicit_paddings=padding_backprop)

        test_grads = np.random.uniform(size=output.shape).astype(np.float64)
        test_input = np.random.uniform(size=tuple(input_shape)).astype(np.float64)

        sess_dace = TFSession()
        sess_tf = tf.Session()

        sess_run = sess_dace.compile(filter_gradients, gpu=True, cudnn=True)
        start = time.time()
        times = [0.0] * 10
        for i in range(10):
            if i == 0:
                sess_run(feed_dict={input: test_input, output_backprop: test_grads})
            else:
                times[i] = time.time()
                sess_run(feed_dict={input: test_input, output_backprop: test_grads})
                times[i] = time.time() - times[i]
                dict1 = {"values": times[i], "version": "cudnn", "sample": num}
                rows_list.append(dict1)


        start = time.time()
        times = [0.0] * 10
        for i in range(10):
            if i == 0:
                sess_tf.run(filter_gradients, feed_dict={input: test_input, output_backprop: test_grads})
            else:
                times[i] = time.time()
                sess_tf.run(filter_gradients, feed_dict={input: test_input, output_backprop: test_grads})
                times[i] = time.time() - times[i]
                dict1 = {"values": times[i], "version": "tf", "sample": num}
                rows_list.append(dict1)

    panda_set_1 = pd.DataFrame(rows_list)
    ax = sns.barplot(x="sample", y="values", hue='version', data=panda_set_1)
    plt.show()
    ax.figure.savefig("backpropfltr.png")