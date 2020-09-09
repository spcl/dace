# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.

try:
    import tensorflow as tf
except ImportError:
    print("WARNING: Tensorflow not found, skipping test")
    exit(0)

import numpy as np

from dace.frontend.tensorflow import TFSession

if __name__ == '__main__':
    print('DaCe Tensorflow frontend compile API test')

    A = np.random.rand(16, 16).astype(np.float32)
    B = np.random.rand(16, 16).astype(np.float32)

    A_tf = tf.placeholder(tf.float32, shape=[16, 16])
    B_tf = tf.placeholder(tf.float32, shape=[16, 16])

    with TFSession() as sess:
        # Simple matrix multiplication
        func = sess.compile(A_tf @ B_tf, False)
        func(feed_dict={A_tf: A, B_tf: B})
        C = func(feed_dict={A_tf: A, B_tf: B})

    diff = np.linalg.norm(C - (A @ B)) / (16 * 16)
    print("Difference:", diff)
    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
