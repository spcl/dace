# Copyright 2019-2020 ETH Zurich and the DaCe authors. All rights reserved.
try:
    import tensorflow as tf
except ImportError:
    print("WARNING: Tensorflow not found, skipping test")
    exit(0)

from dace.frontend.tensorflow import TFSession

if __name__ == '__main__':
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
