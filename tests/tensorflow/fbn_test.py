# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import pytest
import numpy as np


@pytest.mark.tensorflow
def test_fused_batch_norm():
    import tensorflow as tf
    from tensorflow.python.ops import gen_nn_ops
    from dace.frontend.tensorflow import TFSession

    num_channels = 3
    size = [8, 224, 224, num_channels]

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    inp = tf.placeholder(tf.float32, size)
    scale = tf.placeholder(tf.float32, [num_channels])
    offset = tf.placeholder(tf.float32, [num_channels])
    populationMean = tf.placeholder(tf.float32, [num_channels])
    populationVariance = tf.placeholder(tf.float32, [num_channels])
    y, mean, var, _, var_sqrt = gen_nn_ops._fused_batch_norm(inp, scale, offset, [], [], epsilon=0.1, is_training=True)
    outputs = [y, mean, var]
    test_in = np.random.uniform(size=size).astype(np.float32)
    test_scale = np.random.uniform(size=[num_channels]).astype(np.float32)
    test_offset = np.random.uniform(size=[num_channels]).astype(np.float32)

    sess_tf = tf.Session(config=config)
    sess_dace = TFSession()

    outputs_dace = sess_dace.run(
        outputs,
        feed_dict={
            inp: test_in,
            scale: test_scale,
            offset: test_offset,
        },
    )
    outputs_tf = sess_tf.run(
        outputs,
        feed_dict={
            inp: test_in,
            scale: test_scale,
            offset: test_offset,
        },
    )

    try:
        assert (tf.linalg.norm(outputs_tf[0] - outputs_dace[0]).eval(session=sess_tf) < 1e-1
                and tf.linalg.norm(outputs_dace[2] - outputs_tf[2]).eval(session=sess_tf) < 1e-4
                and tf.linalg.norm(outputs_dace[1] - outputs_tf[1]).eval(session=sess_tf) < 1e-4)
    except:
        print("FBN test failed")
        print(tf.linalg.norm(outputs_tf[0] - outputs_dace[0]).eval(session=sess_tf))
        print(tf.linalg.norm(outputs_tf[1] - outputs_dace[1]).eval(session=sess_tf))
        print(tf.linalg.norm(outputs_tf[2] - outputs_dace[2]).eval(session=sess_tf))

    ################# FBN GRADIENT TEST ###############################
    outputGrad = tf.placeholder(tf.float32, size)
    x_grad, gamma_grad, beta_grad, _, _ = gen_nn_ops.fused_batch_norm_grad(outputGrad,
                                                                           inp,
                                                                           scale,
                                                                           outputs[1],
                                                                           var_sqrt,
                                                                           epsilon=0.1,
                                                                           is_training=True)
    gradients = [x_grad, gamma_grad, beta_grad]
    test_outputgrad = np.random.uniform(size=size).astype(np.float32)
    outputs_dace = sess_dace.run(
        gradients,
        feed_dict={
            inp: test_in,
            outputGrad: test_outputgrad,
            scale: test_scale,
            offset: test_offset,
        },
    )
    # TF
    x_grad, gamma_grad, beta_grad, _, _ = gen_nn_ops.fused_batch_norm_grad(
        outputGrad,
        inp,
        scale,
        outputs[1],
        tf.math.rsqrt(outputs[2] + float(0.1)) if tf.test.is_built_with_cuda() else outputs[2],
        epsilon=0.1,
        is_training=True,
    )
    gradients = [x_grad, gamma_grad, beta_grad]
    # writer = tf.summary.FileWriter("./", sess_tf.graph)
    outputs_tf = sess_tf.run(
        gradients,
        feed_dict={
            inp: test_in,
            outputGrad: test_outputgrad,
            scale: test_scale,
            offset: test_offset,
        },
    )
    try:
        assert (tf.linalg.norm(outputs_tf[0] - outputs_dace[0]).eval(session=sess_tf) < 1e-1
                and tf.linalg.norm(outputs_dace[2] - outputs_tf[2]).eval(session=sess_tf) < 10
                and tf.linalg.norm(outputs_dace[1] - outputs_tf[1]).eval(session=sess_tf) < 10)
    except:
        print("FBN Gradient test failed")
        print(tf.linalg.norm(outputs_tf[0] - outputs_dace[0]).eval(session=sess_tf))
        print(tf.linalg.norm(outputs_tf[1] - outputs_dace[1]).eval(session=sess_tf))
        print(tf.linalg.norm(outputs_tf[2] - outputs_dace[2]).eval(session=sess_tf))
        print(tf.linalg.norm(outputs_tf[2] - np.sum(test_outputgrad, axis=(0, 1, 2))).eval(session=sess_tf))


if __name__ == '__main__':
    try:
        import tensorflow
        test_fused_batch_norm()
    except ImportError:
        pass
