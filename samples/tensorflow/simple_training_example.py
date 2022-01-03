# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Author: Roman Haag
import tensorflow as tf
from dace.frontend import tensorflow as dacetf
import numpy as np

SEED = 12983172
ITER = 5

# Small network with one dense layer, having a random standard normal kernel, no bias layer and no
# activation function, softmax-cross-entropy loss.
# Using gradient descent with constant learning rate to optimize.

if __name__ == "__main__":

    # Set up the network
    image_node = tf.placeholder(dtype=tf.float32, shape=(20, 30))
    label_node = tf.placeholder(tf.int32, shape=(20))
    linear_layer = fullConnectionTf1 = tf.layers.dense(image_node,
                                                       units=10,
                                                       activation=None,
                                                       use_bias=False,
                                                       kernel_initializer=tf.random_normal_initializer(seed=SEED))
    softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_node, logits=linear_layer)
    loss = tf.reduce_mean(softmax, name="loss")

    # Set up the optimizer
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=tf.constant(0.01)).minimize(loss)

    # Randomize inputs
    image = np.random.rand(5, 20, 30).astype(np.float32)
    label = np.random.randint(low=0, high=10, size=(5, 20), dtype=np.int32)

    # Create initializer
    trainable_variables = tf.trainable_variables()
    init = tf.global_variables_initializer()

    # Run with DaCe
    with dacetf.TFSession(seed=SEED) as sess:
        # TFSession.train is a special mode where one SDFG runs for the entire
        # training process.
        dace_variables, _ = sess.train(optimizer, init, ITER, {
            image_node: image,
            label_node: label
        }, trainable_variables)

    # Run with Tensorflow (for correctness)
    with tf.Session() as sess:
        sess.run(init)
        for i in range(ITER):
            sess.run(optimizer, {image_node: image[i], label_node: label[i]})
        tf_variables = sess.run(trainable_variables)

    # Correctness check
    for tfvar, dacevar in zip(tf_variables, dace_variables.values()):
        frob_norm = np.linalg.norm((tfvar - dacevar).flatten(), ord=1)
        inf_norm = np.linalg.norm((tfvar - dacevar).flatten(), ord=np.inf)
        print("Abs. Diff:", frob_norm)
        print("Max. Diff:", inf_norm)
        if (frob_norm < 1e-4 and inf_norm < 1e-5):
            continue
        else:
            print("==== Program end ====")
            exit(1)
        print("==== Program end ====")
        exit(0)
