# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
# Author: Roman Haag
# Adapted from https://github.com/tensorflow/models/blob/master/tutorials/image/mnist/convolutional.py

import tensorflow as tf
import numpy as np
from dace.frontend.tensorflow import TFSession

IMAGE_SIZE = 28
NUM_CHANNELS = 1
NUM_LABELS = 10
SEED = 19323
BATCH_SIZE = 64

if __name__ == "__main__":

    # Create synthetic image and label data
    image = np.ndarray(shape=(BATCH_SIZE, IMAGE_SIZE, IMAGE_SIZE, NUM_CHANNELS), dtype=np.float32)
    labels = np.zeros(shape=(BATCH_SIZE, ), dtype=np.int64)
    for i in range(0, BATCH_SIZE):
        label = i % 2
        image[i, :, :, 0] = label - 0.5
        labels[i] = label
    image_node = tf.convert_to_tensor(image)
    label_node = tf.convert_to_tensor(labels)

    # Set up all variables
    conv1_weights = tf.Variable(tf.random_normal([5, 5, NUM_CHANNELS, 32], stddev=0.1, seed=SEED, dtype=tf.float32))
    conv1_biases = tf.Variable(tf.zeros([32], dtype=tf.float32))
    conv2_weights = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev=0.1, seed=SEED, dtype=tf.float32))
    conv2_biases = tf.Variable(tf.constant(0.1, shape=[64], dtype=tf.float32))

    # LeNet with sofmax cross entropy
    convolutionTf1 = tf.nn.conv2d(input=image_node, filter=conv1_weights, strides=[1, 1, 1, 1], padding='VALID')
    biasTf1 = tf.nn.bias_add(convolutionTf1, conv1_biases)
    poolTf1 = tf.nn.max_pool(biasTf1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    reluTf1 = tf.nn.relu(poolTf1)
    convolutionTf2 = tf.nn.conv2d(input=reluTf1, filter=conv2_weights, strides=[1, 1, 1, 1], padding='VALID')
    biasTf2 = tf.nn.bias_add(convolutionTf2, conv2_biases)
    poolTf2 = tf.nn.max_pool(biasTf2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    reluTf2 = tf.nn.relu(poolTf2)
    flattenTf = tf.layers.flatten(reluTf2)
    fullConnectionTf1 = tf.layers.dense(flattenTf,
                                        500,
                                        activation=tf.nn.relu,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-0.1,
                                                                                         maxval=0.1,
                                                                                         seed=SEED))
    fullConnectionTf2 = tf.layers.dense(fullConnectionTf1,
                                        10,
                                        activation=None,
                                        kernel_initializer=tf.random_uniform_initializer(minval=-0.1,
                                                                                         maxval=0.1,
                                                                                         seed=SEED))

    logits = fullConnectionTf2
    softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=label_node, logits=logits)
    loss = tf.reduce_mean(softmax, name="loss")

    # Get gradient tensors and initializer operation
    gradients = tf.gradients(loss, tf.trainable_variables())
    init = tf.global_variables_initializer()

    # Compute gradients and compare
    # Tensorflow
    with tf.Session() as sess:
        sess.run(init)
        tf_gradients = sess.run(gradients)

    # DaCe
    with TFSession(seed=SEED) as sess:
        sess.run(init)
        dace_gradients = sess.run(gradients)

    # Compare
    for tfgrad, dacegrad in zip(tf_gradients, dace_gradients):
        inf_norm = np.linalg.norm((tfgrad - dacegrad).flatten())
        print("Max. Diff:", inf_norm)
        if (inf_norm <= 1e-4):
            continue
        else:
            print("==== Program end====")
            print("Error: norm too large")
            exit(1)
    print("==== Program end ====")
    exit(0)
