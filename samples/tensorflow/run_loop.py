# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from dace.frontend.tensorflow import TFSession
import numpy as np
import tensorflow as tf
from tensorflow.contrib.compiler import xla
import time
import resnet50
from resnet50 import SEED

learning_rate = 0.1
batch_size = 128
num_classes = 10


def random_batch(batch_size):
    shape = (batch_size, 224, 224, 3)
    images = np.random.uniform(size=shape).astype(np.float32)
    labels = np.random.randint(low=0, high=num_classes, size=(batch_size)).astype(np.int32)
    # print(labels.shape)
    return images, labels


input_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, 224, 224, 3))
label_placeholder = tf.placeholder(dtype=tf.int32, shape=(batch_size))


def build_resnet(images, labels):
    # Graph building
    myresnet = resnet50.ResNet50("channels_last", classes=num_classes)  # trainable=False)
    logits = myresnet(images)
    softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.reduce_mean(softmax, name="loss")
    gradients = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(loss)
    gradient_tensors = []
    for tup in gradients:
        gradient_tensors.append(tup[0])
    update_op = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(gradients)

    return logits, update_op


# DaCe
sess = TFSession(seed=SEED)
y = build_resnet(input_placeholder, label_placeholder)

# TensorFlow + XLA
#sess = tf.Session()
#[y] = xla.compile(build_resnet, inputs=[input_placeholder, label_placeholder])

init = tf.global_variables_initializer()
sess.run(init)

images, labels = random_batch(batch_size)

# Warmup run
sess_run = sess.compile(y, gpu=True)  # Change to gpu=True to run on the GPU

start = time.time()
times = [0.0] * 100
for i in range(100):
    times[i] = time.time()
    sess_run(feed_dict={input_placeholder: images, label_placeholder: labels})
    times[i] = time.time() - times[i]

tarr = np.array(times)
print('Median time:', np.median(tarr))
