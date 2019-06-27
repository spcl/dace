import tensorflow as tf
import resnet_small
import resnet50
import numpy as np
import dace
from dace.frontend.tensorflow import TFSession
from dace.transformation.dataflow import (
    GPUTransformLocalStorage,
    TensorflowRedundantArray,
    MapFusion,
    RedundantArray,
    RedundantArrayCopying,
    RedundantArrayCopying2,
    RedundantArrayCopying3,
)
from resnet_small import SEED

learning_rate = 0.01
batch_size = 128
num_classes = 10

# dace.Config.set(
#    "compiler",
#    "cpu",
#    "args",
#    value=dace.Config.get("compiler", "cpu", "args")
#    .replace("O3", "O0")
#    .replace("-ffast-math", ""),
# )


def random_batch(batch_size):
    shape = (batch_size, 224, 224, 3)
    images = np.random.uniform(size=shape).astype(np.float32)
    labels = np.random.randint(low=0, high=num_classes, size=(batch_size)).astype(
        np.int32
    )
    # print(labels.shape)
    return images, labels


images, labels = random_batch(batch_size)

# Small Graph
#small_resnet = resnet_small.ResNet50_small("channels_last", classes=num_classes)
#input_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, 224, 224, 3))
#label_placeholder = tf.placeholder(dtype=tf.int32, shape=(batch_size))
#logits = small_resnet(input_placeholder)
#softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(
#    labels=label_placeholder, logits=logits
#)
#loss = tf.reduce_mean(softmax, name="loss")

# Large Graph
big_resnet = resnet50.ResNet50("channels_last", classes=num_classes)
input_placeholder_1 = tf.placeholder(dtype=tf.float32, shape=(batch_size, 224, 224, 3))
label_placeholder_1 = tf.placeholder(dtype=tf.int32, shape=(batch_size))
logits_big = big_resnet(input_placeholder_1)
softmax_big = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label_placeholder_1, logits=logits_big
)
loss_big = tf.reduce_mean(softmax_big, name="loss")
update_big = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss_big)

sess_dace = TFSession(seed=SEED)
outputs_dace = sess_dace.run(
    update_big,
    gpu=False,
    feed_dict={input_placeholder_1: images, label_placeholder_1: labels},
    transformations=[
        [TensorflowRedundantArray],
        [GPUTransformLocalStorage],
        [RedundantArray, RedundantArrayCopying],
    ],
)
# gradients = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(loss)
# gradient_tensors = []
# for tup in gradients:
#    gradient_tensors.append(tup[0])
# update_op = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(gradients)
# input_gradients = tf.gradients(loss, input_placeholder)
# sess_tf = tf.Session()
# init = tf.global_variables_initializer()
# gradients_dace = sess_dace.train(
#    [],
#    init,
#    1,
#    {input_placeholder: images, label_placeholder: labels},
#    gradient_tensors,
# )[1]
# updates_dace = sess_dace.train(
#    update_op,
#    [],
#    #init,
#    1,
#    {
#        input_placeholder: images,
#        label_placeholder: labels
#    },
# )[1]
# updates_dace = sess_dace.run(
#    update_op, feed_dict={input_placeholder: images, label_placeholder: labels}
# )
# input_grads_dace = sess_dace.run(
#        input_gradients, feed_dict={input_placeholder: images, label_placeholder: labels}
# )
# wrong_grads = sess_dace.run(
#    gradient_tensors[0],
#    feed_dict={input_placeholder: images, label_placeholder: labels},
# )
# tf.summary.FileWriter("./", sess_tf.graph)
# sess_tf.run(init)
##outputs_tf = sess_tf.run(
##    [logits, softmax, loss],
##    feed_dict={input_placeholder: images, label_placeholder: labels},
##)
# gradients_tf = sess_tf.run(
#    gradient_tensors,
#    feed_dict={input_placeholder: images, label_placeholder: labels},
# )
# for name, tfgrad, dacegrad in zip(update_op, updates_tf, updates_dace):
#    inf_norm = np.linalg.norm((tfgrad - dacegrad).flatten())
#    print(str(name), str(inf_norm))
# print(gradients_dace)
# print(gradients_tf)
# print(tf.linalg.norm(gradients_dace - gradients_tf).eval(session=sess_tf))
# print("gradient was ", str(gradient_tensors[0]))
# print("older one was ", str(gradient_tensors[0]), " new one is ", str(gradient_tensors[1]))

################### FORWARD PASS NORM ###################################
# print(tf.linalg.norm(outputs_dace[0]-outputs_tf[0]).eval(session=sess_tf))
# print(tf.linalg.norm(outputs_dace[1]-outputs_tf[1]).eval(session=sess_tf))
# print(tf.linalg.norm(outputs_dace[2]-outputs_tf[2]).eval(session=sess_tf))
# train_accuracy = np.mean(np.argmax(one_hot_train, axis=1) ==
#                         sess.run(predict, feed_dict={X: x_train, y:
#                                                      one_hot_train}))
# test_accuracy  = np.mean(np.argmax(one_hot_test, axis=1) ==
#                         sess.run(predict, feed_dict={X: x_test, y:
#                                                      one_hot_test}))
# print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
