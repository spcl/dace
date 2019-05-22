import tensorflow as tf
import resnet50
import numpy as np
import dace
from dace.frontend.tensorflow import TFSession
from resnet50 import SEED

learning_rate = 0.01
batch_size = 1
num_classes = 10

#dace.Config.set(
#    "compiler",
#    "cpu",
#    "args",
#    value=dace.Config.get("compiler", "cpu", "args")
#    .replace("O3", "O0")
#    .replace("-ffast-math", ""),
#)


def random_batch(batch_size):
    shape = (batch_size, 224, 224, 3)
    images = np.random.uniform(size=shape).astype(np.float32)
    labels = np.random.randint(low=0, high=num_classes, size=(batch_size)).astype(
        np.int32
    )
    # print(labels.shape)
    return images, labels

tf.disable_v2_behavior()
tf.disable_resource_variables()
tf.compat.v1.disable_eager_execution()
images, labels = random_batch(batch_size)
# Graph building
myresnet = resnet50.ResNet50("channels_last", classes=num_classes)  # trainable=False)
input_placeholder = tf.placeholder(dtype=tf.float32, shape=(batch_size, 224, 224, 3))
label_placeholder = tf.placeholder(dtype=tf.int32, shape=(batch_size))
logits = myresnet(input_placeholder)
softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label_placeholder, logits=logits
)
loss = tf.reduce_mean(softmax, name="loss")
gradients = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(loss)
gradient_tensors = []
for tup in gradients:
    gradient_tensors.append(tup[0])
update_op = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(gradients)
input_gradients = tf.gradients(loss, input_placeholder)
sess_dace = TFSession(seed=SEED)
sess_tf = tf.Session()
init = tf.global_variables_initializer()
gradients_dace = sess_dace.train(
    [],
    init,
    1,
    {input_placeholder: images, label_placeholder: labels},
    gradient_tensors,
)[1]
# wrong_grads = sess_dace.run(
#    gradient_tensors[0],
#    feed_dict={input_placeholder: images, label_placeholder: labels},
# )
# outputs_dace = sess_dace.train(
#    update_op,
#    init,
#    1,
#    {input_placeholder: images, label_placeholder: labels},
#    [logits, softmax, loss],
# )[1]
tf.summary.FileWriter("./", sess_tf.graph)
sess_tf.run(init)
##outputs_tf = sess_tf.run(
##    [logits, softmax, loss],
##    feed_dict={input_placeholder: images, label_placeholder: labels},
##)
gradients_tf = sess_tf.run(
    gradient_tensors,
    feed_dict={input_placeholder: images, label_placeholder: labels},
)

for name, tfgrad, dacegrad in zip(gradient_tensors, gradients_tf, gradients_dace):
    inf_norm = np.linalg.norm((tfgrad - dacegrad).flatten())
    print(str(name), str(inf_norm))
#print(gradients_dace)
#print(gradients_tf)
#print(tf.linalg.norm(gradients_dace - gradients_tf).eval(session=sess_tf))
#print("gradient was ", str(gradient_tensors[0]))
#print("older one was ", str(gradient_tensors[0]), " new one is ", str(gradient_tensors[1]))

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
