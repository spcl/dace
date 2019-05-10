import tensorflow as tf
import resnet50
import numpy as np
import dace
from dace.frontend.tensorflow import TFSession
from resnet50 import SEED

learning_rate = 0.01
batch_size = 8
num_classes = 10

dace.Config.append("compiler", "cpu", "args", value=" -faligned-new")


def random_batch(batch_size):
    shape = (batch_size, 224, 224, 3)
    images = np.random.uniform(size=shape).astype(np.float32)
    labels = np.random.randint(
        low=0, high=num_classes, size=(batch_size)).astype(np.int32)
    # print(labels.shape)
    return images, labels


images, labels = random_batch(batch_size)
# Graph building
myresnet = resnet50.ResNet50(
    "channels_last", classes=num_classes, trainable=False)
input_placeholder = tf.placeholder(
    dtype=tf.float32, shape=(batch_size, 224, 224, 3))
label_placeholder = tf.placeholder(dtype=tf.int32, shape=(batch_size))
logits = myresnet(input_placeholder)
# predict = tf.argmax(logits, axis=1)
softmax = tf.nn.sparse_softmax_cross_entropy_with_logits(
    labels=label_placeholder, logits=logits)
loss = tf.reduce_mean(softmax, name="loss")
gradients = tf.train.GradientDescentOptimizer(learning_rate).compute_gradients(
    loss)
gradient_tensors = [tup[0] for tup in gradients]
update_op = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(
    gradients)
sess_dace = TFSession(seed=SEED)
sess_tf = tf.Session()
init = tf.global_variables_initializer()
gradients_dace = sess_dace.train(
    update_op,
    init,
    1,
    {
        input_placeholder: images,
        label_placeholder: labels
    },
    gradient_tensors,
)
# output_dace = sess_dace.run(loss, feed_dict={input_placeholder: images, label_placeholder: labels})
sess_tf.run(init)
writer = tf.summary.FileWriter("./", sess_tf.graph)
gradients_tf = sess_tf.run(
    [gradients],
    feed_dict={
        input_placeholder: images,
        label_placeholder: labels
    },
)
print(gradients_dace)
print(gradients_tf)
# train_accuracy = np.mean(np.argmax(one_hot_train, axis=1) ==
#                         sess.run(predict, feed_dict={X: x_train, y:
#                                                      one_hot_train}))
# test_accuracy  = np.mean(np.argmax(one_hot_test, axis=1) ==
#                         sess.run(predict, feed_dict={X: x_test, y:
#                                                      one_hot_test}))
# print("Epoch = %d, train accuracy = %.2f%%, test accuracy = %.2f%%" % (epoch + 1, 100. * train_accuracy, 100. * test_accuracy))
