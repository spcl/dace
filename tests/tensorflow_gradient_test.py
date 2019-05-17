import tensorflow as tf
import dace
from tensorflow.python.training import training_ops
import numpy as np
test_alpha = np.random.rand(1)
test_delta = np.random.rand(8,224,224,3)
var = tf.get_variable("myvar", [8,224,224,3], initializer=tf.random_uniform_initializer)
#alpha = tf.placeholder(tf.float64, [1])
#delta = tf.placeholder(tf.float64, [8,224,224,3])
loss = tf.reduce_mean(var)
backpass = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
tf.Session().run(tf.initialize_all_variables())
tf.Session().run(backpass)
