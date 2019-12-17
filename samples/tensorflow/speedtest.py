import tensorflow as tf
import numpy as np
import time
from dace.frontend.tensorflow import TFSession

#c*k < 256*256
w, h, c, n, k, s, r, pad_w, pad_h, wstride, hstride = (56, 56, 128, 16, 256, 3, 3, 1, 1, 1, 1)
input_dim = [n,h,w,c]
filter_dim = [r,s,c,k]
input = tf.placeholder(tf.float32, input_dim)
filter = tf.placeholder(tf.float32, filter_dim)
output = tf.nn.conv2d(input, filter, strides=[1, hstride, wstride, 1], padding="VALID", data_format="NHWC")
test_filter = np.random.uniform(size=tuple(filter_dim)).astype(np.float32)
test_input = np.random.uniform(size=tuple(input_dim)).astype(np.float32)

config = tf.ConfigProto(device_count={'GPU': 0})
config.gpu_options.allow_growth = True

sess_dace = TFSession()
sess_tf = tf.Session(config=config)

sess_run = sess_dace.compile(output, gpu=False)
start_dace_cpu = time.time()
output_dace = sess_run(feed_dict={input: test_input, filter: test_filter})
end_dace_cpu = time.time()

sess_run = sess_dace.compile(output, gpu=True, cudnn=False)
start_dace_gpu = time.time()
output_dace = sess_run(feed_dict={input: test_input, filter: test_filter})
end_dace_gpu = time.time()

sess_run = sess_dace.compile(output, gpu=True, cudnn=True)
start_dace_cudnn = time.time()
output_dace = sess_run(feed_dict={input: test_input, filter: test_filter})
end_dace_cudnn = time.time()

start_tf = time.time()
output_tf = sess_tf.run(output, feed_dict={input: test_input, filter: test_filter})
end_tf = time.time()

print("time dace cpu: ", end_dace_cpu-start_dace_cpu)
print("time dace gpu: ", end_dace_gpu-start_dace_gpu)
print("time dace cudnn: ", end_dace_cudnn-start_dace_cudnn)
print("time tf: ", end_tf-start_tf)
