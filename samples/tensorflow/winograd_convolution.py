import dace
import tensorflow as tf
import numpy as np
from dace.transformation.dataflow import MapFusion

TILE_SIZE = 4
OUTPUT_TILE_SIZE = 2
P = 128 * 29 * 29

bt = np.array([[1, 0, -1, 0], [0, 1, 1, 0], [0, -1, 1, 0], [0, 1, 0, -1]]).astype(
    np.float32
)
g = np.array([[1.0, 0.0, 0.0], [0.5, 0.5, 0.5], [0.5, -0.5, 0.5], [0.0, 0.0, 1.0]])
at = np.array([[1, 1, 1, 0], [0, 1, -1, -1]]).astype(np.float32)


def btb(outp, op, inp):
    res = np.matmul(np.matmul(op, inp), np.transpose(op))
    np.copyto(outp, res)

def btb_print(outp, op, inp):
    res = np.matmul(np.matmul(op, inp), np.transpose(op))
    print(res)
    np.copyto(outp, res)

def eltwise(outp, inp1, inp2):
    res = inp1 * inp2
    np.copyto(outp, res)


@dace.program(
    dace.float32[1, 56, 56, 2, 1],
    dace.float32[1, 56, 56, 1],
    dace.float32[3, 3, 1, 2],
    dace.float32[TILE_SIZE, TILE_SIZE],  # B_transpose
    dace.float32[TILE_SIZE, 3],  # G
    dace.float32[OUTPUT_TILE_SIZE, TILE_SIZE],  # A_transpose
    dace.callback(
        None,
        dace.float32[TILE_SIZE, TILE_SIZE],
        dace.float32[TILE_SIZE, TILE_SIZE],  # Operator_input
        dace.float32[TILE_SIZE, TILE_SIZE],
    ),
    dace.callback(
        None,
        dace.float32[TILE_SIZE, TILE_SIZE],
        dace.float32[TILE_SIZE, 3],  # Operator_kernel
        dace.float32[3, 3],
    ),
    dace.callback(
        None,
        dace.float32[OUTPUT_TILE_SIZE, OUTPUT_TILE_SIZE],
        dace.float32[OUTPUT_TILE_SIZE, TILE_SIZE],  # Operator_output
        dace.float32[TILE_SIZE, TILE_SIZE],
    ),
    dace.callback(  # Matrix multiplication
        None,
        dace.float32[TILE_SIZE, TILE_SIZE],
        dace.float32[TILE_SIZE, TILE_SIZE],
        dace.float32[TILE_SIZE, TILE_SIZE],
    ),
)
def winograd_conv(
    output, image, kernel, bt, g, at, input_txform, kernel_txform, output_txform, matmul
):
    tile_output = dace.define_local([4, 4], dtype=dace.float32)
    image_transformed = dace.define_local([4, 4], dace.float32)
    kern_transformed = dace.define_local([4, 4], dace.float32)
    # Loop over all tiles
    @dace.map(_[0:1, 0:28:2, 0:28:2, 0:2, 0:1])
    def image_txform(n, tx, ty, cout, cin):
        # Reduction map over input channels
        with dace.tasklet:
            inp << image[n, tx : tx + 4, ty : ty + 4, cin]
            img_operator << bt
            outp_img >> image_transformed
            input_txform(outp_img, img_operator, inp)

    @dace.map(_[0:1, 0:28:2, 0:28:2, 0:2, 0:1])
    def kernel_txform(n, tx, ty, cout, cin):
        with dace.tasklet:
            fil << kernel[0:3, 0:3, cin, cout]
            fil_operator << g
            outp_fil >> kern_transformed
            kernel_txform(outp_fil, fil_operator, fil)

    @dace.map(_[0:1, 0:28:2, 0:28:2, 0:2, 0:1])
    def eltwise(n, tx, ty, cout, cin):
        with dace.tasklet:
            temp_im << image_transformed
            temp_kern << kern_transformed
            temp_out >> tile_output
            matmul(temp_out, temp_im, temp_kern)

    @dace.map(_[0:1, 0:28:2, 0:28:2, 0:2, 0:1])
    def output_txform(n, tx, ty, cout, cin):
        with dace.tasklet:
            final_out >> output[n, tx : tx + 2, ty : ty + 2, cout, cin]
            temp_in << tile_output
            temp_op << at
            output_txform(final_out, temp_op, temp_in)


image = np.random.rand(1, 56, 56, 1).astype(np.float32)
filter = np.random.rand(3, 3, 1, 2).astype(np.float32)
padding = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])

output_tf = tf.Session().run(tf.nn.conv2d(image, filter, [1, 1, 1, 1], "SAME"))
output = np.zeros_like(output_tf).astype(np.float32)

image_padded = tf.Session().run(tf.pad(image, padding, "CONSTANT"))
wino_sdfg = winograd_conv.to_sdfg()
wino_sdfg.apply_transformations([MapFusion])
wino_sdfg.draw_to_file("winograd.dot")
winograd_conv(
    output, image_padded, filter, bt, g, at, btb, btb, btb_print, eltwise
)
#output = np.sum(output, axis=4, keepdims=False)

print(tf.linalg.norm(output_tf - output).eval(session=tf.Session()))
print(output)
print(output_tf)
