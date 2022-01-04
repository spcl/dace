# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
from __future__ import print_function

import argparse
import dace
import math
import numpy as np
import sys

W = dace.symbol("W")
H = dace.symbol("H")
MAXITER = dace.symbol("MAXITER")


@dace.program
def mandelbrot(output: dace.uint16[H, W]):
    @dace.map(_[0:H, 0:W])
    def compute_pixel(py, px):
        out >> output[py, px]

        x0 = -2.5 + ((float(px) / W) * 3.5)
        y0 = -1 + ((float(py) / H) * 2)
        x = 0.0
        y = 0.0
        iteration = 0
        while (x * x + y * y < 2 * 2 and iteration < MAXITER):
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iteration = iteration + 1

        out = iteration


# Prints out a color (in [0,1]) to a 256-color ANSI terminal)
def printcolor(val):
    ESC = "\x1B["
    MINVAL = 232
    MAXVAL = 255
    color = int(val * (MAXVAL - MINVAL) + MINVAL)
    #232 -- 255
    sys.stdout.write((ESC + "48;5;%dm " + ESC + "0m") % color)


def printmatrix(mat, image_width=20, aspect_ratio=0.5):
    h, w = mat.shape
    ratio = image_width / float(w)
    image_height = int(ratio * h * aspect_ratio)

    mn = np.min(mat)
    mx = np.max(mat)

    # Subsampling
    for y in range(image_height):
        for x in range(image_width):
            printcolor((mat[int(y / (ratio * aspect_ratio)), int(x / ratio)] - mn) / float(mx - mn))
        sys.stdout.write('\n')
    sys.stdout.flush()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("W", type=int, nargs="?", default=64)
    parser.add_argument("H", type=int, nargs="?", default=64)
    parser.add_argument("MAXITER", type=int, nargs="?", default=1000)
    parser.add_argument("--fpga", dest="fpga", action="store_true", default=False)
    args = vars(parser.parse_args())

    W.set(args["W"])
    H.set(args["H"])
    MAXITER.set(args["MAXITER"])

    print('Mandelbrot %dx%d (iterations=%d)' % (W.get(), H.get(), MAXITER.get()))

    out = dace.ndarray([H, W], dtype=dace.uint16)
    out[:] = dace.uint32(0)

    # Run DaCe program
    mandelbrot = mandelbrot.to_sdfg()
    if args["fpga"]:
        from dace.transformation.interstate import FPGATransformSDFG
        mandelbrot.apply_transformations(FPGATransformSDFG)
    mandelbrot(output=out, MAXITER=MAXITER, W=W, H=H)

    print('Result:')
    printmatrix(out)

    # Uncomment to output a PNG file
    #import png
    #with open('dacebrot.png', 'wb') as fp:
    #    w = png.Writer(W.get(), H.get(), greyscale=True, bitdepth=8)
    #    mn = np.min(out)
    #    mx = np.max(out)
    #    w.write(fp, 255.0 * (out - mn) / (mx - mn))
