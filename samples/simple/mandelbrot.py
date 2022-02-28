# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" A code sample that uses a data-centric map to compute the Mandelbrot set in parallel. """
from __future__ import print_function

import argparse
import dace
import numpy as np
import sys

# Define symbols for output size
W = dace.symbol("W")
H = dace.symbol("H")


@dace.program
def mandelbrot(output: dace.uint16[H, W], maxiter: dace.int64):
    for py, px in dace.map[0:H, 0:W]:
        x0 = -2.5 + ((float(px) / W) * 3.5)
        y0 = -1 + ((float(py) / H) * 2)
        x = 0.0
        y = 0.0
        iteration = 0
        while (x * x + y * y < 2 * 2 and iteration < maxiter):
            xtemp = x * x - y * y + x0
            y = 2 * x * y + y0
            x = xtemp
            iteration = iteration + 1

        output[py, px] = iteration


############################################
# Helper functions


def printcolor(val):
    """ Prints out a color (in [0,1]) to a 256-color ANSI terminal). """
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
    parser.add_argument("iterations", type=int, nargs="?", default=1000)
    parser.add_argument("-o", dest="output", type=str, default=None)
    args = parser.parse_args()

    print(f'Mandelbrot {args.W}x{args.H} ({args.iterations} iterations)')

    # Setup data
    out = np.zeros([args.H, args.W], dtype=np.uint16)

    # Run DaCe program
    mandelbrot(out, args.iterations)

    print('Result:')
    printmatrix(out)

    # Output a PNG file
    if args.output:
        try:
            import png
        except (ImportError, ModuleNotFoundError):
            print('Saving to png requires the pypng module. Install with `pip install pypng`')
            exit(1)
        with open(args.output, 'wb') as fp:
            w = png.Writer(args.W, args.H, greyscale=True, bitdepth=8)
            mn = np.min(out)
            mx = np.max(out)
            w.write(fp, (255.0 * (out - mn) / (mx - mn)).astype(np.uint8))
