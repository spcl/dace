# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
""" Predicate-based filtering with dynamic, explicit memlets in DaCe. """
import argparse
import dace
import numpy as np

N = dace.symbol('N', positive=True)


@dace.program
def pbf(A: dace.float32[N], out: dace.float32[N], outsz: dace.uint32[1], ratio: dace.float32):
    # We define a stream (an object that behaves like a queue) so that we can dynamically
    # push values to `out`
    ostream = dace.define_stream(dace.float32, N)

    # The map evaluates a single element from `A` at a time
    for i in dace.map[0:N]:
        with dace.tasklet:
            a << A[i]
            r << ratio

            # The filter predicate is based on the ratio
            filter = (a > r)

            # If we should filter, writing `b = a` pushes `a` onto the stream
            if filter:
                b = a

            # With write-conflict resolution, storing the filter predicate would add it to `outsz`
            osz = filter

            # Writing to the output stream uses a dynamic output memlet, annotated with -1
            b >> ostream(-1)
            
            # Writing to the output size is also dynamic, and uses the sum write-conflict resolution
            osz >> outsz(-1, lambda x, y: x + y, 0)

    # Lastly, we connect ostream to the output array. DaCe detects this pattern and emits
    # fast code that pushes results to `out` directly
    ostream >> out


def regression(A, ratio):
    return A[np.where(A > ratio)]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("N", type=int, nargs="?", default=64)
    parser.add_argument("ratio", type=float, nargs="?", default=0.5)
    args = parser.parse_args()

    # Initialize arrays
    ratio = np.float32(args.ratio)
    A = np.random.rand(args.N).astype(np.float32)
    B = np.zeros_like(A)
    outsize = dace.scalar(dace.uint32)
    outsize[0] = 0

    # Call filter
    pbf(A, B, outsize, ratio)

    filtered = regression(A, ratio)

    if len(filtered) != outsize[0]:
        print(f'Difference in number of filtered items: {outsize[0]} (DaCe) vs. {len(filtered)} (numpy)')
        totalitems = min(outsize[0], args.N)
        print('DaCe:', B[:totalitems])
        print('numpy:', filtered)
        exit(1)

    # Sort the outputs
    filtered = np.sort(filtered)
    B[:outsize[0]] = np.sort(B[:outsize[0]])

    if len(filtered) == 0:
        print('Success, nothing left in array')
        exit(0)

    diff = np.linalg.norm(filtered - B[:outsize[0]]) / float(outsize[0])
    print('Difference:', diff)
    if diff > 1e-5:
        totalitems = min(outsize[0], args.N)
        print('DaCe:', B[:totalitems])
        print('numpy:', filtered)

    print("==== Program end ====")
    exit(0 if diff <= 1e-5 else 1)
