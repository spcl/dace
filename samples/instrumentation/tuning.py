# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This sample uses basic grid-search based tuning to adapt memory layouts for
    a simple matrix multiplication. """
import dace
from dace.codegen.instrumentation.report import InstrumentationReport
import itertools
import math
import numpy as np
import sys

# Set data type
dtype = dace.float64

# Set number of repetitions
REPS = 10

# Define symbols
M, K, N = tuple(dace.symbol(name) for name in ('M', 'K', 'N'))


# Our program is a simple matrix multiplication with unknown dimensions
@dace.program
def matmult(A: dtype[M, K], B: dtype[K, N], C: dtype[M, N]):
    for i in range(REPS):
        C[:] = A @ B


def test_configuration(a_trans: bool, b_trans: bool, a_padding: int, b_padding: int) -> InstrumentationReport:
    """
    Tests a single configuration of A and B and returns the instrumentation
    report from running the SDFG.
    """
    # Convert the program to an SDFG to enable instrumentation
    sdfg = matmult.to_sdfg()
    # Remove extraneous states
    sdfg.coarsen_dataflow()

    # Instrument state that runs in the loop above
    state = next(s for s in sdfg.nodes() if len(s.nodes()) > 0)
    state.instrument = dace.InstrumentationType.Timer

    # Modify properties of SDFG arrays according to the configuration:
    # padding (round to the nearest padding value) and total size.
    if a_trans:
        a_strides = (1, int(math.ceil(M.get() / a_padding) * a_padding))
        total_a = int(a_strides[1] * K.get())
    else:
        a_strides = (int(math.ceil(K.get() / a_padding) * a_padding), 1)
        total_a = int(a_strides[0] * M.get())

    if b_trans:
        b_strides = (1, int(math.ceil(K.get() / b_padding) * b_padding))
        total_b = int(b_strides[1] * N.get())
    else:
        b_strides = (int(math.ceil(N.get() / b_padding) * b_padding), 1)
        total_b = int(b_strides[0] * K.get())

    # NOTE: In DaCe, strides are denoted in absolute values, meaning that each
    #       dimension of "strides" contains the number of elements to skip in
    #       order to get to the next element in that dimension. For example,
    #       contiguous dimensions are denoted by 1
    sdfg.arrays['A'].strides = a_strides
    sdfg.arrays['A'].total_size = total_a
    sdfg.arrays['B'].strides = b_strides
    sdfg.arrays['B'].total_size = total_b

    # Create matching arrays in numpy and fill with random values
    nbytes = dtype.bytes
    A = np.ndarray([M.get(), K.get()],
                   dtype.type,
                   buffer=np.ndarray([total_a], dtype.type),
                   strides=[s * nbytes for s in a_strides])
    B = np.ndarray([K.get(), N.get()],
                   dtype.type,
                   buffer=np.ndarray([total_b], dtype.type),
                   strides=[s * nbytes for s in b_strides])

    A[:] = np.random.rand(M.get(), K.get())
    B[:] = np.random.rand(K.get(), N.get())
    C = np.zeros([M.get(), N.get()], dtype.type)

    # Invoke SDFG: compile without additional transformations and run
    csdfg = sdfg.compile()
    csdfg(A=A, B=B, C=C, M=np.int32(M.get()), K=np.int32(K.get()), N=np.int32(N.get()))
    assert np.allclose(A @ B, C)

    # Return instrumentation report
    return sdfg.get_latest_report()


if __name__ == '__main__':
    # Define some example sizes or use command line arguments
    M.set(int(sys.argv[1] if len(sys.argv) > 1 else 257))
    K.set(int(sys.argv[2] if len(sys.argv) > 2 else 258))
    N.set(int(sys.argv[3] if len(sys.argv) > 3 else 319))

    # Disable debug printouts
    dace.Config.set('debugprint', value=False)

    # Create options for storage orders and padding
    ORDERS = ['normal', 'transposed']
    PADDINGS = [1, 16, 512, 4096]

    best_config = (None, None, None, None)
    best_runtime = np.inf

    # NOTE: To keep tests fast we fix C's storage order and padding
    for tA_order, tB_order in itertools.product(ORDERS, ORDERS):
        for tA_padding, tB_padding in itertools.product(PADDINGS, PADDINGS):
            print(tA_order, tA_padding, tB_order, tB_padding)
            report = test_configuration(tA_order == 'transposed', tB_order == 'transposed', tA_padding, tB_padding)

            # Obtain the first entry type from the report (there is only one)
            entry = np.array(list(report.durations.values())[0])
            print(list(entry))

            # Use median value to rank performance
            runtime_ms = np.median(entry)

            if runtime_ms < best_runtime:
                best_runtime = runtime_ms
                best_config = (tA_order, tA_padding, tB_order, tB_padding)

    # Print out best configuration
    A_order, A_padding, B_order, B_padding = best_config
    print('Fastest configuration for (%dx%dx%d) is:' % (M.get(), K.get(), N.get()))
    print('  A with storage order %s, padding = %d' % (A_order, A_padding))
    print('  B with storage order %s, padding = %d' % (B_order, B_padding))
    print('  Runtime: %f ms' % best_runtime)
