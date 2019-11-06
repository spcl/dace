#!/usr/bin/env python
from __future__ import print_function

import dace
import numpy as np

N = dace.symbol('N')

materialize_V = """
void __dace_materialize(const char* arrayname, int start, int end, void* outarray) {
    for (int i=0; i<end-start; i++) ((double*)outarray)[i] = start+i;
}
"""

serialize_Vout = """
void __dace_serialize(const char* arrayname, int start, int end, const void* outarray) {
    printf("someone asked me to write %s[%i,%i] = %lf\\n", arrayname, start, end, ((double*)outarray)[0]);
}
"""


@dace.program(
    dace.immaterial(dace.float64[N], materialize_V),
    dace.immaterial(dace.float64[N], serialize_Vout))
def mpihello(V, Vout):
    # Transient variable
    @dace.map(_[0:N:2])
    def multiplication(i):
        in_V << V[i:i + 2]
        out >> Vout[i:i + 2]
        printf("Hello %lf, %lf\n", in_V[0], in_V[1])
        out[0] = in_V[1]
        out[1] = in_V[0]


if __name__ == "__main__":

    N.set(128)
    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)

    print('Vector add MPI %d' % (N.get()))

    mpihello(V, Vout)
