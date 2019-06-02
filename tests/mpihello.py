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

V = dace.ndarray([N], dace.float64, materialize_func=materialize_V)
Vout = dace.ndarray([N], dace.float64, materialize_func=serialize_Vout)


@dace.program(
    dace.immaterial(dace.float64[N], materialize_V),
    dace.immaterial(dace.float64[N], serialize_Vout))
def mpihello(V, Vout):
    # Transient variable
    @dace.map(_[0:N])
    def multiplication(i):
        in_V << V[i]
        out >> Vout[i]
        printf("Hello %lf\n", in_V)
        out = in_V


if __name__ == "__main__":

    N.set(50)

    print('Vector add MPI %d' % (N.get()))

    mpihello(V, Vout)

