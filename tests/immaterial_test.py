#!/usr/bin/env python
from __future__ import print_function

import dace
import warnings

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


@dace.program(dace.immaterial(dace.float64[N], materialize_V),
              dace.immaterial(dace.float64[N], serialize_Vout))
def immaterial_test(V, Vout):
    # Transient variable
    @dace.map(_[0:N])
    def multiplication(i):
        in_V << V[i]
        out >> Vout[i]
        printf("Hello %lf\n", in_V)
        out = in_V


if __name__ == "__main__":

    if dace.Config.get_bool('optimizer',
                            'automatic_strict_transformations') == False:
        warnings.warn(
            'This test is not supported in non-strict mode. Skipping')
        exit(0)

    N.set(16)

    V = dace.ndarray([N], dace.float64)
    Vout = dace.ndarray([N], dace.float64)

    print('Immaterial element access test %d' % (N.get()))

    immaterial_test(V, Vout)
