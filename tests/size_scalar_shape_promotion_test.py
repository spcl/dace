# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""A size computed in the program can be used as an array shape.

``nt = Nt + 1`` materializes ``nt`` as a scalar data descriptor, but an array extent has to be a
symbol, and minting a symbol of the same name collided with the descriptor (``FileExistsError``).
The size is now read into a ``__sym_`` symbol on an interstate edge and substituted into the shape,
leaving the descriptor in place so the program can keep reading or reassigning the size afterwards.
"""
import numpy as np
import dace

N = dace.symbol('N')


@dace.program
def size_from_empty(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    b = np.empty(Nt + 1, dace.float64)
    for i in range(N):
        b[i] = a[i] * 2.0
    for i in range(N):
        out[i] = b[i]


@dace.program
def size_read_after_use(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    m = Nt + 1
    b = np.empty(m, dace.float64)
    b[0] = 1.0
    for i in range(N):
        out[i] = a[i] + m  # the size descriptor must survive its use as a shape


@dace.program
def size_reassigned_after_use(a: dace.float64[N], Nt: dace.int64, out: dace.float64[N]):
    m = Nt + 1
    b = np.empty(m, dace.float64)
    b[0] = 1.0
    m = 99
    for i in range(N):
        out[i] = a[i] + m


def test_scalar_size_as_shape():
    n, nt = 5, 7
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    size_from_empty(a, np.int64(nt), out, N=n)
    assert np.allclose(out, a * 2.0)


def test_size_descriptor_survives_its_use_as_a_shape():
    """Promotion must not delete the scalar: the program still reads it afterwards."""
    n, nt = 5, 7
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    size_read_after_use(a, np.int64(nt), out, N=n)
    assert np.allclose(out, a + (nt + 1))


def test_size_can_be_reassigned_after_use_as_a_shape():
    n, nt = 5, 7
    a = np.arange(n, dtype=np.float64)
    out = np.zeros(n)
    size_reassigned_after_use(a, np.int64(nt), out, N=n)
    assert np.allclose(out, a + 99)


def test_promotion_leaves_the_descriptor_in_place():
    sdfg = size_read_after_use.to_sdfg(simplify=False)
    promoted = [s for s in sdfg.symbols if s.startswith('__sym_')]
    assert promoted, 'the size scalar must be read into a symbol'
    # The descriptor stays: deleting it is what broke later reads of the size.
    assert any(s[len('__sym_'):] in sdfg.arrays for s in promoted)
    sdfg.validate()


if __name__ == '__main__':
    test_scalar_size_as_shape()
    test_size_descriptor_survives_its_use_as_a_shape()
    test_size_can_be_reassigned_after_use_as_a_shape()
    test_promotion_leaves_the_descriptor_in_place()
