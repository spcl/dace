# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.

import dace
import numpy as np

@dace.program
def powii(A: dace.int64[1], B: dace.int64[1], R: dace.int64[1]):
    @dace.tasklet('Python')
    def powii():
        a << A[0]
        b << B[0]
        r >> R[0]
        """r = a ** b"""

@dace.program
def powff(A: dace.float64[1], B: dace.float64[1], R: dace.float64[1]):
    @dace.tasklet('Python')
    def powff():
        a << A[0]
        b << B[0]
        r >> R[0]
        """r = a ** b"""

@dace.program
def powfi(A: dace.float64[1], B: dace.int64[1], R: dace.float64[1]):
    @dace.tasklet('Python')
    def powfi():
        a << A[0]
        b << B[0]
        r >> R[0]
        """r = a ** b"""

@dace.program
def powif(A: dace.int64[1], B: dace.float64[1], R: dace.float64[1]):
    @dace.tasklet('Python')
    def powif():
        a << A[0]
        b << B[0]
        r >> R[0]
        """r = a ** b"""

def test_powii():
    """ Tests tasklets containing power operations """
    sdfg = powii.to_sdfg()
    sdfg.validate()

    a = np.random.randint(0, 10, 1).astype(np.int64)
    b = np.random.randint(0, 10, 1).astype(np.int64)
    r = np.random.randint(0, 10, 1).astype(np.int64)
    sdfg(A=a, B=b, R=r)
    assert r[0] == a[0] ** b[0]

def test_powff():
    """ Tests tasklets containing power operations """
    sdfg = powff.to_sdfg()
    sdfg.validate()

    a = np.random.rand(1).astype(np.float64)
    b = np.random.rand(1).astype(np.float64)
    r = np.random.rand(1).astype(np.float64)
    sdfg(A=a, B=b, R=r)
    assert r[0] == a[0] ** b[0]

def test_powfi():
    """ Tests tasklets containing power operations """
    sdfg = powfi.to_sdfg()
    sdfg.validate()

    a = np.random.rand(1).astype(np.float64)
    b = np.random.randint(0, 10, 1).astype(np.int64)
    r = np.random.rand(1).astype(np.float64)
    sdfg(A=a, B=b, R=r)
    assert r[0] == a[0] ** b[0]

def test_powif():
    """ Tests tasklets containing power operations """
    sdfg = powif.to_sdfg()
    sdfg.validate()

    a = np.random.randint(0, 10, 1).astype(np.int64)
    b = np.random.rand(1).astype(np.float64)
    r = np.random.rand(1).astype(np.float64)
    sdfg(A=a, B=b, R=r)
    assert r[0] == a[0] ** b[0]


if __name__ == "__main__":
    test_powii()
    test_powff()
    test_powfi()
    test_powif()
