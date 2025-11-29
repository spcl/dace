import dace
import copy
import numpy as np
from dace.sdfg.utils import specialize_scalar

Y = dace.symbolic.symbol("Y")
X = dace.symbolic.symbol("X")


@dace.program
def gather_load(A: dace.float64[Y, X], B: dace.int64[Y, X], C: dace.float64[Y, X], scale: dace.float64):
    for i, j in dace.map[0:Y:1, 0:X:1]:
        C[i, j] = A[i, B[i, j]] * scale


# Test function
def test_nested_sdfg():
    # Create SDFG
    sdfg = gather_load.to_sdfg()
    sdfg.validate()
    copy_sdfg = copy.deepcopy(sdfg)
    copy_sdfg.validate()
    copy_sdfg.save("s0.sdfg")
    # Specialize the scalar 'scale' in the copy
    specialize_scalar(copy_sdfg, "scale", 2.0)

    # Create inputs
    y_val, x_val = 32, 16
    A = np.random.rand(y_val, x_val)
    B = np.random.randint(0, x_val, size=(y_val, x_val), dtype=np.int64)
    C_orig = np.zeros((y_val, x_val))
    C_spec = np.zeros((y_val, x_val))

    # Run original SDFG
    sdfg(A=A, B=B, C=C_orig, scale=2.0, Y=y_val, X=x_val)

    # Run specialized SDFG (scale is fixed to 2.0)
    # scale arg is unused but need to pass
    copy_sdfg(A=A, B=B, C=C_spec, Y=y_val, X=x_val, scale=2.0)

    # Verify correctness for scale=2.0
    np.testing.assert_allclose(C_spec, A[np.arange(y_val)[:, None], B] * 2.0)
    np.testing.assert_allclose(C_spec, C_orig)


if __name__ == "__main__":
    test_nested_sdfg()
