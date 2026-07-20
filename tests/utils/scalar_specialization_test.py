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


def test_python_tasklet_with_cast_falls_back_to_token_replace():
    """A scalar-consuming Python tasklet whose code contains a type cast (e.g.
    ``dace.int32(1)``) is not valid sympy syntax: sympy's parser misreads ``dace.int32``
    as an attribute-access node and then tries to call it, raising ``TypeError: 'Attr'
    object is not callable``. Regression for a real crash hit specializing CloudSC's
    ``kidia`` scalar (its consuming tasklets promote it via an explicit int32 cast).
    Must fall back to plain token substitution instead of crashing.
    """
    sdfg = dace.SDFG('cast_specialize')
    sdfg.add_scalar('kidia', dace.int32)
    sdfg.add_array('out', [1], dace.int32)
    state = sdfg.add_state()
    in_node = state.add_read('kidia')
    out_node = state.add_write('out')
    tasklet = state.add_tasklet('addcast', {'__in1'}, {'__out'}, '__out = __in1 + dace.int32(1)')
    state.add_edge(in_node, None, tasklet, '__in1', dace.memlet.Memlet.from_array('kidia', sdfg.arrays['kidia']))
    state.add_edge(tasklet, '__out', out_node, None, dace.memlet.Memlet.from_array('out', sdfg.arrays['out']))
    sdfg.validate()

    specialize_scalar(sdfg, 'kidia', 5)
    sdfg.validate()

    out = np.zeros(1, dtype=np.int32)
    # kidia stays a required (now-unused) argument: specialize_scalar only removes the
    # data descriptor for a NESTED SDFG (root != sdfg); the root SDFG keeps it, per
    # _specialize_scalar_impl's own "can't remove non-transient scalar" comment.
    sdfg(out=out, kidia=5)
    assert out[0] == 6


if __name__ == "__main__":
    test_nested_sdfg()
    test_python_tasklet_with_cast_falls_back_to_token_replace()
