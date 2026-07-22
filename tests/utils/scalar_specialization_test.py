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


def _scalar_into_tasklet(code, language, value):
    """One float64 output driven by a tasklet reading float64 scalar 's', specialized to `value`."""
    sdfg = dace.SDFG('spec_' + str(abs(hash((code, language, value))))[:8])
    sdfg.add_scalar('s', dace.float64)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('compute', {'_in'}, {'o'}, code, language=language)
    state.add_edge(state.add_read('s'), None, tasklet, '_in', dace.Memlet('s[0]'))
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))
    specialize_scalar(sdfg, 's', value)
    result = np.zeros([1])
    sdfg(out=result, s=np.float64(value))
    return result[0], tasklet.code.as_string


def test_integer_value_keeps_floating_point_division():
    """An int replacing a float scalar must stay a float literal, or `/` becomes integer division."""
    for language, code in ((dace.dtypes.Language.Python, 'o = _in / 2'), (dace.dtypes.Language.CPP, 'o = _in / 2;')):
        value, emitted = _scalar_into_tasklet(code, language, 5)
        assert '5.0' in emitted, emitted
        assert value == 2.5, (language, value)


def test_cpp_tasklet_substitution_matches_unspaced_operators():
    """Token matching split only on whitespace and brackets, so `_in/2` was never substituted."""
    value, code = _scalar_into_tasklet('o = _in/2;', dace.dtypes.Language.CPP, 5.0)
    assert '_in' not in code, code
    assert value == 2.5


def test_float_value_round_trips_exactly():
    """repr, not a 15-significant-digit rendering, so a float64 survives substitution."""
    value, _ = _scalar_into_tasklet('o = _in', dace.dtypes.Language.Python, 1 / 3)
    assert value == 1 / 3


if __name__ == "__main__":
    test_nested_sdfg()
    test_integer_value_keeps_floating_point_division()
    test_cpp_tasklet_substitution_matches_unspaced_operators()
    test_float_value_round_trips_exactly()
