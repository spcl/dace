import dace
import copy
import numpy as np
from dace.sdfg.state import LoopRegion
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


def test_symbol_only_nested_propagation():
    """N reaches the nested SDFG purely through symbol_mapping (key='N', value='N', a plain
    passthrough) -- never as a Scalar/array data connection. This is the CloudSC ``nclv``
    case: a plain dace.symbol used as an array-shape/loop bound never satisfies the
    data-memlet-gated recursion (in_data_mapping), so without handling symbol_mapping
    directly, the nested SDFG kept N as an unresolved free symbol while the outer scope no
    longer passed it in -- specialize_scalar used to only remove root-level Scalar data, it
    did not clean up symbol_mapping for a pure symbol at all.
    """
    sdfg = dace.SDFG('symbol_only_specialize')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [1], dace.int32)

    nested_sdfg = dace.SDFG('inner')
    nested_sdfg.add_symbol('N', dace.int32)
    nested_sdfg.add_symbol('i', dace.int32)
    nested_sdfg.add_array('A', [1], dace.int32)
    loop = LoopRegion('loop', condition_expr='i < N', loop_var='i', initialize_expr='i = 0', update_expr='i = i + 1')
    nested_sdfg.add_node(loop, is_start_block=True)
    body = loop.add_state('body', is_start_block=True)
    tasklet = body.add_tasklet('write', {}, {'out'}, 'out = i')
    inner_out = body.add_write('A')
    body.add_edge(tasklet, 'out', inner_out, None, dace.Memlet('A[0]'))

    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(nested_sdfg, set(), {'A'}, symbol_mapping={'N': 'N'})
    outer_out = state.add_write('A')
    state.add_edge(nsdfg_node, 'A', outer_out, None, dace.memlet.Memlet.from_array('A', sdfg.arrays['A']))
    sdfg.validate()

    specialize_scalar(sdfg, 'N', 5)
    sdfg.validate()

    assert 'N' not in nsdfg_node.symbol_mapping
    assert 'N' not in nested_sdfg.symbols
    assert 'N' not in {str(s) for s in nested_sdfg.free_symbols}

    A = np.zeros(1, dtype=np.int32)
    sdfg(A=A, N=5)
    assert A[0] == 4  # loop runs i=0..4 (i<5), last write is i=4


def test_symbol_in_nested_mapping_value_only_substitutes_expression():
    """scalar_name can appear on the VALUE side of some OTHER inner symbol's symbol_mapping
    entry (e.g. {'M': 'N + 1'}) without the nested SDFG ever having a free symbol literally
    named scalar_name. Only the bound expression should be substituted -- the mapping entry
    for the other inner symbol (M) must remain (it still needs a binding, just a constant
    one), and nothing should be removed from or recursed into the nested SDFG.
    """
    sdfg = dace.SDFG('value_side_specialize')
    sdfg.add_symbol('N', dace.int32)
    sdfg.add_array('A', [5], dace.float64)

    nested_sdfg = dace.SDFG('inner')
    nested_sdfg.add_symbol('M', dace.int32)
    nested_sdfg.add_array('A', [5], dace.float64)
    nstate = nested_sdfg.add_state()
    tasklet = nstate.add_tasklet('write', {}, {'out'}, 'out = M')
    inner_out = nstate.add_write('A')
    nstate.add_edge(tasklet, 'out', inner_out, None, dace.Memlet('A[0]'))

    state = sdfg.add_state()
    nsdfg_node = state.add_nested_sdfg(nested_sdfg, set(), {'A'}, symbol_mapping={'M': 'N + 1'})
    outer_out = state.add_write('A')
    state.add_edge(nsdfg_node, 'A', outer_out, None, dace.memlet.Memlet.from_array('A', sdfg.arrays['A']))
    sdfg.validate()

    specialize_scalar(sdfg, 'N', 4)
    sdfg.validate()

    # Real sympy substitution (unlike the tasklet-code token-replace fallback), so the
    # expression is folded to a plain constant, not left as an unevaluated '4 + 1'.
    assert nsdfg_node.symbol_mapping['M'] == 5
    assert 'M' in nested_sdfg.symbols

    A = np.zeros(5)
    sdfg(A=A, N=4)
    assert A[0] == 5


if __name__ == "__main__":
    test_nested_sdfg()
    test_python_tasklet_with_cast_falls_back_to_token_replace()
    test_symbol_only_nested_propagation()
    test_symbol_in_nested_mapping_value_only_substitutes_expression()


def _scalar_into_tasklet(code: str, dtype, value):
    """One float64 output driven by a single tasklet reading scalar 's', specialized to `value`."""
    sdfg = dace.SDFG('spec_' + str(abs(hash(code)))[:8])
    sdfg.add_scalar('s', dtype)
    sdfg.add_array('out', [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('compute', {'_in'}, {'o'}, code)
    state.add_edge(state.add_read('s'), None, tasklet, '_in', dace.Memlet('s[0]'))
    state.add_edge(tasklet, 'o', state.add_write('out'), None, dace.Memlet('out[0]'))
    specialize_scalar(sdfg, 's', value)
    result = np.zeros([1])
    sdfg(out=result, s=np.float64(value))
    return result[0], tasklet.code.as_string


def test_specialization_preserves_floating_point_semantics():
    """ The substituted value used to be parsed into sympy and printed back, which rewrote the
        arithmetic rather than just filling in a number: it folded across floating-point
        operations, reassociated sums, and printed only 15 significant digits. Each case below
        returned a different double than the unspecialized program did. """
    # folding across a float add: sympy produced 'b + 0.3', but 0.2 + 0.1 is 0.30000000000000004
    value, _ = _scalar_into_tasklet('o = _in + 0.1 + 1.0', dace.float64, 0.2)
    assert value == 0.2 + 0.1 + 1.0

    # 15-significant-digit printing does not round-trip a float64
    value, _ = _scalar_into_tasklet('o = _in', dace.float64, 1 / 3)
    assert value == 1 / 3

    value, _ = _scalar_into_tasklet('o = _in*3.0', dace.float64, 1 / 3)
    assert value == (1 / 3) * 3.0


def test_integer_value_keeps_floating_point_division():
    """ An int substituted into code that read a floating-point scalar must be written as a float
        literal. `_in / 2` compiled as float division before specialization; emitting a bare `5`
        turns it into integer division in the generated C++, i.e. 2 instead of 2.5. """
    value, code = _scalar_into_tasklet('o = _in / 2', dace.float64, 5)
    assert '5.0' in code
    assert value == 2.5


def test_substitution_matches_unspaced_operators():
    """ Token matching split only on whitespace and brackets, so `_in/2` stayed one token and was
        never replaced -- leaving the tasklet referencing a connector that specialization had just
        removed. """
    value, code = _scalar_into_tasklet('o = _in/2', dace.float64, 5.0)
    assert '_in' not in code
    assert value == 2.5
