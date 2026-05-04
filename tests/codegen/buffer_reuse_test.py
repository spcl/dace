import pytest
import dace
from dace.sdfg import InterstateEdge

def test_interstate_edge_has_reuse_field():
    edge = InterstateEdge()
    assert hasattr(edge, 'reuse')
    assert edge.reuse == []

def test_reuse_accepts_pairs():
    edge = InterstateEdge()
    edge.reuse.append(['B', 'A'])
    assert edge.reuse == [['B', 'A']]

def test_reuse_survives_deepcopy():
    import copy
    edge = InterstateEdge()
    edge.reuse.append(['B', 'A'])
    edge2 = copy.deepcopy(edge)
    assert edge2.reuse == [['B', 'A']]
    assert edge2.guid != edge.guid

def test_reuse_serializes_to_json():
    edge = InterstateEdge()
    edge.reuse.append(['B', 'A'])
    j = edge.to_json()
    edge2 = InterstateEdge.from_json(j)
    assert edge2.reuse == [['B', 'A']]

from dace import dtypes

def _make_two_array_sdfg():
    """
    Simple SDFG:  s0 --[e]--> s1
    Arrays A and B are both float64[10], Explicit lifetime.
    Edge e has reuse=[['B','A']].
    A non-transient array 'x' and a tasklet are added so that codegen
    runs successfully on this minimal SDFG.
    Returns (sdfg, edge).
    """
    sdfg = dace.SDFG('reuse_codegen_test')
    sdfg.add_array('x', [10], dace.float64)
    sdfg.add_array('A', [10], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64, transient=True)
    sdfg.arrays['A'].lifetime = dtypes.AllocationLifetime.Explicit
    sdfg.arrays['B'].lifetime = dtypes.AllocationLifetime.Explicit
    s0 = sdfg.add_state('s0', is_start_block=True)
    t = s0.add_tasklet('t', {'_in': dace.float64}, {'_out': dace.float64}, '_out = _in')
    r = s0.add_read('x')
    w = s0.add_write('x')
    s0.add_edge(r, None, t, '_in', dace.Memlet('x[0]'))
    s0.add_edge(t, '_out', w, None, dace.Memlet('x[0]'))
    s1 = sdfg.add_state('s1')
    e = sdfg.add_edge(s0, s1, InterstateEdge())
    e.data.reuse.append(['B', 'A'])
    return sdfg, e

def test_codegen_emits_pointer_assignment():
    sdfg, _ = _make_two_array_sdfg()
    cpp = sdfg.generate_code()[0].clean_code
    assert '__state->__0_B = __state->__0_A' in cpp

def test_codegen_emits_donor_nullout():
    sdfg, _ = _make_two_array_sdfg()
    cpp = sdfg.generate_code()[0].clean_code
    assert '__state->__0_A = nullptr' in cpp

def test_codegen_no_new_for_reused_array():
    sdfg, _ = _make_two_array_sdfg()
    cpp = sdfg.generate_code()[0].clean_code
    # B must not be freshly allocated — it reuses A's pointer
    assert '__state->__0_B = new double' not in cpp


def test_validation_passes_for_valid_reuse():
    sdfg, _ = _make_two_array_sdfg()
    sdfg.validate()   # must not raise

def test_validation_fails_reuse_unknown_array():
    sdfg, e = _make_two_array_sdfg()
    e.data.reuse.append(['NONEXISTENT', 'A'])
    with pytest.raises(Exception, match="non-existent"):
        sdfg.validate()

def test_validation_fails_reuse_non_explicit_lifetime():
    sdfg, e = _make_two_array_sdfg()
    sdfg.arrays['A'].lifetime = dtypes.AllocationLifetime.Scope
    with pytest.raises(Exception, match="AllocationLifetime.Explicit"):
        sdfg.validate()

def test_validation_fails_new_arr_in_both_alloc_and_reuse():
    sdfg, e = _make_two_array_sdfg()
    e.data.alloc.append('B')   # B is already in reuse — double allocation
    with pytest.raises(Exception, match="alloc and reuse"):
        sdfg.validate()

def test_validation_fails_donor_in_both_reuse_and_free():
    sdfg, e = _make_two_array_sdfg()
    e.data.free.append('A')    # A is already donor in reuse — ownership conflict
    with pytest.raises(Exception, match="reuse.*donor.*free|free.*reuse"):
        sdfg.validate()

from dace.libraries.allocation import _apply_reuse

def _make_sequential_sdfg():
    """
    SDFG:  init --[e0]--> use_A --[e1]--> use_B --[e2]--> done
    A and B are both float64[10] transients with default (Scope) lifetime.
    use_A has an AccessNode for A.
    use_B has an AccessNode for B.
    A and B have the same shape/dtype — a valid reuse pair.
    """
    sdfg = dace.SDFG('seq_test')
    sdfg.add_array('A', [10], dace.float64, transient=True)
    sdfg.add_array('B', [10], dace.float64, transient=True)

    init  = sdfg.add_state('init', is_start_block=True)
    use_A = sdfg.add_state('use_A')
    use_B = sdfg.add_state('use_B')
    done  = sdfg.add_state('done')

    sdfg.add_edge(init,  use_A, InterstateEdge())
    sdfg.add_edge(use_A, use_B, InterstateEdge())
    sdfg.add_edge(use_B, done,  InterstateEdge())

    # Add access nodes connected via tasklets so they are not isolated
    an_A = use_A.add_access('A')
    t_A = use_A.add_tasklet('write_A', {}, {'out'}, 'out = 0.0')
    use_A.add_edge(t_A, 'out', an_A, None, dace.Memlet('A[0]'))

    an_B = use_B.add_access('B')
    t_B = use_B.add_tasklet('write_B', {}, {'out'}, 'out = 0.0')
    use_B.add_edge(t_B, 'out', an_B, None, dace.Memlet('B[0]'))

    return sdfg

def test_apply_reuse_sets_explicit_lifetime():
    sdfg = _make_sequential_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    assert sdfg.arrays['A'].lifetime == dtypes.AllocationLifetime.Explicit
    assert sdfg.arrays['B'].lifetime == dtypes.AllocationLifetime.Explicit

def test_apply_reuse_donor_has_alloc_not_free():
    sdfg = _make_sequential_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    all_allocs = [n for e in sdfg.edges() for n in e.data.alloc]
    all_frees  = [n for e in sdfg.edges() for n in e.data.free]
    assert 'A' in all_allocs
    assert 'A' not in all_frees   # donor is never freed — new_arr takes over

def test_apply_reuse_new_arr_has_reuse_not_alloc():
    sdfg = _make_sequential_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    all_allocs = [n for e in sdfg.edges() for n in e.data.alloc]
    all_reuses = [pair for e in sdfg.edges() for pair in e.data.reuse]
    assert 'B' not in all_allocs
    assert ['B', 'A'] in all_reuses

def test_apply_reuse_new_arr_still_freed():
    sdfg = _make_sequential_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    all_frees = [n for e in sdfg.edges() for n in e.data.free]
    assert 'B' in all_frees

def test_apply_reuse_sdfg_validates():
    sdfg = _make_sequential_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    sdfg.validate()  # must not raise

def test_apply_reuse_error_unknown_array():
    sdfg = _make_sequential_sdfg()
    with pytest.raises(ValueError, match="not found"):
        _apply_reuse(sdfg, 'NOPE', 'A')

def test_apply_reuse_error_non_transient():
    sdfg = dace.SDFG('err')
    sdfg.add_array('inp', [10], dace.float64, transient=False)
    sdfg.add_array('buf', [10], dace.float64, transient=True)
    s = sdfg.add_state('s', is_start_block=True)
    with pytest.raises(ValueError, match="transient"):
        _apply_reuse(sdfg, 'buf', 'inp')

import numpy as np

def _build_fill_read_sdfg():
    """
    SDFG:  init --> fill_A --> read_B --> done
    fill_A writes A[i] = i (for i in 0..9).
    read_B sums B into out[0].
    After _apply_reuse(sdfg, 'B', 'A'), B holds A's data -> out[0] = 45.
    """
    sdfg = dace.SDFG('fill_read')
    sdfg.add_array('A',   [10], dace.float64, transient=True)
    sdfg.add_array('B',   [10], dace.float64, transient=True)
    sdfg.add_array('out', [1],  dace.float64, transient=False)

    init   = sdfg.add_state('init',   is_start_block=True)
    fill_A = sdfg.add_state('fill_A')
    read_B = sdfg.add_state('read_B')
    done   = sdfg.add_state('done')
    sdfg.add_edge(init,   fill_A, InterstateEdge())
    sdfg.add_edge(fill_A, read_B, InterstateEdge())
    sdfg.add_edge(read_B, done,   InterstateEdge())

    # fill_A state: A[i] = (double)i
    me, mx  = fill_A.add_map('fill', {'i': '0:10'})
    t_fill  = fill_A.add_tasklet('fill_t', {}, {'a'}, 'a = (double)i;',
                                  language=dace.Language.CPP)
    a_write = fill_A.add_write('A')
    fill_A.add_edge(me, None, t_fill, None, dace.Memlet())
    fill_A.add_memlet_path(t_fill, mx, a_write,
                           src_conn='a', memlet=dace.Memlet('A[i]'))

    # read_B state: out[0] = sum(B[i])
    me2, mx2 = read_B.add_map('sum', {'i': '0:10'})
    t_sum    = read_B.add_tasklet('sum_t', {'b'}, {'o'}, 'o = b;',
                                   language=dace.Language.CPP)
    b_read   = read_B.add_read('B')
    out_acc  = read_B.add_write('out')
    read_B.add_memlet_path(b_read, me2, t_sum,
                           dst_conn='b', memlet=dace.Memlet('B[i]'))
    read_B.add_memlet_path(t_sum, mx2, out_acc,
                           src_conn='o',
                           memlet=dace.Memlet('out[0]', wcr='lambda a, b: a + b'))

    return sdfg

def test_cpp_contains_reuse_pointer_assignment():
    sdfg = _build_fill_read_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    cpp = sdfg.generate_code()[0].clean_code
    # A must be allocated
    assert '__state->__0_A = new double' in cpp
    # B must NOT be freshly allocated -- it reuses A's pointer
    assert '__state->__0_B = new double' not in cpp
    # B must receive A's pointer
    assert '__state->__0_B = __state->__0_A' in cpp
    # A must be nulled out
    assert '__state->__0_A = nullptr' in cpp
    # B must be freed (takes over A's memory)
    assert 'delete[] __state->__0_B' in cpp
    # A must NOT be freed
    assert 'delete[] __state->__0_A' not in cpp

def test_compile_run_simple_reuse():
    sdfg = _build_fill_read_sdfg()
    _apply_reuse(sdfg, 'B', 'A')
    sdfg.validate()

    out = np.zeros(1, dtype=np.float64)
    csdfg = sdfg.compile()
    csdfg(out=out)

    expected = 10 * 9 / 2   # 0+1+...+9 = 45
    assert out[0] == expected, f"Expected {expected}, got {out[0]}"
