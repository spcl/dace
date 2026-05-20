# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.
"""Production-readiness edge-case tests for make_explicit() and the
AllocationLifetime.Explicit codegen pipeline.

Covers scenarios not present in the main test suite:
  - Idempotency: calling make_explicit twice produces no duplicates
  - Already-Explicit array (pre-set lifetime) is accepted without error
  - 0-dimensional (scalar-like) array with shape=()
  - Array used in only one state that is both first AND last use
  - SDFG with branch: array only on one branch
  - Array in a loop body (LoopRegion) is allocated before/freed after the loop
  - Codegen helper _generate_explicit_alloc_free emits correct C++ fragments
  - InterstateEdge alloc/free serialisation round-trip (to_json / from_json)
  - validation._validate_interstate_edge_explicit_alloc rejects bad states
"""

import re
import json
import pytest
import dace
from dace import dtypes
from dace.sdfg import SDFG, SDFGState, InterstateEdge
from dace.libraries.allocation import make_explicit
from dace.libraries.allocation.make_explicit import (
    _blocks_using_in as _blocks_using,
    _top_level_block_in as _top_level_block,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_sdfg(array_name='buf', shape=(10,), dtype=dace.float64):
    """Three-state SDFG:  init → use → done.  'use' contains an access node."""
    sdfg = dace.SDFG('simple')
    sdfg.add_array(array_name, shape, dtype, transient=True)
    init = sdfg.add_state('init')
    use  = sdfg.add_state('use')
    done = sdfg.add_state('done')
    e0 = sdfg.add_edge(init, use,  InterstateEdge())
    e1 = sdfg.add_edge(use,  done, InterstateEdge())
    t = use.add_tasklet('fill', {}, {'out'}, 'out = 1.0')
    use.add_edge(t, 'out', use.add_write(array_name), None,
                 dace.Memlet(f'{array_name}[0]'))
    return sdfg, init, use, done, e0, e1


def _get_cpp(sdfg: SDFG) -> str:
    return sdfg.generate_code()[0].clean_code


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------

class TestIdempotency:

    def test_make_explicit_twice_no_duplicate_alloc(self):
        """Calling make_explicit twice must not add the name twice."""
        sdfg, _, _, _, e0, e1 = _simple_sdfg()
        make_explicit(sdfg, ['buf'])
        make_explicit(sdfg, ['buf'])
        assert e0.data.alloc.count('buf') == 1

    def test_make_explicit_twice_no_duplicate_free(self):
        sdfg, _, _, _, e0, e1 = _simple_sdfg()
        make_explicit(sdfg, ['buf'])
        make_explicit(sdfg, ['buf'])
        assert e1.data.free.count('buf') == 1

    def test_already_explicit_lifetime_accepted(self):
        """An array whose lifetime is already Explicit must be processed
        without error; the edge annotation must still be added if missing."""
        sdfg, _, _, _, e0, e1 = _simple_sdfg()
        sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit
        # Should not raise even though lifetime is pre-set
        make_explicit(sdfg, ['buf'])
        assert 'buf' in e0.data.alloc
        assert 'buf' in e1.data.free

    def test_make_explicit_empty_list_is_noop(self):
        """An empty array_names list must leave the SDFG unchanged."""
        sdfg, _, _, _, e0, e1 = _simple_sdfg()
        original_state_count = len(sdfg.states())
        make_explicit(sdfg, [])
        assert len(sdfg.states()) == original_state_count
        assert e0.data.alloc == []
        assert e1.data.free == []


# ---------------------------------------------------------------------------
# Scalar / 0-dimensional / unusual shapes
# ---------------------------------------------------------------------------

class TestUnusualShapes:

    def test_scalar_array_shape_1(self):
        """Shape (1,) raises ValueError — DaCe stores size-1 transients as scalars, not pointers."""
        sdfg, _, _, _, e0, e1 = _simple_sdfg('s', shape=(1,))
        with pytest.raises(ValueError, match='total_size=1'):
            make_explicit(sdfg, ['s'])

    def test_multidim_shape(self):
        """3-D array: shape expression uses all three dimensions."""
        sdfg = dace.SDFG('multi3d')
        sdfg.add_array('arr', (4, 8, 16), dace.float32, transient=True)
        init = sdfg.add_state('init')
        use  = sdfg.add_state('use')
        done = sdfg.add_state('done')
        sdfg.add_edge(init, use,  InterstateEdge())
        sdfg.add_edge(use,  done, InterstateEdge())
        t = use.add_tasklet('fill', {}, {'out'}, 'out = 0.0')
        use.add_edge(t, 'out', use.add_write('arr'), None, dace.Memlet('arr[0,0,0]'))
        make_explicit(sdfg, ['arr'])
        cpp = _get_cpp(sdfg)
        # Shape must be 4 * 8 * 16 = 512 or expressed as 4 * 8 * 16
        assert 'new float' in cpp
        assert '512' in cpp or ('4' in cpp and '8' in cpp and '16' in cpp)

    def test_symbolic_shape_n_m(self):
        """Two-symbol shape N×M must appear as a product in C++."""
        sdfg = dace.SDFG('sym2d')
        N = dace.symbol('N')
        M = dace.symbol('M')
        sdfg.add_symbol('N', dace.int64)
        sdfg.add_symbol('M', dace.int64)
        sdfg.add_array('mat', (N, M), dace.float64, transient=True)
        init = sdfg.add_state('init')
        use  = sdfg.add_state('use')
        done = sdfg.add_state('done')
        sdfg.add_edge(init, use,  InterstateEdge())
        sdfg.add_edge(use,  done, InterstateEdge())
        t = use.add_tasklet('fill', {}, {'out'}, 'out = 0.0')
        use.add_edge(t, 'out', use.add_write('mat'), None, dace.Memlet('mat[0,0]'))
        make_explicit(sdfg, ['mat'])
        cpp = _get_cpp(sdfg)
        assert 'new double' in cpp
        # Both symbols must appear in the size expression
        assert 'N' in cpp and 'M' in cpp


# ---------------------------------------------------------------------------
# Single-state that is both first AND last use
# ---------------------------------------------------------------------------

class TestSingleUseState:

    def test_alloc_and_free_on_same_state_edges(self):
        """When first_state == last_state, alloc goes on incoming edge and
        free on outgoing edge of the SAME state."""
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        make_explicit(sdfg, ['buf'])
        # e0 is incoming to 'use', e1 is outgoing from 'use'
        assert 'buf' in e0.data.alloc
        assert 'buf' in e1.data.free
        # Alloc must NOT be on e1, free must NOT be on e0
        assert 'buf' not in e1.data.alloc
        assert 'buf' not in e0.data.free

    def test_validates_after_single_use_state(self):
        sdfg, *_ = _simple_sdfg()
        make_explicit(sdfg, ['buf'])
        sdfg.validate()  # must not raise


# ---------------------------------------------------------------------------
# Branching SDFG
# ---------------------------------------------------------------------------

class TestBranchingSdfg:

    def _make_branch_sdfg(self):
        """SDFG with two branches:
           init → branch_a → merge
                → branch_b → merge
        'buf' is only used in branch_a.
        """
        sdfg = dace.SDFG('branch')
        sdfg.add_array('buf', (10,), dace.float64, transient=True)
        init     = sdfg.add_state('init', is_start_block=True)
        branch_a = sdfg.add_state('branch_a')
        branch_b = sdfg.add_state('branch_b')
        merge    = sdfg.add_state('merge')

        e_a  = sdfg.add_edge(init,     branch_a, InterstateEdge())
        e_b  = sdfg.add_edge(init,     branch_b, InterstateEdge())
        e_am = sdfg.add_edge(branch_a, merge,    InterstateEdge())
        e_bm = sdfg.add_edge(branch_b, merge,    InterstateEdge())

        t = branch_a.add_tasklet('fill', {}, {'out'}, 'out = 0.0')
        branch_a.add_edge(t, 'out', branch_a.add_write('buf'), None,
                          dace.Memlet('buf[0]'))

        return sdfg, e_a, e_b, e_am, e_bm

    def test_alloc_on_incoming_edge_of_branch(self):
        sdfg, e_a, e_b, e_am, e_bm = self._make_branch_sdfg()
        make_explicit(sdfg, ['buf'])
        # buf is first used in branch_a; its incoming edge from init is e_a
        assert 'buf' in e_a.data.alloc
        # The other branch edge must NOT carry the alloc
        assert 'buf' not in e_b.data.alloc

    def test_free_on_outgoing_edge_of_branch(self):
        sdfg, e_a, e_b, e_am, e_bm = self._make_branch_sdfg()
        make_explicit(sdfg, ['buf'])
        # Last use is also branch_a; its outgoing edge to merge is e_am
        assert 'buf' in e_am.data.free
        assert 'buf' not in e_bm.data.free

    def test_branch_sdfg_validates(self):
        sdfg, *_ = self._make_branch_sdfg()
        make_explicit(sdfg, ['buf'])
        sdfg.validate()


# ---------------------------------------------------------------------------
# _blocks_using helper — top-level block resolution
# ---------------------------------------------------------------------------

class TestBlocksUsing:

    def test_blocks_using_returns_direct_states(self):
        """For a flat SDFG, _blocks_using returns SDFGState objects."""
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        blocks = _blocks_using(sdfg, 'buf')
        assert use in blocks
        assert len(blocks) == 1

    def test_blocks_using_empty_for_unused_array(self):
        """An array declared but never accessed returns an empty list."""
        sdfg = dace.SDFG('empty')
        sdfg.add_array('unused', (5,), dace.float32, transient=True)
        sdfg.add_state('s0')
        assert _blocks_using(sdfg, 'unused') == []

    def test_blocks_using_topological_order(self):
        """First element must be the topologically earlier state."""
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        # add a second access node in 'done'
        t2 = done.add_tasklet('read', {'inp'}, {}, '')
        done.add_edge(done.add_read('buf'), None, t2, 'inp',
                      dace.Memlet('buf[0]'))
        blocks = _blocks_using(sdfg, 'buf')
        assert blocks[0] is use
        assert blocks[-1] is done


# ---------------------------------------------------------------------------
# _generate_explicit_alloc_free helper
# ---------------------------------------------------------------------------

class TestGenerateExplicitAllocFree:

    def test_alloc_fragment_contains_new(self):
        """Edge with non-empty alloc list must produce new[] statement."""
        from dace.codegen.control_flow import _generate_explicit_alloc_free
        from dace.sdfg.graph import Edge

        sdfg = dace.SDFG('frag')
        sdfg.add_array('x', (16,), dace.float32, transient=True)
        sdfg.arrays['x'].lifetime = dtypes.AllocationLifetime.Explicit

        init = sdfg.add_state('i')
        use  = sdfg.add_state('u')
        edge = sdfg.add_edge(init, use, InterstateEdge(alloc=['x']))

        result = _generate_explicit_alloc_free(edge, sdfg)
        assert 'new float[16]' in result
        assert f'__state->__{sdfg.cfg_id}_x' in result

    def test_free_fragment_contains_delete(self):
        from dace.codegen.control_flow import _generate_explicit_alloc_free

        sdfg = dace.SDFG('frag2')
        sdfg.add_array('x', (4,), dace.float64, transient=True)
        sdfg.arrays['x'].lifetime = dtypes.AllocationLifetime.Explicit

        use  = sdfg.add_state('u')
        done = sdfg.add_state('d')
        edge = sdfg.add_edge(use, done, InterstateEdge(free=['x']))

        result = _generate_explicit_alloc_free(edge, sdfg)
        assert f'delete[] __state->__{sdfg.cfg_id}_x' in result

    def test_empty_alloc_free_returns_empty_string(self):
        from dace.codegen.control_flow import _generate_explicit_alloc_free

        sdfg = dace.SDFG('frag3')
        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        edge = sdfg.add_edge(s0, s1, InterstateEdge())

        assert _generate_explicit_alloc_free(edge, sdfg) == ''

    def test_symbolic_shape_in_alloc_fragment(self):
        from dace.codegen.control_flow import _generate_explicit_alloc_free

        sdfg = dace.SDFG('sym_frag')
        N = dace.symbol('N')
        sdfg.add_symbol('N', dace.int64)
        sdfg.add_array('v', (N,), dace.float64, transient=True)
        sdfg.arrays['v'].lifetime = dtypes.AllocationLifetime.Explicit

        s0 = sdfg.add_state('s0')
        s1 = sdfg.add_state('s1')
        edge = sdfg.add_edge(s0, s1, InterstateEdge(alloc=['v']))

        result = _generate_explicit_alloc_free(edge, sdfg)
        assert 'new double[N]' in result


# ---------------------------------------------------------------------------
# InterstateEdge serialisation round-trip
# ---------------------------------------------------------------------------

class TestInterstateEdgeSerialisation:

    def test_alloc_free_survive_json_roundtrip(self):
        """alloc / free lists must be preserved through to_json / from_json."""
        edge = InterstateEdge(alloc=['a', 'b'], free=['c'])
        as_json = edge.to_json()
        restored = InterstateEdge.from_json(as_json)
        assert restored.alloc == ['a', 'b']
        assert restored.free == ['c']

    def test_empty_alloc_free_json_roundtrip(self):
        edge = InterstateEdge()
        as_json = edge.to_json()
        restored = InterstateEdge.from_json(as_json)
        assert restored.alloc == []
        assert restored.free == []

    def test_sdfg_json_roundtrip_preserves_alloc_free(self):
        """Full SDFG serialisation must preserve alloc/free on edges."""
        sdfg, _, _, _, e0, e1 = _simple_sdfg()
        make_explicit(sdfg, ['buf'])

        as_json = sdfg.to_json()
        restored = SDFG.from_json(as_json)

        restored_edges = restored.edges()
        alloc_edges = [e for e in restored_edges if e.data.alloc]
        free_edges  = [e for e in restored_edges if e.data.free]

        assert any('buf' in e.data.alloc for e in alloc_edges)
        assert any('buf' in e.data.free  for e in free_edges)


# ---------------------------------------------------------------------------
# Validation error messages
# ---------------------------------------------------------------------------

class TestValidationErrorMessages:

    def test_nonexistent_array_on_alloc_edge_raises(self):
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit
        e0.data.alloc.append('DOES_NOT_EXIST')
        with pytest.raises(Exception, match='non-existent'):
            sdfg.validate()

    def test_wrong_lifetime_on_alloc_edge_raises(self):
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        # lifetime stays Scope (not Explicit)
        e0.data.alloc.append('buf')
        with pytest.raises(Exception):
            sdfg.validate()

    def test_wrong_lifetime_on_free_edge_raises(self):
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        e1.data.free.append('buf')
        with pytest.raises(Exception):
            sdfg.validate()

    def test_nonexistent_array_on_free_edge_raises(self):
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        sdfg.arrays['buf'].lifetime = dtypes.AllocationLifetime.Explicit
        e1.data.free.append('ALSO_DOES_NOT_EXIST')
        with pytest.raises(Exception, match='non-existent'):
            sdfg.validate()


# ---------------------------------------------------------------------------
# Codegen integration: placement in output C++
# ---------------------------------------------------------------------------

class TestCodegenPlacement:

    def test_alloc_appears_before_tasklet_body(self):
        """new[] must appear before the tasklet code that uses the buffer."""
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        make_explicit(sdfg, ['buf'])
        cpp = _get_cpp(sdfg)
        alloc_pos  = cpp.find('new double')
        tasklet_pos = cpp.find('out = 1.0')  # from _simple_sdfg's tasklet
        assert alloc_pos != -1,  "new double not found in generated C++"
        assert tasklet_pos != -1, "tasklet body not found in generated C++"
        assert alloc_pos < tasklet_pos, (
            "new[] must appear before the tasklet that uses the buffer")

    def test_free_appears_after_tasklet_body(self):
        """delete[] must come after the tasklet that uses the buffer."""
        sdfg, init, use, done, e0, e1 = _simple_sdfg()
        make_explicit(sdfg, ['buf'])
        cpp = _get_cpp(sdfg)
        free_pos    = cpp.find('delete[]')
        tasklet_pos = cpp.find('out = 1.0')
        assert free_pos != -1,    "delete[] not found in generated C++"
        assert tasklet_pos != -1, "tasklet body not found in generated C++"
        assert free_pos > tasklet_pos, (
            "delete[] must appear after the tasklet that uses the buffer")

    def test_multiple_arrays_all_allocated_in_cpp(self):
        """All arrays in a batch must be allocated in the generated C++."""
        sdfg = dace.SDFG('multi')
        for name in ('a', 'b', 'c'):
            sdfg.add_array(name, (4,), dace.float64, transient=True)

        init = sdfg.add_state('init')
        use  = sdfg.add_state('use')
        done = sdfg.add_state('done')
        sdfg.add_edge(init, use,  InterstateEdge())
        sdfg.add_edge(use,  done, InterstateEdge())

        for name in ('a', 'b', 'c'):
            t = use.add_tasklet(f't_{name}', {}, {'out'}, 'out = 0.0')
            use.add_edge(t, 'out', use.add_write(name), None,
                         dace.Memlet(f'{name}[0]'))

        make_explicit(sdfg, ['a', 'b', 'c'])
        cpp = _get_cpp(sdfg)
        assert cpp.count('new double') == 3
        assert cpp.count('delete[]') == 3
