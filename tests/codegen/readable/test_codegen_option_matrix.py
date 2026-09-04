# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Pins the INVARIANT the whole ``compiler.cpu.codegen_params.*`` matrix (plus
``compiler.cpu.implementation`` itself) promises: every knob is a SPELLING choice over the identical
program, never a SEMANTIC one. Where the rest of this directory (``test_codegen_style_flags.py``,
``test_index_ctype.py``, ``test_ptr_increment.py``, ...) each pin ONE knob's own emitted TEXT, this
module parametrizes over the FULL option matrix -- default plus one deviation per knob -- and requires
every arm to COMPILE, RUN, and reproduce the exact same numbers as a numpy reference. A knob that
silently changes the answer is a miscompile, and until now only a throwaway shell sweep caught it.

THE KERNEL (:func:`option_matrix_sdfg`) is a single, hand-built SDFG (not a ``@dace.program`` -- several
arms need explicit control the frontend does not expose: a forced SEQUENTIAL map schedule for
``loop_decl_style``/``loop_access_form``, and ``no_inline`` nested SDFGs for the split/inline knobs)
that in one pass touches: a symbolic-size heap transient (``heap_ptr_restrict``), a write-once constant
scalar (``const_init``) and a separate genuinely MUTABLE, twice-reassigned scalar
(``scalar_init_style``/``decl_placement``) feeding a write-once length-1 array, two SEQUENTIAL strided maps
(``loop_index_type``/``loop_bound_cmp``/
``loop_decl_style``/``loop_access_form``/``index_ctype``), a direct array-to-array copy
(``explicit_copy``), and TWO separate top-level ``no_inline`` nested SDFGs: one with ONLY full-array
connectors (``split_nsdfg_translation_units``/``external_translation_units``/
``inline_full_array_nsdfg``) and one with a READ-ONLY SCALAR connector (``const_scalar_abi`` --
per ``inline_full_array_nsdfg``'s own doc, a nest with any scalar connector never qualifies for that
inlining, so it has to be a separate nest).

Every arithmetic step is a small-integer multiply/add/subtract, so every intermediate value stays
exact in float64 -- every arm is checked with ``np.array_equal``, never a tolerance.

DEGENERATE ARMS. Several entries in the requested matrix are NOT distinct settable values: the
implementation only special-cases one literal string per key, and these are OTHER terms borrowed from
that key's own schema prose (an internal classification label, or an unrelated flag) rather than a
value the code branches on. Each is confirmed by reading the branch in question and is pinned below as
its OWN regression test rather than silently "fixed":
  - ``index_fn_qualifier = 'inline'``     -- only 'always_inline' is special-cased; byte-identical to
                                             the default ('inline_constexpr').
  - ``decl_placement = 'inverted'``       -- only 'late' is special-cased; byte-identical to 'eager'
                                             ('inverted' names a LOOP shape in this key's own prose).
  - ``scalar_init_style = 'const'/'setzero'`` -- only 'fused' is special-cased; both fall through to
                                             'split' ('setzero' names a node property, not a value).
  - ``const_init = 'const'/'constexpr_static'`` -- only 'on' is special-cased; both behave like 'off'
                                             (both are ``const_init_kind`` classification labels).
  - ``heap_ptr_restrict = 'may_alias'``   -- only 'restrict' is special-cased; behaves like 'none'
                                             ('may_alias' is the ARRAY-DESCRIPTOR flag this key reads,
                                             not a value of the key itself).
Each requested value is still exercised in the main compile+run matrix below (nothing is dropped), and
each degenerate one additionally gets a dedicated codegen-only test documenting what it actually does.

FORMERLY FAILING ARM. ``inline_full_array_nsdfg = True`` used to fail to COMPILE on this kernel:
inlining rewrites the nest's ``io[io_idx(k)]`` to ``B[B_idx(k)]``, assuming a ``B_idx`` helper
exists for the outer array, and back when the ``heap -> B`` copy lowered to a single ``std::memcpy``
``B`` was never subscripted at the frame's top level, so no such helper was ever emitted. That copy
is now the canonical parallel element map, which subscripts ``B`` in the frame and therefore emits
``B_idx``; the arm compiles and matches. The underlying codegen gap (an array reached only by
pointer still needs its index helper when a nest is inlined onto it) is no longer REACHABLE from
this kernel -- the arm stays in the matrix as the regression guard for the shape it does cover.
"""
import contextlib
import re
from typing import Dict, Optional, Tuple

import numpy as np
import pytest

import dace
from dace import dtypes
from dace.config import set_temporary

from tests.codegen.readable.conftest import run_isolated

N = dace.symbol('N')
SIZE = 32  # even (the two strided maps below split 0:N:2 / 1:N:2 to cover it exactly)


def option_matrix_sdfg(name: str) -> dace.SDFG:
    """Builds the fixture SDFG described in the module docstring. ``name`` must be UNIQUE per build
    (each config arm gets its own build-cache entry) except when two calls are deliberately meant to be
    byte-comparable, in which case pass the SAME name to both (the name is baked into the output)."""
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_scalar('s', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    sdfg.add_scalar('acc', dace.float64, transient=True, storage=dtypes.StorageType.Register)
    sdfg.add_array('buf', [1], dace.float64, transient=True)
    sdfg.add_array('heap', [N], dace.float64, transient=True)  # symbolic size -> heap allocation

    # Write-once constant scalar `s` (const_init).
    init_s = sdfg.add_state('init_s', is_start_block=True)
    ts = init_s.add_tasklet('set_s', {}, {'o': None}, 'o = 3.0')
    init_s.add_edge(ts, 'o', init_s.add_access('s'), None, dace.Memlet('s[0]'))

    # A genuinely MUTABLE, twice-reassigned scalar `acc`, entirely within this ONE state/scope (the
    # gate scalar_init_style/decl_placement require), feeding a write-once length-1 array `buf`.
    init_buf = sdfg.add_state_after(init_s, 'init_buf')
    ra0 = init_buf.add_read('A')
    acc0 = init_buf.add_access('acc')
    t_acc0 = init_buf.add_tasklet('acc0', {'a': None}, {'o': None}, 'o = a + 1.0')
    init_buf.add_edge(ra0, None, t_acc0, 'a', dace.Memlet('A[0]'))
    init_buf.add_edge(t_acc0, 'o', acc0, None, dace.Memlet('acc[0]'))

    ra1 = init_buf.add_read('A')
    acc1 = init_buf.add_access('acc')
    t_acc1 = init_buf.add_tasklet('acc1', {'a': None, 'x': None}, {'o': None}, 'o = a + x')
    init_buf.add_edge(acc0, None, t_acc1, 'a', dace.Memlet('acc[0]'))
    init_buf.add_edge(ra1, None, t_acc1, 'x', dace.Memlet('A[1]'))
    init_buf.add_edge(t_acc1, 'o', acc1, None, dace.Memlet('acc[0]'))

    rs = init_buf.add_read('s')
    t_buf = init_buf.add_tasklet('mkbuf', {'sc': None, 'ac': None}, {'o': None}, 'o = sc * 2.0 + ac')
    init_buf.add_edge(rs, None, t_buf, 'sc', dace.Memlet('s[0]'))
    init_buf.add_edge(acc1, None, t_buf, 'ac', dace.Memlet('acc[0]'))
    init_buf.add_edge(t_buf, 'o', init_buf.add_access('buf'), None, dace.Memlet('buf[0]'))

    # Two SEQUENTIAL, strided (non-unit-stride) maps: loop_index_type / loop_bound_cmp /
    # loop_decl_style / loop_access_form all gate on SEQUENTIAL, so an OpenMP map would leave them
    # vacuous. Together 0:N:2 and 1:N:2 cover the whole array (SIZE is even).
    compute = sdfg.add_state_after(init_buf, 'compute')
    for parity, sign, label in ((0, '+', 'even'), (1, '-', 'odd')):
        ra = compute.add_read('A')
        rs2 = compute.add_read('s')
        rb = compute.add_read('buf')
        wh = compute.add_write('heap')
        entry, exit_node = compute.add_map(f'fill_{label}', {'i': f'{parity}:N:2'},
                                           schedule=dtypes.ScheduleType.Sequential)
        t = compute.add_tasklet(f't_{label}', {'a': None, 'sc': None, 'b': None}, {'o': None}, 'o = a * sc %s b' % sign)
        compute.add_memlet_path(ra, entry, t, dst_conn='a', memlet=dace.Memlet('A[i]'))
        compute.add_memlet_path(rs2, entry, t, dst_conn='sc', memlet=dace.Memlet('s[0]'))
        compute.add_memlet_path(rb, entry, t, dst_conn='b', memlet=dace.Memlet('buf[0]'))
        compute.add_memlet_path(t, exit_node, wh, src_conn='o', memlet=dace.Memlet('heap[i]'))

    # Direct array-to-array copy, no map (explicit_copy: lifts this to a CopyLibraryNode).
    copy_state = sdfg.add_state_after(compute, 'copy')
    copy_state.add_edge(copy_state.add_read('heap'), None, copy_state.add_write('B'), None, dace.Memlet('heap[0:N]'))

    # Nested SDFG #1: ONLY a full-array connector (offset 0, whole range, matching shape/strides) --
    # split_nsdfg_translation_units / external_translation_units / inline_full_array_nsdfg.
    inner = dace.SDFG(f'{name}_inner')
    inner.add_array('io', [N], dace.float64)
    ist = inner.add_state('scale', is_start_block=True)
    ime, imx = ist.add_map('scale_m', {'k': '0:N'})
    it = ist.add_tasklet('scale_t', {'v': None}, {'w': None}, 'w = v * 2.0')
    ist.add_memlet_path(ist.add_read('io'), ime, it, dst_conn='v', memlet=dace.Memlet('io[k]'))
    ist.add_memlet_path(it, imx, ist.add_write('io'), src_conn='w', memlet=dace.Memlet('io[k]'))

    nest_state = sdfg.add_state_after(copy_state, 'nested_call')
    nn = nest_state.add_nested_sdfg(inner, {'io': None}, {'io': None}, symbol_mapping={'N': N})
    nn.no_inline = True  # survive the readable pipeline's blanket InlineSDFG sweep
    nest_state.add_edge(nest_state.add_read('B'), None, nn, 'io', dace.Memlet('B[0:N]'))
    nest_state.add_edge(nn, 'io', nest_state.add_write('B'), None, dace.Memlet('B[0:N]'))

    # Nested SDFG #2: a full-array connector PLUS a READ-ONLY SCALAR connector (const_scalar_abi).
    # Kept separate from #1 -- a scalar connector disqualifies a nest from inline_full_array_nsdfg.
    inner2 = dace.SDFG(f'{name}_inner2')
    inner2.add_array('io', [N], dace.float64)
    inner2.add_scalar('sc', dace.float64)
    ist2 = inner2.add_state('addsc', is_start_block=True)
    ime2, imx2 = ist2.add_map('addsc_m', {'j': '0:N'})
    it2 = ist2.add_tasklet('addsc_t', {'v': None, 'c': None}, {'w': None}, 'w = v + c')
    ist2.add_memlet_path(ist2.add_read('io'), ime2, it2, dst_conn='v', memlet=dace.Memlet('io[j]'))
    ist2.add_memlet_path(ist2.add_read('sc'), ime2, it2, dst_conn='c', memlet=dace.Memlet('sc[0]'))
    ist2.add_memlet_path(it2, imx2, ist2.add_write('io'), src_conn='w', memlet=dace.Memlet('io[j]'))

    nest_state2 = sdfg.add_state_after(nest_state, 'nested_call_2')
    nn2 = nest_state2.add_nested_sdfg(inner2, {'io': None, 'sc': None}, {'io': None}, symbol_mapping={'N': N})
    nn2.no_inline = True
    nest_state2.add_edge(nest_state2.add_read('B'), None, nn2, 'io', dace.Memlet('B[0:N]'))
    nest_state2.add_edge(nest_state2.add_read('s'), None, nn2, 'sc', dace.Memlet('s[0]'))
    nest_state2.add_edge(nn2, 'io', nest_state2.add_write('B'), None, dace.Memlet('B[0:N]'))

    sdfg.validate()
    return sdfg


def reference(A: np.ndarray) -> np.ndarray:
    """Exact numpy translation of :func:`option_matrix_sdfg`'s dataflow (see states above)."""
    s = 3.0
    acc = (A[0] + 1.0) + A[1]
    buf0 = s * 2.0 + acc
    heap = np.empty_like(A)
    heap[0::2] = A[0::2] * s + buf0
    heap[1::2] = A[1::2] * s - buf0
    return heap * 2.0 + s


_INPUT_A = np.random.default_rng(0).integers(-10, 10, size=SIZE).astype(np.float64)
_EXPECTED_B = reference(_INPUT_A)

#: (arm id, config path or None for the default control arm, value). Exactly the requested matrix.
ARMS: Tuple[Tuple[str, Optional[Tuple[str, ...]], object], ...] = (
    ('default', None, None),
    ('impl_legacy', ('compiler', 'cpu', 'implementation'), 'legacy'),
    ('index_ctype_int32', ('compiler', 'cpu', 'codegen_params', 'index_ctype'), 'int32'),
    ('heap_ptr_restrict_none', ('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict'), 'none'),
    ('heap_ptr_restrict_may_alias', ('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict'), 'may_alias'),
    ('index_fn_qualifier_always_inline', ('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier'), 'always_inline'),
    ('index_fn_qualifier_inline', ('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier'), 'inline'),
    ('loop_index_type_int32', ('compiler', 'cpu', 'codegen_params', 'loop_index_type'), 'int32'),
    ('loop_bound_cmp_le', ('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp'), 'le'),
    ('loop_bound_cmp_ne', ('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp'), 'ne'),
    ('loop_decl_style_hoisted', ('compiler', 'cpu', 'codegen_params', 'loop_decl_style'), 'hoisted'),
    ('loop_access_form_ptr_increment', ('compiler', 'cpu', 'codegen_params', 'loop_access_form'), 'ptr_increment'),
    ('decl_placement_late', ('compiler', 'cpu', 'codegen_params', 'decl_placement'), 'late'),
    ('decl_placement_inverted', ('compiler', 'cpu', 'codegen_params', 'decl_placement'), 'inverted'),
    ('scalar_init_style_fused', ('compiler', 'cpu', 'codegen_params', 'scalar_init_style'), 'fused'),
    ('scalar_init_style_const', ('compiler', 'cpu', 'codegen_params', 'scalar_init_style'), 'const'),
    ('scalar_init_style_setzero', ('compiler', 'cpu', 'codegen_params', 'scalar_init_style'), 'setzero'),
    ('const_init_off', ('compiler', 'cpu', 'codegen_params', 'const_init'), 'off'),
    ('const_init_const', ('compiler', 'cpu', 'codegen_params', 'const_init'), 'const'),
    ('const_init_constexpr_static', ('compiler', 'cpu', 'codegen_params', 'const_init'), 'constexpr_static'),
    ('const_scalar_abi_by_value', ('compiler', 'cpu', 'codegen_params', 'const_scalar_abi'), 'by_value'),
    ('split_nsdfg_translation_units', ('compiler', 'cpu', 'codegen_params', 'split_nsdfg_translation_units'), True),
    ('external_translation_units', ('compiler', 'cpu', 'codegen_params', 'external_translation_units'), True),
    # Formerly broken (see the module docstring): inlining rewrites the nest's `io[io_idx(k)]` to
    # `B[B_idx(k)]`, which used to reference an undeclared helper because a `memcpy`'d `B` was never
    # subscripted in the frame. The parallel element copy subscripts it, so `B_idx` is emitted.
    ('inline_full_array_nsdfg', ('compiler', 'cpu', 'codegen_params', 'inline_full_array_nsdfg'), True),
)


def run_arm(arm_id: str, path: Optional[Tuple[str, ...]], value: object) -> np.ndarray:
    """Compile + run one arm in a forked child (a miscompiled arm must not take down pytest)."""

    def work() -> Dict[str, np.ndarray]:
        ctx = set_temporary(*path, value=value) if path is not None else contextlib.nullcontext()
        with ctx:
            csdfg = option_matrix_sdfg(f'optmatrix_{arm_id}').compile()
            B = np.zeros(SIZE, dtype=np.float64)
            csdfg(A=_INPUT_A.copy(), B=B, N=SIZE)
            return {'B': B}

    return run_isolated(work)['B']


@pytest.mark.parametrize('arm_id, path, value', ARMS, ids=[arm[0] for arm in ARMS])
def test_arm_matches_numpy_reference(arm_id: str, path: Optional[Tuple[str, ...]], value: object) -> None:
    """A codegen knob changes the spelling, never the answer. Exact equality: every step in the
    kernel is a small-integer multiply/add/subtract, so nothing here is float-inexact."""
    B = run_arm(arm_id, path, value)
    assert np.array_equal(B, _EXPECTED_B), f'{arm_id}: numeric result diverges from the numpy reference'


# --------------------------------------------------------------------------------------------- #
# Non-vacuity: prove the kernel actually reaches the knobs (>= 3 required; these are 8).
# --------------------------------------------------------------------------------------------- #
def cpp_text(sdfg_name: str, path: Optional[Tuple[str, ...]] = None, value: object = None) -> str:
    """Codegen-only (no compile) C++ text for a FRESH build under (path, value). Pass the SAME
    ``sdfg_name`` to a paired call for a byte-comparable pair -- the name is baked into the output."""
    sdfg = option_matrix_sdfg(sdfg_name)
    if path is None:
        return '\n'.join(o.code for o in sdfg.generate_code() if o.language == 'cpp')
    with set_temporary(*path, value=value):
        return '\n'.join(o.code for o in sdfg.generate_code() if o.language == 'cpp')


def test_index_ctype_int32_changes_the_helper_type() -> None:
    default = cpp_text('nv_index_ctype')
    arm = cpp_text('nv_index_ctype', ('compiler', 'cpu', 'codegen_params', 'index_ctype'), 'int32')
    assert default != arm
    assert 'int64_t A_idx' in default and 'int32_t A_idx' in arm


def test_heap_ptr_restrict_none_drops_restrict() -> None:
    default = cpp_text('nv_heap_restrict')
    arm = cpp_text('nv_heap_restrict', ('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict'), 'none')
    assert default != arm
    assert '__restrict__ heap' in default
    assert '__restrict__ heap' not in arm


def test_explicit_copy_always_lifts_in_readable() -> None:
    """The readable generator lifts ``heap[0:N] -> B`` unconditionally (the ``explicit_copy`` knob
    governs only the classic generator -- see ``tests/codegen/readable/test_explicit_copy.py``).

    The lifted copy lowers to the canonical PARALLEL element map: the maximally parallel form is
    the default on every device, and a symbolic count is assumed big. A single ``std::memcpy`` is
    the CPU SPECIALIZATION of a ``Sequential`` contiguous transfer (``SpecializeCpuTransfers``) --
    this copy is top level and never re-entered, so nothing sequentializes it and no ``memcpy`` is
    emitted here. The ``memcpy``/``memset`` selection itself is pinned in
    ``tests/passes/copy_memset_parallel_selection_test.py`` and
    ``tests/passes/cpu_specialization_fork_join_test.py``, not here."""
    default = cpp_text('nv_explicit_copy')
    routine = re.search(r'inline void copy_heap_to_B_\w+\(.*?\n\}', default, re.DOTALL)
    assert routine is not None, 'the readable generator must emit the lifted heap->B copy routine'
    body = routine.group(0)
    assert '#pragma omp parallel for' in body
    assert re.search(r'_cpy_out\[[^\]]+\] = _cpy_in\[[^\]]+\];', body) is not None
    assert 'dace::CopyND' not in default


def test_const_scalar_abi_by_value_drops_the_reference() -> None:
    default = cpp_text('nv_const_scalar_abi')
    arm = cpp_text('nv_const_scalar_abi', ('compiler', 'cpu', 'codegen_params', 'const_scalar_abi'), 'by_value')
    assert default != arm
    assert 'const double&  sc' in default
    assert 'const double&  sc' not in arm


def test_inline_full_array_nsdfg_inlines_the_full_array_nest_only() -> None:
    """The all-full-array nest (``_inner_``) is inlined away entirely; the nest with a scalar
    connector (``_inner2_``) is NEVER eligible and keeps its function form, per the schema."""
    default = cpp_text('nv_inline_full')
    arm = cpp_text('nv_inline_full', ('compiler', 'cpu', 'codegen_params', 'inline_full_array_nsdfg'), True)
    assert default != arm
    assert '_inner_0_' in default
    assert '_inner_0_' not in arm
    assert '_inner2_0_' in arm


def test_split_nsdfg_translation_units_splits_the_frame() -> None:
    default_objs = [o for o in option_matrix_sdfg('nv_split_default').generate_code() if o.language == 'cpp']
    with set_temporary('compiler', 'cpu', 'codegen_params', 'split_nsdfg_translation_units', value=True):
        split_objs = [o for o in option_matrix_sdfg('nv_split_on').generate_code() if o.language == 'cpp']
    assert len(split_objs) > len(default_objs)


def test_loop_bound_cmp_le_drops_the_plus_one() -> None:
    default = cpp_text('nv_loop_bound_cmp')
    arm = cpp_text('nv_loop_bound_cmp', ('compiler', 'cpu', 'codegen_params', 'loop_bound_cmp'), 'le')
    assert default != arm
    assert any(line.strip().startswith('for (auto i = 0; i <= ') for line in arm.splitlines())


def test_loop_access_form_ptr_increment_walks_the_sequential_maps() -> None:
    default = cpp_text('nv_loop_access_form')
    arm = cpp_text('nv_loop_access_form', ('compiler', 'cpu', 'codegen_params', 'loop_access_form'), 'ptr_increment')
    assert default != arm
    assert '__walk_heap' in arm and '__walk_A' in arm
    assert '__walk_' not in default


# --------------------------------------------------------------------------------------------- #
# Degenerate arms: each requested value below is confirmed (by reading the branch, then verifying
# here) to be indistinguishable from an existing value rather than a genuine third state. Each is
# still exercised for the numeric invariant in ARMS above; these pin what it ACTUALLY does.
# --------------------------------------------------------------------------------------------- #
def test_index_fn_qualifier_inline_is_not_a_recognized_value() -> None:
    """Only the literal 'always_inline' is special-cased; 'inline' takes the same path as leaving the
    key at its default ('inline_constexpr') -- byte-identical output."""
    default = cpp_text('nv_ifq')
    arm = cpp_text('nv_ifq', ('compiler', 'cpu', 'codegen_params', 'index_fn_qualifier'), 'inline')
    assert arm == default


def test_decl_placement_inverted_is_not_a_recognized_value() -> None:
    """Only 'late' is special-cased; 'inverted' -- a LOOP-SHAPE term from this key's own schema prose,
    not a settable value -- takes the same 'eager' path: byte-identical to the default."""
    default = cpp_text('nv_declp')
    arm = cpp_text('nv_declp', ('compiler', 'cpu', 'codegen_params', 'decl_placement'), 'inverted')
    assert arm == default


@pytest.mark.parametrize('value', ['const', 'setzero'])
def test_scalar_init_style_recognizes_only_fused(value: str) -> None:
    """Only 'fused' is special-cased; 'const' and 'setzero' (an emission-time NODE PROPERTY, not a
    settable value of this key) both fall through to the same 'split' default path."""
    default = cpp_text('nv_sis')
    arm = cpp_text('nv_sis', ('compiler', 'cpu', 'codegen_params', 'scalar_init_style'), value)
    assert arm == default


@pytest.mark.parametrize('value', ['const', 'constexpr_static'])
def test_const_init_recognizes_only_on(value: str) -> None:
    """Only the literal 'on' runs MarkConstInit; 'const' and 'constexpr_static' are
    ``const_init_kind`` CLASSIFICATION labels (``dace/data/core.py``), not settable values of this
    key, so both behave like 'off'."""
    off = cpp_text('nv_ci_off', ('compiler', 'cpu', 'codegen_params', 'const_init'), 'off')
    arm = cpp_text('nv_ci_off', ('compiler', 'cpu', 'codegen_params', 'const_init'), value)
    assert arm == off


def test_heap_ptr_restrict_may_alias_behaves_like_none() -> None:
    """Only the literal 'restrict' is special-cased; 'may_alias' is the ARRAY-DESCRIPTOR flag this
    key reads (``desc.may_alias``), not a value of the key itself -- behaves exactly like 'none'."""
    none_arm = cpp_text('nv_hpr', ('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict'), 'none')
    arm = cpp_text('nv_hpr', ('compiler', 'cpu', 'codegen_params', 'heap_ptr_restrict'), 'may_alias')
    assert arm == none_arm


def test_external_translation_units_restructures_a_cpu_only_kernel_too() -> None:
    """Documented as GPU-only, but ``OutlineTopLevelNests`` runs whenever EITHER split flag is set,
    regardless of GPU content: the two sequential maps get wrapped into no_inline nested-SDFG
    functions and the frame's text changes, even though no work is ever routed to a .cu (there is
    none). Not a no-op on a CPU-only kernel -- pinned so it is not mistaken for one."""
    default = cpp_text('nv_ext_tu')
    arm = cpp_text('nv_ext_tu', ('compiler', 'cpu', 'codegen_params', 'external_translation_units'), True)
    assert default != arm


if __name__ == '__main__':
    # A representative slice, not the full (expensive) ARMS matrix -- see the module docstring.
    for _arm_id, _path, _value in ARMS[:4]:
        test_arm_matches_numpy_reference(_arm_id, _path, _value)
    test_index_ctype_int32_changes_the_helper_type()
    test_heap_ptr_restrict_none_drops_restrict()
    test_explicit_copy_always_lifts_in_readable()
    test_const_scalar_abi_by_value_drops_the_reference()
    test_inline_full_array_nsdfg_inlines_the_full_array_nest_only()
    test_split_nsdfg_translation_units_splits_the_frame()
    test_loop_bound_cmp_le_drops_the_plus_one()
    test_loop_access_form_ptr_increment_walks_the_sequential_maps()
    test_index_fn_qualifier_inline_is_not_a_recognized_value()
    test_decl_placement_inverted_is_not_a_recognized_value()
    for _value in ('const', 'setzero'):
        test_scalar_init_style_recognizes_only_fused(_value)
    for _value in ('const', 'constexpr_static'):
        test_const_init_recognizes_only_on(_value)
    test_heap_ptr_restrict_may_alias_behaves_like_none()
    test_external_translation_units_restructures_a_cpu_only_kernel_too()
    print('ok')
