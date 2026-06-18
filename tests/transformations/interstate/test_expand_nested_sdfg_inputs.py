"""Unit tests for :class:`ExpandNestedSDFGInputs`."""
import copy

import dace
import numpy as np

from dace.sdfg import nodes
from dace.transformation.dataflow.map_expansion import MapExpansion
from dace.transformation.dataflow.map_for_loop import MapToForLoop
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.interstate.multistate_inline import InlineMultistateSDFG
from dace.transformation.passes.pattern_matching import PatternMatchAndApplyRepeated

N = dace.symbol('N')
M = dace.symbol('M')
K = 4


@dace.program
def _jacobi2d_map_tile(a: dace.float64[N, M], b: dace.float64[N, M]):
    for ii, jj in dace.map[0:N - 2:K, 0:M - 2:K]:
        for i, j in dace.map[0:K, 0:K]:
            b[ii + i + 1, jj + j + 1] = a[ii + i + 1, jj + j + 1] * 2.0


def _count_nsdfgs(sdfg):
    return sum(1 for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nodes.NestedSDFG))


def _build_map_to_for_loop_test_sdfg():
    """Build the post-MapToForLoop NSDFG-wrapped shape that
    ``ExpandNestedSDFGInputs`` must widen."""
    sdfg = _jacobi2d_map_tile.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([MapToForLoop()]).apply_pass(sdfg, {})
    return sdfg


def test_widens_narrowed_inedges_to_full_array():
    """Every in/out edge of the top-level NSDFG must read the full outer
    array after ``ExpandNestedSDFGInputs``."""
    from dace import subsets
    sdfg = _build_map_to_for_loop_test_sdfg()
    n_before = _count_nsdfgs(sdfg)
    assert n_before > 0
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    # Top-level NSDFGs (those NOT scoped by a Map) must now read the
    # full outer array on every in/out edge.
    for state in sdfg.states():
        for n in state.nodes():
            if not isinstance(n, nodes.NestedSDFG):
                continue
            if state.entry_node(n) is not None:
                continue  # Map-scoped: deliberately not widened
            for e in (*state.in_edges(n), *state.out_edges(n)):
                if e.data is None or e.data.data is None:
                    continue
                full = subsets.Range.from_array(sdfg.arrays[e.data.data])
                assert e.data.subset == full, \
                    f'NSDFG in-edge for {e.data.data!r} should be full {full}; got {e.data.subset}'


def test_apply_preserves_numerics_via_inline():
    """End-to-end: ExpandNestedSDFGInputs followed by InlineMultistateSDFG
    produces a numerically-identical SDFG to the un-modified one.

    Regression (fixed 2026-06-15): the inner-memlet uncollapse step
    (:func:`_rewrite_memlets_with_offset`) indexed the inner subset
    assuming the inner descriptor had DROPPED the length-1 (collapsed)
    dims, so for a full-rank inner descriptor (which modern
    ``NestInnermostMapBodyIntoNSDFG`` produces) the non-collapsed dim
    consumed the collapsed dim's inner begin and silently dropped the
    per-access intra-window offset. jacobi2d then computed only the
    ``j=0`` column of each ``i``-row. The fix indexes the inner subset
    1:1 with the outer dims when the inner memlet is full-rank."""
    sdfg = _build_map_to_for_loop_test_sdfg()
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(sdfg, {})
    sdfg.validate()
    n, m = 10, 10
    rng = np.random.default_rng(0xCAFE)
    a = rng.standard_normal((n, m))
    b = np.zeros((n, m))
    ref = b.copy()
    copy.deepcopy(_jacobi2d_map_tile.to_sdfg(simplify=True))(a=a.copy(), b=ref, N=n, M=m)
    sdfg(a=a, b=b, N=n, M=m)
    assert np.allclose(b, ref), f'max diff: {np.abs(b - ref).max():.3e}'


def test_refuses_inside_map_scope():
    """If the NSDFG is inside a Map scope, the per-iteration narrowing
    is intentional and the transformation must refuse."""

    @dace.program
    def kernel(a: dace.float64[N, N], b: dace.float64[N, N]):
        # The outer Map wraps a NSDFG implicitly via to_sdfg.
        for i in dace.map[0:N]:
            for j in range(N):
                b[i, j] = a[i, j] * 2.0

    sdfg = kernel.to_sdfg(simplify=True)
    # No top-level NSDFGs to widen here, but if there are any nested
    # ones inside the Map scope the pass must leave them alone.
    before = _count_nsdfgs(sdfg)
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    after = _count_nsdfgs(sdfg)
    assert before == after, 'pass must not touch Map-scoped NSDFGs'


def test_introduced_symbol_picks_up_outer_type():
    """When the offset expression introduces a symbol not in the inner
    SDFG's table, the new symbol must inherit the outer SDFG's declared
    type (not silently fall back to ``int64``)."""
    sdfg = _build_map_to_for_loop_test_sdfg()
    # Verify the outer SDFG has at least one symbol of a non-default
    # type that the inner SDFG should pick up.
    outer_syms = set(sdfg.symbols)
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    # Each NSDFG that introduced a new symbol must carry the outer
    # symbol's type.
    for state in sdfg.states():
        for n in state.nodes():
            if not isinstance(n, nodes.NestedSDFG):
                continue
            for sym, mapping in n.symbol_mapping.items():
                if sym not in n.sdfg.symbols:
                    continue
                inner_t = n.sdfg.symbols[sym]
                if sym in sdfg.symbols:
                    assert inner_t == sdfg.symbols[sym], \
                        f'symbol {sym!r}: inner type {inner_t} != outer type {sdfg.symbols[sym]}'


NB = dace.symbol('NB')
NLEV = dace.symbol('NLEV')
NPROMA = dace.symbol('NPROMA')


@dace.program
def _icon_zekinh_gather_kernel(e_bln: dace.float64[NB, 3, NPROMA], edge_idx: dace.int32[NB, NPROMA, 3],
                               edge_blk: dace.int32[NB, NPROMA, 3], z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
                               z_ekinh: dace.float64[NB, NLEV, NPROMA]):
    """Minimal ICON velocity_zekinh-style data-dependent gather kernel.

    The 3-edge bilinear gather (``z_kin_hor_e[edge_blk[..], jk, edge_idx[..]]``)
    is the canonical in-map body NSDFG that the vectorization tile descent
    needs ``ExpandNestedSDFGInputs`` to normalize (widen connector subsets +
    uncollapse inner memlets) before any tile rewriting can run.
    """
    for jb in range(NB):
        for jk in range(NLEV):
            for jc in range(NPROMA):
                z_ekinh[jb, jk, jc] = (e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[jb, jc, 0], jk, edge_idx[jb, jc, 0]])


def test_widens_icon_zekinh_gather_inmap_nsdfg():
    """ICON zekinh-style data-dependent gather must successfully widen.

    After preprocessing (LoopToMap + RefineNestedAccess) the kernel's body
    is one in-map NSDFG with per-iter inputs ``e_bln[jb, 0:3, jc]``,
    ``z_kin_hor_e[0:NB, jk, 0:NPROMA]`` and an output
    ``z_ekinh[jb, jk, jc]``. Running ExpandNestedSDFGInputs on every
    NSDFG must produce a valid SDFG with full-array outer subsets.

    Regression coverage for two bugs that prevented this pattern from
    widening:

    * ``sympy.Subscript`` (does not exist on ``sympy``) was used instead
      of :class:`dace.symbolic.Subscript`.
    * ``_uncollapse_subscript`` was declared with a single ``node``
      parameter, but ``sympy.Basic.replace`` invokes the callback with
      the matched node's ``args`` splatted positionally.
    """
    from dace.transformation.interstate import LoopToMap, RefineNestedAccess
    sdfg = _icon_zekinh_gather_kernel.to_sdfg(simplify=True)
    sdfg.apply_transformations_repeated(LoopToMap, permissive=False, validate=False)
    sdfg.apply_transformations_repeated(RefineNestedAccess, permissive=False, validate=False)
    # The pass must apply without raising on any of the body NSDFGs the
    # loop nest produces.
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    sdfg.validate()


def test_expand_terminates_on_already_full_inputs():
    """``apply_transformations_repeated`` must terminate when every NSDFG
    in/out edge already reads/writes the full outer array.

    Regression: ``can_be_applied`` previously returned ``True``
    unconditionally, so the orchestrator's repeat loop spun forever.
    """
    sdfg = _build_map_to_for_loop_test_sdfg()
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    # Second invocation must do nothing (no NSDFG matches any more).
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    sdfg.validate()


def test_icon_zekinh_gather_numerics_via_expand_then_inline():
    """End-to-end: post-LoopToMap, Expand + Inline must preserve numerics.

    Runs the ICON kernel against numpy-generated random inputs (with
    valid in-bounds index arrays) and asserts the post-rewrite SDFG
    produces the same outputs as a fresh, untransformed reference.

    Regression (fixed 2026-06-15): the gather index lives in an
    interstate-edge assignment ``edge_idx_index = edge_idx[0, 0, 0]``.
    When ``ExpandNestedSDFGInputs`` widened the ``edge_idx`` connector to
    the full array it must uncollapse that subscript to
    ``edge_idx[jb, jc, 0]``, but ``_uncollapse_subscript`` compared the
    subscript base via ``base == sympy.Symbol(outer_name)`` -- which is
    always False for a ``dace.symbolic.symbol`` (the dtype-carrying
    subclass). The offset was never applied, so every lane gathered
    ``edge_idx[0, 0, 0]``. Fixed by comparing the base by name."""
    from dace.transformation.interstate import LoopToMap, RefineNestedAccess

    nb, nlev, nproma = 4, 6, 8
    rng = np.random.default_rng(0xC0FFEE)
    e_bln = rng.standard_normal((nb, 3, nproma))
    edge_idx = rng.integers(0, nproma, size=(nb, nproma, 3), dtype=np.int32)
    edge_blk = rng.integers(0, nb, size=(nb, nproma, 3), dtype=np.int32)
    z_kin_hor_e = rng.standard_normal((nb, nlev, nproma))
    ref_out = np.zeros((nb, nlev, nproma))
    test_out = np.zeros_like(ref_out)

    # Reference: unmodified SDFG
    ref_sdfg = _icon_zekinh_gather_kernel.to_sdfg(simplify=True)
    ref_sdfg(e_bln=e_bln.copy(),
             edge_idx=edge_idx.copy(),
             edge_blk=edge_blk.copy(),
             z_kin_hor_e=z_kin_hor_e.copy(),
             z_ekinh=ref_out,
             NB=nb,
             NLEV=nlev,
             NPROMA=nproma)

    # Transformed: LoopToMap + Expand + Inline
    tsdfg = _icon_zekinh_gather_kernel.to_sdfg(simplify=True)
    tsdfg.apply_transformations_repeated(LoopToMap, permissive=False, validate=False)
    tsdfg.apply_transformations_repeated(RefineNestedAccess, permissive=False, validate=False)
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(tsdfg, {})
    PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(tsdfg, {})
    tsdfg.validate()
    tsdfg(e_bln=e_bln.copy(),
          edge_idx=edge_idx.copy(),
          edge_blk=edge_blk.copy(),
          z_kin_hor_e=z_kin_hor_e.copy(),
          z_ekinh=test_out,
          NB=nb,
          NLEV=nlev,
          NPROMA=nproma)
    assert np.allclose(test_out, ref_out), f'max diff: {np.abs(test_out - ref_out).max():.3e}'


@dace.program
def _scatter_no_conflict_kernel(a: dace.float64[N], b: dace.float64[N], idx: dace.int32[N]):
    """Per-iteration scatter ``a[idx[i]] = b[i] + 1.0``.

    Used together with a guaranteed-non-conflicting (permutation) ``idx``
    -- every write lands on a distinct destination so there is no race;
    the test generator constructs ``idx`` as ``np.random.permutation``.
    """
    for i in dace.map[0:N]:
        a[idx[i]] = b[i] + 1.0


def test_scatter_no_conflict_widens_and_validates():
    """Indirect store ``a[idx[i]]`` -- ExpandNestedSDFGInputs must widen
    the scatter pattern and keep the inner subscript intact."""
    from dace.transformation.dataflow.map_for_loop import MapToForLoop
    sdfg = _scatter_no_conflict_kernel.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([MapToForLoop()]).apply_pass(sdfg, {})
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})
    sdfg.validate()


def test_scatter_no_conflict_numerics_via_expand_then_inline():
    """End-to-end: scatter with permutation indices (zero conflicts) --
    the post-rewrite SDFG must match an untransformed reference run."""
    from dace.transformation.dataflow.map_for_loop import MapToForLoop
    n = 64
    rng = np.random.default_rng(0xBAD1DEA)
    idx = rng.permutation(n).astype(np.int32)
    b = rng.standard_normal(n)
    ref_a = np.zeros(n)
    test_a = np.zeros(n)
    _scatter_no_conflict_kernel.to_sdfg(simplify=True)(a=ref_a, b=b.copy(), idx=idx.copy(), N=n)
    tsdfg = _scatter_no_conflict_kernel.to_sdfg(simplify=True)
    PatternMatchAndApplyRepeated([MapExpansion()]).apply_pass(tsdfg, {})
    PatternMatchAndApplyRepeated([MapToForLoop()]).apply_pass(tsdfg, {})
    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(tsdfg, {})
    PatternMatchAndApplyRepeated([InlineMultistateSDFG()]).apply_pass(tsdfg, {})
    tsdfg.validate()
    tsdfg(a=test_a, b=b.copy(), idx=idx.copy(), N=n)
    assert np.allclose(test_a, ref_a), f'max diff: {np.abs(test_a - ref_a).max():.3e}'


def test_widened_shape_introduces_symbol_already_in_inner_table():
    """Repro for 2026-06-10 fix: when ExpandNestedSDFGInputs widens an inner connector's
    memlet (``A[ii]`` -> ``A[0:N, 0:N]``), the inner array's SHAPE gains a reference to
    the outer dim symbol ``N``. If ``N`` is already declared in ``inner_sdfg.symbols``
    (e.g. inherited during nesting), the prior logic excluded it from the
    ``introduced_symbols`` set via ``defined_syms`` filtering -- yet ``N`` was still
    NOT bound in ``symbol_mapping``. The validator then trips with
    ``Missing symbols on nested SDFG: ['N']``.

    The fix walks every inner array's ``free_symbols`` (which aggregates shape + strides
    + offset) and adds any non-connector, non-mapped symbol to ``introduced_symbols``.
    """
    sdfg = dace.SDFG("widened_shape_sym")
    N_sym = dace.symbol("N", dtype=dace.int64)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", (N_sym, N_sym), dace.float64)
    state = sdfg.add_state("s")
    a_outer = state.add_access("A")

    # Build an inner NSDFG that already has N in its symbols but no symbol_mapping entry
    # for it. The inner SDFG accepts a connector "A_conn" whose shape is (N, N) -- which
    # will trip the "Missing symbols on nested SDFG" check unless symbol_mapping
    # auto-propagates N.
    inner = dace.SDFG("inner")
    inner.add_symbol("N", dace.int64)
    inner.add_array("A_conn", (N_sym, N_sym), dace.float64)
    inner.add_state("body")
    nsdfg = state.add_nested_sdfg(inner, {"A_conn"}, set(), symbol_mapping={})
    # Outer memlet feeds a narrowed subset; Expand will widen to the full source shape.
    state.add_edge(a_outer, None, nsdfg, "A_conn", dace.Memlet("A[0, 0]"))
    # add_nested_sdfg auto-populates symbol_mapping with any inner.symbols. Drop N
    # to recreate the original bug shape (real-world callers can hit this via passes
    # that reset the mapping or via nested NSDFG construction patterns).
    nsdfg.symbol_mapping.pop("N", None)

    # Pre-expansion: nsdfg.symbol_mapping has no N. inner.symbols already has N.
    assert "N" not in nsdfg.symbol_mapping
    assert "N" in inner.symbols

    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})

    # After expansion: A_conn's shape is (N, N) referencing N. symbol_mapping MUST bind N.
    assert "N" in nsdfg.symbol_mapping, \
        f"N should be auto-propagated to symbol_mapping; got {dict(nsdfg.symbol_mapping)}"
    # Validation must succeed (the original bug raised here).
    sdfg.validate()


def test_scalar_source_not_subscripted_in_interstate_assignment():
    """Per user direction 2026-06-10: ``on codeblocks we treat scalar as if it is a
    symbol``. ``_uncollapse_scalar`` was unconditionally wrapping references to
    ``outer_name`` with ``[offset_dims]`` -- emitting ``c1[0] * c2[0]`` in an
    interstate-edge assignment whose outer source is a Scalar passed as
    ``const T&``. The fix skips the wrap when the outer descriptor is
    :class:`dace.data.Scalar`.
    """
    sdfg = dace.SDFG("scalar_in_iedge")
    N_sym = dace.symbol("N", dtype=dace.int64)
    sdfg.add_symbol("N", dace.int64)
    sdfg.add_array("A", (N_sym, ), dace.float64)
    sdfg.add_scalar("c1", dace.float64)
    sdfg.add_scalar("c2", dace.float64)
    state = sdfg.add_state("s")
    a_an = state.add_access("A")
    c1_an = state.add_access("c1")
    c2_an = state.add_access("c2")

    # Inner NSDFG: has its own c1, c2 connectors and an interstate edge that
    # references c1, c2 by bare name (the @dace.program shape).
    inner = dace.SDFG("inner")
    inner.add_array("A_conn", (1, ), dace.float64)  # length-1 connector for outer A
    inner.add_scalar("c1", dace.float64)
    inner.add_scalar("c2", dace.float64)
    s_init = inner.add_state("init")
    s_body = inner.add_state("body")
    # interstate edge with bare-symbol assignment (the shape that codegens wrong).
    inner.add_edge(s_init, s_body, dace.InterstateEdge(assignments={"prod": "(c1 * c2)"}))

    nsdfg = state.add_nested_sdfg(inner, {"A_conn", "c1", "c2"}, set(), symbol_mapping={})
    state.add_edge(a_an, None, nsdfg, "A_conn", dace.Memlet("A[0]"))
    state.add_edge(c1_an, None, nsdfg, "c1", dace.Memlet("c1[0]"))
    state.add_edge(c2_an, None, nsdfg, "c2", dace.Memlet("c2[0]"))

    PatternMatchAndApplyRepeated([ExpandNestedSDFGInputs()]).apply_pass(sdfg, {})

    # The interstate-edge assignment in the inner SDFG MUST still reference c1, c2
    # as bare symbols -- NOT as ``c1[0]``, ``c2[0]``. The outer descriptors are
    # Scalars, so the codegen will pass them as ``const T&`` references; subscripting
    # a reference is invalid C++.
    found_assigns = []
    for edge in nsdfg.sdfg.edges():
        found_assigns.extend(edge.data.assignments.values())
    expr = next((a for a in found_assigns if "c1" in a and "c2" in a), None)
    assert expr is not None, f"expected an assignment referencing c1 and c2; got {found_assigns}"
    assert "[0]" not in expr, f"Scalar reference should not be subscripted; got {expr!r}"
