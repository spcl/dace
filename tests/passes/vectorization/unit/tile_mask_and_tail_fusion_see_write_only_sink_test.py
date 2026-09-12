# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The tile mask generator and the tail fuser over a body ending in a write-only scratch scalar.

CloudSC's ``zanew_0`` is an ordinary shape: a one-element ``Register`` transient with ``Scope``
lifetime, one in-edge and no out-edge. ``SDFGState.all_nodes_between`` throws its ENTIRE walk away
on reaching such a node (``dace/sdfg/graph.py``), so a pass reading that walk as "the body" answers
over nothing.

The two passes here answered that way for OPPOSITE reasons, and the tests keep them apart:

* :class:`GenerateTileIterationMask` looked for the body nest in the emptied walk and found none,
  so it attached no mask -- while ``ConvertTaskletsToTileOps`` finds the very same nest through
  ``scope_subgraph``, tiles it, and wires every tile op ``has_mask=False``. The map comes out
  strided by W with nothing masking the lanes past the bound: a wrong answer, so the walk is gone
  from that pass.
* :class:`FuseBranchedTailRemainder` demands a body of EXACTLY one nested SDFG, and the empty list
  fails that the same way the scope-based body does. An emptied walk can only leave the pair as two
  correct un-fused maps, so the walk stays and these tests pin the refusal as deliberate.
"""
import dace
from dace import SDFGState, nodes
from dace.libraries.tileops import TileMaskGen
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.vectorization.fuse_branched_tail_remainder import FuseBranchedTailRemainder
from dace.transformation.passes.vectorization.generate_tile_iteration_mask import GenerateTileIterationMask
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (MASKED_TAIL_MARKER, TILE_MAIN_MARKER)
from dace.transformation.passes.vectorization.utils.map_predicates import is_vectorizable_map, map_body_nodes
from dace.transformation.passes.vectorization.utils.name_schemes import TileNameScheme

N = 16
W = 8
SINK = 'zanew_0'


def body_nest() -> dace.SDFG:
    """``B[i] = A[i, 0] * 2`` -- the single-nest body shape both passes require.

    Connectors carry the OUTER array names, the way ``nest_state_subgraph`` leaves them: the tail
    fuser reads the enclosing SDFG's descriptor by connector name to build the fused body.
    """
    inner = dace.SDFG('body')
    inner.add_symbol('i', dace.int64)
    inner.add_array('A', [N, N], dace.float64)
    inner.add_array('B', [N], dace.float64)
    st = inner.add_state('compute', is_start_block=True)
    t = st.add_tasklet('acc', {'cur'}, {'out'}, 'out = cur * 2.0')
    st.add_edge(st.add_access('A'), None, t, 'cur', dace.Memlet('A[i, 0]'))
    st.add_edge(t, 'out', st.add_access('B'), None, dace.Memlet('B[i]'))
    return inner


def attach_scratch_sink(sdfg: dace.SDFG, state: SDFGState, map_entry: nodes.MapEntry) -> nodes.AccessNode:
    """A ``zanew_0``-shaped write-only scratch scalar inside ``map_entry``'s scope."""
    sdfg.add_array(SINK, [1],
                   dace.float64,
                   storage=dace.dtypes.StorageType.Register,
                   transient=True,
                   lifetime=dace.dtypes.AllocationLifetime.Scope)
    stage = state.add_tasklet('stage', {}, {'o'}, 'o = 0.0')
    state.add_nedge(map_entry, stage, dace.Memlet())
    sink = state.add_access(SINK)
    state.add_edge(stage, 'o', sink, None, dace.Memlet(f'{SINK}[0]'))
    return sink


def walk_is_degenerate(state: SDFGState, map_entry: nodes.MapEntry) -> bool:
    """``all_nodes_between`` threw the whole body away on this map."""
    return len(state.all_nodes_between(map_entry, state.exit_node(map_entry))) == 0


def add_nested_body_map(sdfg: dace.SDFG, state: SDFGState, label: str, rng: str,
                        with_sink: bool) -> tuple[nodes.MapEntry, nodes.NestedSDFG]:
    """One map whose body is a single nest, optionally beside the write-only scratch scalar."""
    me, mx = state.add_map(label, dict(i=rng))
    ns = state.add_nested_sdfg(body_nest(), {'A': None}, {'B': None}, symbol_mapping={'i': 'i'})
    state.add_memlet_path(state.add_access('A'), me, ns, dst_conn='A', memlet=dace.Memlet(f'A[0:{N}, 0:{N}]'))
    state.add_memlet_path(ns, mx, state.add_access('B'), src_conn='B', memlet=dace.Memlet(f'B[0:{N}]'))
    sinks = {True: attach_scratch_sink, False: lambda *_: None}
    sinks[with_sink](sdfg, state, me)
    return me, ns


def nested_body_map(label: str, with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry]:
    """A single-kernel SDFG holding one such map."""
    sdfg = dace.SDFG('nested_body')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, _ = add_nested_body_map(sdfg, state, label, f'0:{N}', with_sink)
    sdfg.validate()
    return sdfg, state, me


def split_pair(with_sink: bool) -> tuple[dace.SDFG, SDFGState, nodes.MapEntry, nodes.MapEntry]:
    """The post-split shape: interior ``[0:W]`` marked ``__tile_main`` plus its ``__masked_tail``."""
    sdfg = dace.SDFG('split_pair')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    main, _ = add_nested_body_map(sdfg, state, f'row{TILE_MAIN_MARKER}', f'0:{W}:{W}', with_sink)
    tail, _ = add_nested_body_map(sdfg, state, f'row{MASKED_TAIL_MARKER}', f'{W}:{N}:{W}', False)
    sdfg.validate()
    return sdfg, state, main, tail


def mask_generators(sdfg: dace.SDFG) -> list[TileMaskGen]:
    return [n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, TileMaskGen)]


def body_nest_of(state: SDFGState, map_entry: nodes.MapEntry) -> nodes.NestedSDFG:
    return next(n for n in map_body_nodes(state, map_entry) if isinstance(n, nodes.NestedSDFG))


# ---------------------------------------------------------------------------
# GenerateTileIterationMask -- the mask must reach a body that ends in the sink.
# ---------------------------------------------------------------------------


def test_a_tiled_map_whose_body_ends_in_a_scratch_scalar_still_gets_its_iteration_mask() -> None:
    """Without this the converter tiles the same nest maskless and W-1 lanes run past the bound."""
    sdfg, state, map_entry = nested_body_map('row', with_sink=True)
    nest = body_nest_of(state, map_entry)

    attached = GenerateTileIterationMask(widths=(W, )).apply_pass(sdfg, {})

    gens = mask_generators(sdfg)
    mask = nest.sdfg.arrays[TileNameScheme.ITER_MASK]
    assert walk_is_degenerate(state, map_entry)
    assert is_vectorizable_map(state, map_entry, 1) is True
    assert attached == 1
    assert len(gens) == 1
    assert list(gens[0].widths) == [W]
    assert list(gens[0].iter_vars) == ['i']
    assert [str(s) for s in mask.shape] == [str(W)]
    assert mask.dtype is dace.bool_
    assert mask.transient is True
    assert nest.sdfg.start_block is next(s for s in nest.sdfg.states() if any(
        isinstance(n, TileMaskGen) for n in s.nodes()))
    sdfg.validate()


def test_a_clean_single_nest_body_is_masked_exactly_as_it_was() -> None:
    sdfg, state, map_entry = nested_body_map('row', with_sink=False)

    attached = GenerateTileIterationMask(widths=(W, )).apply_pass(sdfg, {})

    assert walk_is_degenerate(state, map_entry) is False
    assert attached == 1
    assert len(mask_generators(sdfg)) == 1


def test_the_all_main_interior_map_ending_in_a_scratch_scalar_stays_maskless() -> None:
    """``__tile_main`` is in bounds on every tiled dim by construction -- the mask-free fast path."""
    sdfg, state, map_entry = nested_body_map(f'row{TILE_MAIN_MARKER}', with_sink=True)

    attached = GenerateTileIterationMask(widths=(W, )).apply_pass(sdfg, {})

    assert walk_is_degenerate(state, map_entry)
    assert attached is None
    assert mask_generators(sdfg) == []


def test_a_body_ending_in_a_scratch_scalar_is_not_masked_twice() -> None:
    sdfg, state, map_entry = nested_body_map('row', with_sink=True)
    pass_instance = GenerateTileIterationMask(widths=(W, ))
    pass_instance.apply_pass(sdfg, {})

    reattached = pass_instance.apply_pass(sdfg, {})

    assert reattached is None
    assert len(mask_generators(sdfg)) == 1


def test_a_bare_tasklet_body_ending_in_a_scratch_scalar_carries_no_nest_to_mask() -> None:
    """The mask lives inside the body nest, so an unnested map is still refused -- now knowingly."""
    sdfg = dace.SDFG('flat_body')
    sdfg.add_array('A', [N, N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    state = sdfg.add_state('main', is_start_block=True)
    me, mx = state.add_map('row', dict(i=f'0:{N}'))
    t = state.add_tasklet('scale', {'a'}, {'o'}, 'o = a * 2.0')
    state.add_memlet_path(state.add_access('A'), me, t, dst_conn='a', memlet=dace.Memlet('A[i, 0]'))
    state.add_memlet_path(t, mx, state.add_access('B'), src_conn='o', memlet=dace.Memlet('B[i]'))
    attach_scratch_sink(sdfg, state, me)

    attached = GenerateTileIterationMask(widths=(W, )).apply_pass(sdfg, {})

    assert walk_is_degenerate(state, me)
    assert not any(isinstance(n, nodes.NestedSDFG) for n in map_body_nodes(state, me))
    assert attached is None
    assert mask_generators(sdfg) == []


# ---------------------------------------------------------------------------
# FuseBranchedTailRemainder -- the emptied walk can only refuse, and does.
# ---------------------------------------------------------------------------


def test_an_interior_and_masked_tail_pair_of_single_nest_bodies_is_fused_into_one_branched_map() -> None:
    sdfg, state, main, tail = split_pair(with_sink=False)

    fused = FuseBranchedTailRemainder(widths=(W, )).apply_pass(sdfg, {})

    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    conditionals = [b for b in sdfg.all_control_flow_blocks(recursive=True) if isinstance(b, ConditionalBlock)]
    assert fused == 1
    assert entries == [main]
    assert tail not in state.nodes()
    assert [str(r) for r in main.map.range.ranges] == [f'(0, {N - 1}, {W})']
    assert main.map.label == 'row'
    assert len(conditionals) == 1
    assert len(conditionals[0].branches) == 2


def test_an_interior_body_beside_a_scratch_scalar_is_left_as_two_correct_unfused_maps() -> None:
    """The deliberate refusal: a scope holding more than the nest is not the fusable shape.

    The emptied walk and the scope-based body agree here -- ``len(body) == 1`` fails either way --
    so the pass keeps reading the walk and this test pins the answer both give.
    """
    sdfg, state, main, tail = split_pair(with_sink=True)

    fused = FuseBranchedTailRemainder(widths=(W, )).apply_pass(sdfg, {})

    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    assert walk_is_degenerate(state, main)
    assert len(map_body_nodes(state, main)) == 3
    assert FuseBranchedTailRemainder._sole_body_nsdfg(state, main) is None
    assert fused is None
    assert entries == [main, tail]
    assert not any(isinstance(b, ConditionalBlock) for b in sdfg.all_control_flow_blocks(recursive=True))
    assert [str(r) for r in main.map.range.ranges] == [f'(0, {W - 1}, {W})']
    sdfg.validate()
