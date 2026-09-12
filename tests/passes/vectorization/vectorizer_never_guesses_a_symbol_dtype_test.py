# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A vectorization pass that re-declares a symbol must resolve its dtype, never assume ``int64``.

``_hashable_content`` folds dtype into symbol identity on this branch, so one name at two dtypes is
TWO symbols. Every pass below re-declares an outer-scope symbol inside a body it builds; each one
used to fall back to ``int64`` when ``sdfg.symbols`` did not answer -- and ``sdfg.symbols`` never
answers for a map parameter, which is where these names overwhelmingly come from. The re-declared
copy then stops folding against the parameter it stands for: ``Min(i, i)`` survives, ``i - i``
prints as ``i - i``, and an injectivity test downstream calls an affine write non-affine and ships
the kernel un-tiled.

Each test therefore asserts the FOLD, not just an attribute: a symbol minted at the authoritative
dtype must cancel against one minted at the declared dtype. Every fixture builds its map range as
an explicit :class:`~dace.subsets.Range` over two int32 symbols, because the ``'M:N'`` string form
stores the end as ``N - 1`` and that integer literal widens the inferred parameter back to int64 --
a fixture that merely "uses an int32 symbol" reproduces nothing.
"""
import sympy

import dace
from dace import subsets, symbolic
from dace.sdfg import nodes
from dace.sdfg.sdfg import SDFG
from dace.sdfg.state import SDFGState
from dace.transformation.passes.analysis import scopes
from dace.transformation.passes.vectorization.fuse_branched_tail_remainder import FuseBranchedTailRemainder
from dace.transformation.passes.vectorization.split_map_for_tile_remainder import (MASKED_TAIL_MARKER, TILE_MAIN_MARKER)
from dace.transformation.passes.vectorization.utils.mask_scaffold import thread_symbols_into_nsdfg
from dace.transformation.passes.vectorization.utils.subsets import repl_subset_to_use_laneid_offset
from dace.transformation.passes.vectorization.utils.tasklets import materialise_lane_id_index_tile
from dace.transformation.passes.vectorization.vectorize_multi_dim import (_RunExpandNestedSDFGInputs,
                                                                          _resolve_body_nsdfg_symbol_aliases)
from dace.transformation.passes.vectorization.widen_accesses import emit_per_lane_symbol_fanout

W = 8


def cancels_against(name: str, declared: dace.dtypes.typeclass, authoritative: dace.dtypes.typeclass) -> bool:
    """Does a ``name`` minted at ``declared`` still fold against one minted at ``authoritative``?

    This is the whole defect in one expression. Two dtypes for one name are two symbols, so ``Min``
    keeps both arms and the difference refuses to cancel; the printed form of both is identical, so
    nothing downstream can see that it happened.
    """
    mine = symbolic.symbol(name, declared)
    theirs = symbolic.symbol(name, authoritative)
    return sympy.Min(mine, theirs) is mine and symbolic.simplify(mine - theirs) == 0


def int32_parameter_map(sdfg: SDFG, state: SDFGState, label: str, param: str) -> tuple[nodes.MapEntry, nodes.MapExit]:
    """A map whose parameter genuinely infers as int32: both bounds reach ``Range`` as bare symbols."""
    lo = symbolic.symbol('LO', dace.int32)
    hi = symbolic.symbol('HI', dace.int32)
    sdfg.symbols.setdefault('LO', dace.int32)
    sdfg.symbols.setdefault('HI', dace.int32)
    return state.add_map(label, {param: subsets.Range([(lo, hi, 1)])})


def map_parameter_dtype(state: SDFGState, entry: nodes.MapEntry, param: str) -> dace.dtypes.typeclass:
    """The dtype the map itself gives its parameter -- the authority every re-declaration owes."""
    return entry.new_symbols(state.sdfg, state, {})[param]


def body_nsdfg_in_map_scope(state: SDFGState, entry: nodes.MapEntry, exit_node: nodes.MapExit, inner: SDFG,
                            symbol_mapping: dict[str, symbolic.SymbolicType]) -> nodes.NestedSDFG:
    """Hang ``inner`` inside ``entry``'s scope, wired by an ordering edge on each side.

    The node is constructed directly rather than through ``add_nested_sdfg``: that helper refuses a
    body whose free symbols are unmapped, which is precisely the graph two of these passes exist to
    repair. Building it here is the only way to hand them their own input.
    """
    node = nodes.NestedSDFG(inner.label, inner, {}, {}, symbol_mapping=symbol_mapping)
    state.add_node(node)
    inner.parent_nsdfg_node = node
    inner.parent = state
    inner.parent_sdfg = state.sdfg
    state.add_edge(entry, None, node, None, dace.Memlet())
    state.add_edge(node, None, exit_node, None, dace.Memlet())
    return node


def leaf_sdfg(label: str, free_symbol: str) -> SDFG:
    """An SDFG whose single state reads ``free_symbol`` from an interstate condition, nothing else."""
    inner = SDFG(label)
    inner.add_array('t', [1], dace.float64)
    first = inner.add_state('first', is_start_block=True)
    second = inner.add_state('second')
    inner.add_edge(first, second, dace.InterstateEdge(condition=f'{free_symbol} > 0'))
    return inner


# --- vectorize_multi_dim: the symbol-alias inliner, several NSDFGs down -------------------------


def two_deep_alias_nest() -> tuple[SDFG, SDFG, dace.dtypes.typeclass]:
    """Top SDFG -> middle NSDFG holding an int32-parameter map -> leaf NSDFG aliasing that parameter.

    The parameter is declared in NO symbol table: not the top SDFG's, not the middle's. The middle
    SDFG's state is the only place it exists, which is why a resolver anchored on the top-level
    SDFG -- the shape of the earlier partial fix -- still guesses here.
    """
    middle = SDFG('middle')
    middle.add_array('m', [1], dace.float64)
    mid_state = middle.add_state('mid', is_start_block=True)
    entry, exit_node = int32_parameter_map(middle, mid_state, 'inner_map', 'i')
    leaf = leaf_sdfg('leaf', 'it')
    body_nsdfg_in_map_scope(mid_state, entry, exit_node, leaf, {'it': symbolic.pystr_to_symbolic('i')})

    top = SDFG('top')
    top.add_array('m', [1], dace.float64)
    for name in ('LO', 'HI'):
        top.add_symbol(name, dace.int32)
    top_state = top.add_state('top_state', is_start_block=True)
    wrapper = top_state.add_nested_sdfg(
        middle,
        inputs={'m': None},
        outputs={},
        symbol_mapping={name: symbolic.pystr_to_symbolic(name)
                        for name in ('LO', 'HI')})
    top_state.add_edge(top_state.add_access('m'), None, wrapper, 'm', dace.Memlet('m[0]'))
    return top, leaf, map_parameter_dtype(mid_state, entry, 'i')


def test_an_alias_inlined_two_nested_sdfgs_down_keeps_the_map_parameters_own_dtype():
    top, leaf, authoritative = two_deep_alias_nest()
    assert authoritative == dace.int32, 'fixture is not exercising an int32 parameter'
    assert 'i' not in top.symbols, 'the top-level table must not answer, or the fixture proves nothing'

    _resolve_body_nsdfg_symbol_aliases(top)

    assert 'it' not in leaf.symbols and leaf.symbols['i'] == authoritative
    assert cancels_against('i', leaf.symbols['i'], authoritative)


def test_a_free_symbol_bound_back_onto_a_body_keeps_the_map_parameters_own_dtype():
    """``_bind_missing_free_symbols`` repairs "Missing symbols on nested SDFG" -- at the right dtype."""
    sdfg = SDFG('bind_missing')
    sdfg.add_array('m', [1], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = int32_parameter_map(sdfg, state, 'outer_map', 'i')
    leaf = leaf_sdfg('leaf', 'i')
    body_nsdfg_in_map_scope(state, entry, exit_node, leaf, {})
    authoritative = map_parameter_dtype(state, entry, 'i')
    assert authoritative == dace.int32 and 'i' not in sdfg.symbols

    bound = _RunExpandNestedSDFGInputs._bind_missing_free_symbols(sdfg)

    assert bound == 1
    assert leaf.symbols['i'] == authoritative
    assert cancels_against('i', leaf.symbols['i'], authoritative)


# --- widen_accesses: the per-lane fanout of an interstate-defined gather index -------------------


def test_per_lane_gather_planes_inherit_the_dtype_of_the_edge_that_defined_the_index():
    """``__sym = ii`` is a definition. The plane symbols it fans out to must not be widened past it."""
    inner = SDFG('gather_index')
    inner.add_symbol('ii', dace.int32)
    inner.add_array('src', [64], dace.float64)
    first = inner.add_state('first', is_start_block=True)
    second = inner.add_state('second')
    inner.add_edge(first, second, dace.InterstateEdge(assignments={'__sym': 'ii'}))
    assert '__sym' not in inner.symbols, 'an interstate-defined symbol is declared nowhere; that is the point'

    planes = emit_per_lane_symbol_fanout(inner, '__sym', ('ii', ), (W, ))

    assert planes is not None and len(planes) == W
    declared = {inner.symbols[name] for name in planes.values()}
    assert declared == {dace.int32}
    assert all(cancels_against(name, inner.symbols[name], dace.int32) for name in planes.values())


# --- utils/subsets: the dtype lives one level below Range.free_symbols --------------------------


def test_a_lane_offset_symbol_inherits_the_dtype_carried_by_the_subsets_own_symbol():
    """``Range.free_symbols`` answers ``str``; the instance that knows the dtype is a level down."""
    sdfg = SDFG('lane_offset')
    sdfg.add_symbol('ii', dace.int32)
    k = symbolic.symbol('k', dace.int32)
    subset = subsets.Range([(k, k, 1)])
    assert 'k' not in sdfg.symbols, 'k stands for a map parameter: declared in no table, only in the subset'
    assert all(isinstance(name, str) for name in subset.free_symbols)

    repl_subset_to_use_laneid_offset(sdfg, subset, '3', 'ii')

    lane_names = [name for name in sdfg.symbols if name != 'ii']
    assert len(lane_names) == 1
    assert sdfg.symbols[lane_names[0]] == dace.int32
    assert cancels_against(lane_names[0], sdfg.symbols[lane_names[0]], k.dtype)


# --- utils/tasklets: the emitted cast is the descriptor's, not a repeated spelling ---------------


def test_the_index_tile_cast_is_read_off_the_tile_descriptor():
    """The materialised index tile is int64 by the never-narrow rule; the cast must SAY the same."""
    sdfg = SDFG('index_tile')
    sdfg.add_symbol('ii', dace.int64)
    state = sdfg.add_state('s', is_start_block=True)

    out = materialise_lane_id_index_tile(state, 'ii', ('ii', ), (W, ))

    desc = sdfg.arrays[out.data]
    assert desc.dtype == dace.int64, 'an index tile must not be narrowed'
    tasklet = next(n for n in state.nodes() if isinstance(n, nodes.Tasklet))
    assert f'({desc.dtype.ctype})' in tasklet.code.as_string
    assert len([line for line in tasklet.code.as_string.splitlines() if '_out[' in line]) == 1


# --- mask_scaffold: the mask's bound symbols threaded into the body -----------------------------


def test_a_mask_bound_symbol_threaded_into_a_body_keeps_the_map_parameters_own_dtype():
    """The mask's ``iv < ub`` compares the threaded copy with the parameter it came from."""
    sdfg = SDFG('mask_thread')
    sdfg.add_array('m', [1], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)
    entry, exit_node = int32_parameter_map(sdfg, state, 'tiled_map', 'i')
    body = SDFG('body')
    body.add_array('t', [1], dace.float64)
    body.add_state('only', is_start_block=True)
    node = body_nsdfg_in_map_scope(state, entry, exit_node, body, {})
    authoritative = map_parameter_dtype(state, entry, 'i')
    assert authoritative == dace.int32 and 'i' not in sdfg.symbols

    thread_symbols_into_nsdfg(body, node, ('i', 'HI'), sdfg, state, scopes.ScopedSymbolResolver())

    assert body.symbols['i'] == authoritative and body.symbols['HI'] == dace.int32
    assert cancels_against('i', body.symbols['i'], authoritative)
    assert node.symbol_mapping['i'] == symbolic.pystr_to_symbolic('i')


# --- fuse_branched_tail_remainder: the fused body re-declares the fused map's own parameter ------


def tile_body(label: str) -> SDFG:
    """A body SDFG with one boundary array, shaped as the fuser's single-NSDFG contract expects."""
    body = SDFG(label)
    body.add_array('a', [64], dace.float64)
    st = body.add_state('only', is_start_block=True)
    tasklet = st.add_tasklet('t', {'_i'}, {'_o'}, '_o = _i + 1.0')
    st.add_edge(st.add_access('a'), None, tasklet, '_i', dace.Memlet('a[0]'))
    st.add_edge(tasklet, '_o', st.add_access('a'), None, dace.Memlet('a[0]'))
    return body


def split_main_and_masked_tail() -> tuple[SDFG, SDFGState, dace.dtypes.typeclass]:
    """The structural signature the fuser pairs on: ``[LO : HI : W]`` plus ``[HI + 1 : TOP : 1]``."""
    lo = symbolic.symbol('LO', dace.int32)
    hi = symbolic.symbol('HI', dace.int32)
    top = symbolic.symbol('TOP', dace.int32)
    sdfg = SDFG('split_pair')
    for name in ('LO', 'HI', 'TOP'):
        sdfg.add_symbol(name, dace.int32)
    sdfg.add_array('a', [64], dace.float64)
    state = sdfg.add_state('s', is_start_block=True)

    main_entry, main_exit = state.add_map(f'k{TILE_MAIN_MARKER}', {'i': subsets.Range([(lo, hi, W)])})
    tail_entry, tail_exit = state.add_map(f'k{MASKED_TAIL_MARKER}', {'i': subsets.Range([(hi + 1, top, W)])})
    for entry, exit_node, body in ((main_entry, main_exit, tile_body('main_body')), (tail_entry, tail_exit,
                                                                                     tile_body('tail_body'))):
        node = state.add_nested_sdfg(body,
                                     inputs={'a': None},
                                     outputs={'a': None},
                                     symbol_mapping={'i': symbolic.pystr_to_symbolic('i')})
        state.add_memlet_path(state.add_access('a'), entry, node, dst_conn='a', memlet=dace.Memlet('a[0:64]'))
        state.add_memlet_path(node, exit_node, state.add_access('a'), src_conn='a', memlet=dace.Memlet('a[0:64]'))
    return sdfg, state, map_parameter_dtype(state, main_entry, 'i')


def test_the_fused_remainder_body_redeclares_the_fused_maps_parameter_at_its_own_dtype():
    """The branch predicate ``i <= TOP - W + 1`` only folds while the body's ``i`` IS the map's."""
    sdfg, state, authoritative = split_main_and_masked_tail()
    assert authoritative == dace.int32 and 'i' not in sdfg.symbols

    fused = FuseBranchedTailRemainder(widths=(W, )).apply_pass(sdfg, {})

    assert fused == 1
    bodies = [n for n in state.nodes() if isinstance(n, nodes.NestedSDFG)]
    assert len(bodies) == 1, 'the two bodies should have collapsed into one conditional body'
    fused_body = bodies[0].sdfg
    assert fused_body.symbols['i'] == authoritative
    assert cancels_against('i', fused_body.symbols['i'], authoritative)
