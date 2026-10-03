# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``ScatterConflictCheck`` libnode: flag duplicate values in a 1-D integer array.

Runtime proof that a scatter index ``ip`` is a permutation -- no duplicates means no
write-write races, so ``a[ip[i]] = ...`` may run as a parallel Map. Opaque: the tile
vectorizer never looks through a library node, so the scatter guard never perturbs
tiling (the compute Map it guards vectorizes normally).

``_count_out`` is a length-1 ``int64`` **host** scalar in every backend, including CUDA,
so the downstream ``trap_sym`` interstate binding and the trap read stay on the host
regardless of where ``ip`` lives. It is 0 iff the index is a permutation and 1 otherwise:
the verify pass OR-reduces, so the value is a flag, not a duplicate count.

The tag array the check indexes by index-*value* is an optional ``_owner_out`` connector: an
``int64`` host Array the caller sizes by the scattered array's domain (see
:func:`~dace.transformation.passes.scatter_conflict_guard.insert_scatter_guard`) and owns as a
regular SDFG transient, so DaCe allocates it -- persistently, outside the timed program body.
Left unconnected, the node falls back to sizing a heap buffer from a runtime ``max(idx)``
sweep, which costs one extra full pass over ``idx`` plus an allocation per call.

Implementations (the ``_owner_out`` tag array is host memory in every backend; the CUDA
expansion therefore uses a device scratch buffer of its own and reads ``_owner_out`` only for
its size):

- ``Auto`` -- ``CPU``, ``CUDA`` or ``pure`` by the node's schedule (the default).
- ``pure`` -- the serial check as SDFG components (loop regions and Python tasklets), the single-core lowering.
- ``CPU``  -- tagged-write + verify, OpenMP-parallel; serial when the node is re-entered.
- ``CUDA`` -- the same tagged-write + verify run ON the device (``gpucub::BlockReduce`` fold, one
  atomic per block), with only the resulting flag copied back.
"""
from typing import Dict, List, Optional, Tuple

import dace
from dace import dtypes, library, nodes, symbolic
from dace.codegen.common import global_code_id, sym2cpp
from dace.libraries.standard.helper import schedule_dispatch
from dace.libraries.standard.pure_components import chain, counted_loop, element, operand_array, tasklet_state
from dace.memlet import Memlet
from dace.transformation.transformation import ExpandTransformation
from . import _helpers

INPUT_CONNECTOR_NAME = "_idx_in"
OUTPUT_CONNECTOR_NAME = "_count_out"
SCRATCH_CONNECTOR_NAME = "_owner_out"

#: Storage the tag array may use: every expansion runs the check in host code.
HOST_STORAGE = (dtypes.StorageType.Default, dtypes.StorageType.Register, dtypes.StorageType.CPU_Heap,
                dtypes.StorageType.CPU_ThreadLocal, dtypes.StorageType.CPU_Pinned)


def _validate(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG):
    """Resolve + check the edges; return ``(in_desc, in_name, out_name, owner_desc)``.

    ``owner_desc`` is the tag-array descriptor when the optional ``_owner_out`` connector is
    wired, else ``None``.
    """
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    out_edges = [e for e in state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME]
    if len(in_edges) != 1 or len(out_edges) != 1:
        raise ValueError(f"ScatterConflictCheck {node.label}: one '{INPUT_CONNECTOR_NAME}' in-edge + "
                         f"one '{OUTPUT_CONNECTOR_NAME}' out-edge required.")
    in_desc = sdfg.arrays[in_edges[0].data.data]
    out_desc = sdfg.arrays[out_edges[0].data.data]
    if not isinstance(in_desc, dace.data.Array) or not _helpers.is_integer_dtype(in_desc.dtype):
        raise ValueError(f"ScatterConflictCheck input must be an integer Array; got {in_desc}.")
    if out_desc.dtype != dtypes.int64:
        raise ValueError(f"ScatterConflictCheck output must be int64; got {out_desc.dtype}.")

    owner_edges = [e for e in state.out_edges(node) if e.src_conn == SCRATCH_CONNECTOR_NAME]
    if len(owner_edges) > 1:
        raise ValueError(f"ScatterConflictCheck {node.label}: at most one '{SCRATCH_CONNECTOR_NAME}' out-edge.")
    owner_desc = None
    if owner_edges:
        owner_desc = sdfg.arrays[owner_edges[0].data.data]
        if not isinstance(owner_desc, dace.data.Array) or owner_desc.dtype != dtypes.int64:
            raise ValueError(f"ScatterConflictCheck '{SCRATCH_CONNECTOR_NAME}' must be an int64 Array; "
                             f"got {owner_desc}.")
        # The check is host code in every expansion, so a device-resident tag array would be
        # dereferenced from the host. Refuse loudly rather than corrupt memory.
        if owner_desc.storage not in HOST_STORAGE:
            raise ValueError(f"ScatterConflictCheck '{SCRATCH_CONNECTOR_NAME}' must live in host memory; "
                             f"got storage {owner_desc.storage}.")
    return in_desc, in_edges[0].data.data, out_edges[0].data.data, owner_desc


def _length(node: "ScatterConflictCheck", state: dace.SDFGState) -> str:
    """C++ expression for the input length from the in-edge memlet."""
    in_edges = [e for e in state.in_edges(node) if e.dst_conn == INPUT_CONNECTOR_NAME]
    return sym2cpp(in_edges[0].data.subset.num_elements())


def _outputs(owner_desc: Optional[dace.data.Array]) -> Dict[str, None]:
    """Output connectors of the expanded tasklet (the tag array only when it is wired). A dict, not
    a set: connector iteration order is observable in the emitted code."""
    if owner_desc is None:
        return {OUTPUT_CONNECTOR_NAME: None}
    return {OUTPUT_CONNECTOR_NAME: None, SCRATCH_CONNECTOR_NAME: None}


def _owner(node: "ScatterConflictCheck", state: dace.SDFGState,
           owner_desc: Optional[dace.data.Array]) -> Optional[Tuple[str, str]]:
    """``(connector, capacity)`` for the wired tag array, or ``None``. Capacity comes from the
    edge's own subset, so it tracks whatever the caller actually sized the descriptor by."""
    if owner_desc is None:
        return None
    edge = next(e for e in state.out_edges(node) if e.src_conn == SCRATCH_CONNECTOR_NAME)
    return SCRATCH_CONNECTOR_NAME, sym2cpp(edge.data.subset.num_elements())


def _tagcount_call(n: str, src: str, omp: bool, owner: Optional[Tuple[str, str]]) -> str:
    """C++ calling :cpp:func:`dace::detect_collision` (``dace/runtime/include/dace/detect.h``).

    The tagged-write + verify algorithm, why two passes are the floor and why out-of-range values
    are skipped are all documented at the runtime function; keeping the loops there rather than in
    generated text means one implementation to tune and one place where the pragmas are reviewed.

    ``owner`` is ``(connector, capacity)`` for the caller-provided tag array, which must span the
    scattered array's domain. Without one the runtime sizes its own buffer from ``max(idx)``, which
    costs an extra reduction pass plus an allocation. ``omp`` toggles the parallel form.
    """
    par = 'true' if omp else 'false'
    if owner is None:
        call = f"dace::detect_collision({src}, ({n}), {par})"
    else:
        tag, cap = owner
        call = f"dace::detect_collision({src}, ({n}), {tag}, ({cap}), {par})"
    return f"{OUTPUT_CONNECTOR_NAME} = {call};\n"


#: Loop variable and runtime tag-array size of the ``pure`` check, symbols of its own nested SDFG.
CHECK_INDEX = 'conflict_i'
CAPACITY = 'conflict_capacity'
#: The tag array the ``pure`` check allocates itself when no ``_owner_out`` is wired.
OWN_TAGS = 'tags'


def index_loop(label: str, n: str, code: str, reads: Dict[str, Memlet], writes: Dict[str, Memlet],
               nsdfg: dace.SDFG) -> dace.sdfg.state.LoopRegion:
    """One pass over the index array: ``x`` is element ``CHECK_INDEX`` of ``_idx_in``."""
    loop = counted_loop(label, CHECK_INDEX, '0', n)
    body = tasklet_state(nsdfg, f'{label}_body', code, {
        'x': Memlet(element(nsdfg, INPUT_CONNECTOR_NAME, CHECK_INDEX)),
        **reads
    }, writes)
    chain(loop, [body])
    return loop


def sized_tags(nsdfg: dace.SDFG, n: str) -> list:
    """Blocks that size a tag array from ``max(idx)`` as the runtime does, ending on the edge that binds it."""
    nsdfg.add_scalar('top', dtypes.int64, transient=True)
    nsdfg.add_symbol(CAPACITY, dtypes.int64)
    nsdfg.add_array(OWN_TAGS, [CAPACITY], dtypes.int64, transient=True)
    start = tasklet_state(nsdfg, 'top_start', 'm = 0', {}, {'m': Memlet('top[0]')})
    sweep = index_loop('top_sweep', n, 'v = x\nm = v if v > t else t', {'t': Memlet('top[0]')}, {'m': Memlet('top[0]')},
                       nsdfg)
    return [start, sweep]


@library.expansion
class ExpandPure(ExpandTransformation):
    """The serial tagged-write + verify as SDFG components: tag every in-range slot with the last position that
    names it, then flag a position whose slot carries another position's tag."""

    environments: List[type] = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> dace.SDFG:
        _validate(node, state, sdfg)
        edges = {e.dst_conn: e for e in state.in_edges(node)}
        edges.update({e.src_conn: e for e in state.out_edges(node)})
        nsdfg = dace.SDFG(f'{node.label}_pure')
        for conn, edge in edges.items():
            operand_array(nsdfg, conn, edge, sdfg)
        n = symbolic.symstr(edges[INPUT_CONNECTOR_NAME].data.subset.num_elements())
        blocks = [] if SCRATCH_CONNECTOR_NAME in edges else sized_tags(nsdfg, n)
        tags = SCRATCH_CONNECTOR_NAME if SCRATCH_CONNECTOR_NAME in edges else OWN_TAGS
        capacity = symbolic.symstr(nsdfg.arrays[tags].total_size)
        whole = Memlet(f'{tags}[0:{capacity}]', dynamic=True)
        in_range = f'v >= 0 and v < {capacity}'
        nsdfg.add_scalar('flag', dtypes.int64, transient=True)
        tag = index_loop('tag', n, f'v = x\nif {in_range}:\n    t[v] = {CHECK_INDEX}', {}, {'t': whole}, nsdfg)
        clear = tasklet_state(nsdfg, 'clear', 'f = 0', {}, {'f': Memlet('flag[0]')})
        verify = index_loop('verify', n,
                            f'v = x\nnf = f\nif {in_range}:\n    if t[v] != {CHECK_INDEX}:\n        nf = 1', {
                                't': whole,
                                'f': Memlet('flag[0]')
                            }, {'nf': Memlet('flag[0]')}, nsdfg)
        done = tasklet_state(nsdfg, 'write_back', 'c = f', {'f': Memlet('flag[0]')},
                             {'c': Memlet(f'{OUTPUT_CONNECTOR_NAME}[0]')})
        chain(nsdfg, [*blocks, tag, clear, verify, done])
        if blocks:
            nsdfg.edges_between(blocks[-1], tag)[0].data.assignments[CAPACITY] = 'top + 1'
        return nsdfg


@library.expansion
class ExpandCPU(ExpandTransformation):
    """Tagged-write + verify, OpenMP-parallel (2 passes ~= 2x the scatter's own cost). A ``Sequential``
    node, or one re-entered by an enclosing parallel scope or loop, runs both passes on one thread."""

    environments: List[type] = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.transformation.auto.auto_optimize import libnode_is_sequential
        in_desc, _in, _out, owner_desc = _validate(node, state, sdfg)
        n = _length(node, state)
        owner = _owner(node, state, owner_desc)
        omp = node.schedule != dace.ScheduleType.Sequential and not libnode_is_sequential(node, state, sdfg)
        body = _tagcount_call(n, INPUT_CONNECTOR_NAME, omp=omp, owner=owner)
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME: None},
                             _outputs(owner_desc),
                             "{\n" + body + "}",
                             language=dace.Language.CPP)


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """Tagged-write + verify ON THE DEVICE: :cpp:func:`dace::detect_collision_device`.

    The index never leaves the GPU. Both passes are grid-stride kernels and the verify pass folds
    its per-lane flag through ``gpucub::BlockReduce`` into one atomic per block, the shape DaCe's own
    GPU WCR lowering emits; only the resulting flag crosses back, so the cost is two device passes
    plus one 8-byte copy instead of the whole index array.

    The tag buffer is DEVICE memory from the per-stream CUB scratch pool, not the ``_owner_out``
    array -- that one is host memory by :func:`_validate` and a device kernel cannot touch it. When
    it is wired its SUBSET is still what sizes the device buffer, so the caller keeps control of the
    domain; without it the runtime pays an extra device max-reduction and a round trip to size the
    buffer itself.

    The kernel launch lives in a ``DACE_EXPORTED`` wrapper appended to the device global code, the
    way the CUB libnodes do it, so the tasklet itself stays host code in the ``.cpp``.
    """

    # Filled in on first expansion to dodge the sort<->standard import cycle.
    environments: List[type] = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        if not ExpandCUDA.environments:
            from dace.libraries.sort.environments.cub import DetectScratch
            ExpandCUDA.environments = [DetectScratch]
        in_desc, _in, _out, owner_desc = _validate(node, state, sdfg)
        n, ct = _length(node, state), in_desc.dtype.ctype
        owner = _owner(node, state, owner_desc)

        idstr = global_code_id(sdfg, state, node)
        cap_param = '' if owner is None else ', long long __sc_capacity'
        cap_arg = '' if owner is None else ', __sc_capacity'
        prototype = (f'DACE_EXPORTED gpuError_t __dace_scatter_conflict_{idstr}(const {ct} *__sc_idx, '
                     f'long long __sc_n{cap_param}, long long *__sc_out, gpuStream_t __sc_stream);')
        sdfg.append_global_code(prototype + '\n')
        sdfg.append_global_code(
            f'{prototype}\n'
            f'gpuError_t __dace_scatter_conflict_{idstr}(const {ct} *__sc_idx, long long __sc_n{cap_param}, '
            f'long long *__sc_out, gpuStream_t __sc_stream) {{\n'
            f'    return ::dace::detect_collision_device(__sc_idx, __sc_n{cap_arg}, __sc_out, __sc_stream);\n'
            f'}}\n', 'cuda')

        cap_call = '' if owner is None else f', ({owner[1]})'
        code = ('{\n'
                'long long __sc_result;\n'
                f'DACE_GPU_CHECK(__dace_scatter_conflict_{idstr}({INPUT_CONNECTOR_NAME}, ({n}){cap_call}, '
                f'&__sc_result, __dace_current_stream));\n'
                f'{OUTPUT_CONNECTOR_NAME} = __sc_result;\n'
                '}')
        return nodes.Tasklet(node.name, {INPUT_CONNECTOR_NAME: None},
                             _outputs(owner_desc),
                             code,
                             language=dace.Language.CPP)


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Picks ``CPU``, ``CUDA`` or ``pure`` from the node's schedule (:func:`schedule_dispatch`)."""

    environments: List[type] = []

    @staticmethod
    def expansion(node: "ScatterConflictCheck", state: dace.SDFGState, sdfg: dace.SDFG):
        return schedule_dispatch(ExpandAuto, node, state)


@library.node
class ScatterConflictCheck(nodes.LibraryNode):
    """Flag duplicate values in a 1-D integer index array (scatter no-conflict proof).

    - ``_idx_in``: input 1-D integer index array of length ``N``.
    - ``_count_out``: output length-1 ``int64`` **host** scalar, ``0`` iff ``_idx_in`` is a
      permutation and ``1`` otherwise. The verify pass OR-reduces, so this is a FLAG, not a
      duplicate count -- an OR carries per lane where a sum needs a widening accumulator, and
      the only consumer is the trap's ``> 0``. Host in every backend so the guard's
      ``trap_sym`` binding + trap execute on the host.
    - ``_owner_out`` (optional): scratch tag array, an ``int64`` **host** Array spanning
      every value ``_idx_in`` can hold. Wiring it lets DaCe own the allocation (so it can
      be persistent, outside the program body) and drops the ``max(idx)`` sizing pass.
      Contents are never read before being written, so it needs no initialization and may
      carry stale values from a previous call.
    """

    INPUT_CONNECTOR_NAME = INPUT_CONNECTOR_NAME
    OUTPUT_CONNECTOR_NAME = OUTPUT_CONNECTOR_NAME
    SCRATCH_CONNECTOR_NAME = SCRATCH_CONNECTOR_NAME

    #: Both stay on the host in EVERY expansion, the CUDA one included -- see ``_validate`` and
    #: :class:`ExpandCUDA`. Declared so an offloader does not move them with the rest of the state.
    host_connectors = frozenset({OUTPUT_CONNECTOR_NAME, SCRATCH_CONNECTOR_NAME})

    implementations = {"Auto": ExpandAuto, "CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure}
    default_implementation = "Auto"

    def __init__(self, name: str = "ScatterConflictCheck", *args, **kwargs):
        super().__init__(name, *args, inputs={INPUT_CONNECTOR_NAME}, outputs={OUTPUT_CONNECTOR_NAME}, **kwargs)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState):
        _validate(self, state, sdfg)
