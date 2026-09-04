# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""":class:`FindFirst`: the smallest index in a range at which a predicate holds.

The lowered form of an early-exit loop (``for i: if cond(i): break``), which
:class:`~dace.transformation.passes.canonicalize.early_exit_to_find_index.EarlyExitToFindIndex`
lifts. It is an argmin over the firing indices, but a SHORT-CIRCUITING one: the value of a
find-first is that the range past the answer is never read, which a plain
:class:`~dace.libraries.standard.nodes.arg_reduce.ArgReduce` over a materialized
``i if cond(i) else N`` array cannot give -- that form pays a whole extra array and three full
sweeps to compute what one cancelling sweep already knows.

The CPU expansions call :cpp:func:`dace::find_first_index` (``dace/runtime/include/dace/detect.h``)
and differ only in whether the chunks go to OpenMP; the CUDA expansion calls
:cpp:func:`dace::find_first_index_device` (``dace/runtime/include/dace/cuda/detect.cuh``), a
cancelling block-per-tile argmin folded by ``gpucub::BlockReduce``. Each runtime function owns its
chunking, its shared cancellation state and its reduction, so there is one implementation per
machine to tune rather than a copy per expansion.

The node is opaque on purpose: the tile vectorizer never looks inside a library node, so the
search neither perturbs nor is perturbed by the tiling of the Maps around it.
"""
from typing import Dict, List, Optional, Tuple

import dace
from dace import library, properties, symbolic
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

#: The index the predicate expression is written against.
INDEX_NAME = '__i'

#: Scalar out-connector carrying the answer (or ``end`` when the predicate never holds).
OUTPUT_CONNECTOR_NAME = '_out_idx'


def find_first_code(node: "FindFirst", parallel: bool) -> str:
    """The expansion body: one :cpp:func:`dace::find_first_index` call over ``node.predicate``.

    The predicate becomes the body of a ``[&]`` lambda taking :data:`INDEX_NAME`, so it reads its
    arrays through the node's own in-connectors -- the pointer arrives on a connector and the
    subscript is generated here, inside the library expansion, which is the one place a hand-written
    index belongs."""
    from dace.codegen.targets.cpp import sym2cpp
    begin, end = sym2cpp(node.begin), sym2cpp(node.end)
    par = 'true' if parallel else 'false'
    return (f"{OUTPUT_CONNECTOR_NAME} = dace::find_first_index(({begin}), ({end}), "
            f"[&](long long {INDEX_NAME}) -> bool {{ return ({node.predicate}); }}, {par});")


def find_first_connectors(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> Dict[str, object]:
    """In-connector types for the expanded tasklet: a pointer per array read, by value per scalar.

    An array is handed to the lambda whole and indexed inside it, so its connector must be a
    pointer; a loop-invariant scalar is read once and stays a value connector."""
    conns: Dict[str, object] = {}
    for edge in state.in_edges(node):
        if edge.dst_conn is None:
            continue
        desc = sdfg.arrays[edge.data.data]
        conns[edge.dst_conn] = None if isinstance(desc, dace.data.Scalar) else dace.pointer(desc.dtype)
    return conns


#: Storages whose pointers only a device kernel may dereference.
GPU_STORAGE = (dace.dtypes.StorageType.GPU_Global, dace.dtypes.StorageType.GPU_Shared)


def find_first_reads_device_memory(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> bool:
    """Whether any input the predicate reads lives in GPU memory.

    The expansions differ in which machine dereferences those pointers, and nothing else picks
    between them -- ``apply_gpu_transformations`` sets a library node's SCHEDULE but never its
    implementation. So each expansion checks, and a mismatch is refused with the knob to turn
    rather than silently reading a device pointer from the host."""
    return any(sdfg.arrays[e.data.data].storage in GPU_STORAGE for e in state.in_edges(node) if e.dst_conn is not None)


def refuse_wrong_machine(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG, want_device: bool) -> None:
    """Refuse an expansion whose machine does not match where the predicate's inputs live."""
    on_device = find_first_reads_device_memory(node, state, sdfg)
    if on_device and not want_device:
        raise NotImplementedError(f"{node.label}: FindFirst reads GPU memory; set implementation='CUDA' on the node "
                                  "(a host expansion would dereference a device pointer).")
    if want_device and not on_device:
        raise NotImplementedError(f"{node.label}: FindFirst(CUDA) reads host memory; use the 'OpenMP' or 'pure' "
                                  "implementation (a device kernel cannot dereference a host pointer).")


@library.expansion
class ExpandFindFirstPure(ExpandTransformation):
    """Serial CPU lowering: :cpp:func:`dace::find_first_index` with the chunk loop unthreaded.

    Still blocked and ``simd``-scanned inside a chunk, and still cancels between chunks -- serial
    here means one thread, not one element at a time."""

    environments = []

    @staticmethod
    def expansion(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        node.validate(sdfg, state)
        refuse_wrong_machine(node, state, sdfg, want_device=False)
        return nodes.Tasklet(f'{node.label}_pure',
                             find_first_connectors(node, state, sdfg), {OUTPUT_CONNECTOR_NAME: None},
                             find_first_code(node, parallel=False),
                             language=dace.dtypes.Language.CPP)


@library.expansion
class ExpandFindFirstOpenMP(ExpandTransformation):
    """Parallel CPU lowering: chunks handed out ``schedule(dynamic, 1)``, cancelling on a shared
    hint. See :cpp:func:`dace::find_first_index` for why the hint's race is benign."""

    environments = []

    @staticmethod
    def expansion(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        node.validate(sdfg, state)
        refuse_wrong_machine(node, state, sdfg, want_device=False)
        return nodes.Tasklet(f'{node.label}_openmp',
                             find_first_connectors(node, state, sdfg), {OUTPUT_CONNECTOR_NAME: None},
                             find_first_code(node, parallel=True),
                             language=dace.dtypes.Language.CPP)


def find_first_signature(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> List[Tuple[str, str]]:
    """``(connector, C++ declaration)`` per in-connector, sorted by connector name.

    An array read is a pointer the predicate subscripts; a loop-invariant scalar is passed by
    value. The one list drives the functor's members, the wrapper's parameters and the call site,
    so those three cannot drift apart, and sorting keeps the emitted text independent of edge
    insertion order."""
    out: List[Tuple[str, str]] = []
    for edge in state.in_edges(node):
        if edge.dst_conn is None:
            continue
        desc = sdfg.arrays[edge.data.data]
        if isinstance(desc, dace.data.Scalar):
            out.append((edge.dst_conn, f'{desc.dtype.ctype} {edge.dst_conn}'))
        else:
            out.append((edge.dst_conn, f'const {desc.dtype.ctype} *{edge.dst_conn}'))
    return sorted(out)


@library.expansion
class ExpandFindFirstCUDA(ExpandTransformation):
    """Device lowering: :cpp:func:`dace::find_first_index_device`, one ``gpucub::BlockReduce`` argmin
    per tile and one ``atomicMin`` per firing tile.

    The predicate becomes a device FUNCTOR struct appended to the device global code, not a device
    lambda: a lambda written in the host translation unit would need ``--extended-lambda`` on every
    build. The wrapper next to it is the same ``DACE_EXPORTED`` shape the CUB libnodes use -- it is
    what puts the kernel launch in the ``.cu`` while the tasklet stays host code."""

    # Filled in on first expansion to dodge the sort<->standard import cycle.
    environments = []

    @staticmethod
    def expansion(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.codegen.targets.cpp import sym2cpp
        if not ExpandFindFirstCUDA.environments:
            from dace.libraries.sort.environments.cub import DetectScratch
            ExpandFindFirstCUDA.environments = [DetectScratch]
        node.validate(sdfg, state)
        refuse_wrong_machine(node, state, sdfg, want_device=True)

        state_id = state.parent_graph.node_id(state)
        idstr = f'{sdfg.name}_{state_id}_{state.node_id(node)}'
        signature = find_first_signature(node, state, sdfg)
        members = '\n'.join(f'    {decl};' for _conn, decl in signature)
        params = ', '.join(decl for _conn, decl in signature)
        args = ', '.join(conn for conn, _decl in signature)
        prototype = (f'DACE_EXPORTED gpuError_t __dace_findfirst_{idstr}({params}, long long __ff_begin, '
                     f'long long __ff_end, long long *__ff_out, gpuStream_t __ff_stream);')

        sdfg.append_global_code(prototype + '\n')
        sdfg.append_global_code(
            f'struct __ff_pred_{idstr} {{\n'
            f'{members}\n'
            f'    __device__ __forceinline__ bool operator()(long long {INDEX_NAME}) const '
            f'{{ return ({node.predicate}); }}\n'
            f'}};\n'
            f'{prototype}\n'
            f'gpuError_t __dace_findfirst_{idstr}({params}, long long __ff_begin, long long __ff_end, '
            f'long long *__ff_out, gpuStream_t __ff_stream) {{\n'
            f'    __ff_pred_{idstr} __ff_pred{{{args}}};\n'
            f'    return ::dace::find_first_index_device(__ff_begin, __ff_end, __ff_pred, __ff_out, __ff_stream);\n'
            f'}}\n', 'cuda')

        begin, end = sym2cpp(node.begin), sym2cpp(node.end)
        code = (f'long long __ff_result;\n'
                f'DACE_GPU_CHECK(__dace_findfirst_{idstr}({args}, ({begin}), ({end}), &__ff_result, '
                f'__dace_current_stream));\n'
                f'{OUTPUT_CONNECTOR_NAME} = __ff_result;')
        return nodes.Tasklet(f'{node.label}_cuda',
                             find_first_connectors(node, state, sdfg), {OUTPUT_CONNECTOR_NAME: None},
                             code,
                             language=dace.dtypes.Language.CPP)


@library.node
class FindFirst(nodes.LibraryNode):
    """Smallest ``i`` in ``[begin, end)`` with ``predicate`` true, or ``begin >= end``'s ``end``.

    :cvar implementations: ``"OpenMP"`` (parallel chunked search, the default), ``"pure"``
        (the same search on one thread) and ``"CUDA"`` (the device search).
    """

    implementations = {
        'pure': ExpandFindFirstPure,
        'OpenMP': ExpandFindFirstOpenMP,
        'CUDA': ExpandFindFirstCUDA,
    }
    default_implementation = 'OpenMP'

    #: The ANSWER is a host scalar in every expansion, the CUDA one included: the device search
    #: leaves its result in CUB scratch and ``find_first_index_device`` copies it back and writes
    #: ``*out`` on the host. Promoting it to device memory makes host code write a device pointer --
    #: which validates, then corrupts. Declared so an offloader keeps it where the expansion writes.
    host_connectors = frozenset({OUTPUT_CONNECTOR_NAME})

    predicate = properties.Property(dtype=str,
                                    default='false',
                                    desc="C++ predicate over the in-connectors, indexed by "
                                    f"'{INDEX_NAME}'. True at the index the search returns.")
    begin = properties.SymbolicProperty(default=0, desc='First index to test.')
    end = properties.SymbolicProperty(default=0, desc='One past the last index to test; the no-hit answer.')

    def __init__(self,
                 name: str,
                 predicate: str = 'false',
                 begin: symbolic.SymbolicType = 0,
                 end: symbolic.SymbolicType = 0,
                 location: Optional[str] = None):
        super().__init__(name, location=location, inputs={}, outputs={OUTPUT_CONNECTOR_NAME: None})
        self.predicate = predicate
        self.begin = begin
        self.end = end

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Require the index output and a predicate that actually names the index.

        A predicate free of :data:`INDEX_NAME` is constant over the range, so the search would
        answer ``begin`` or ``end`` without reading anything -- always a lifting bug upstream, and
        silent if it is not refused here."""
        out_conns = {e.src_conn for e in state.out_edges(self) if e.src_conn is not None}
        if out_conns != {OUTPUT_CONNECTOR_NAME}:
            raise ValueError(f"{self.label}: FindFirst requires exactly one output "
                             f"'{OUTPUT_CONNECTOR_NAME}', got {sorted(out_conns)}")
        if INDEX_NAME not in self.predicate:
            raise ValueError(f"{self.label}: FindFirst predicate does not read the index "
                             f"'{INDEX_NAME}': {self.predicate!r}")
        for edge in state.in_edges(self):
            if edge.dst_conn is None:
                raise ValueError(f"{self.label}: FindFirst input edge from {edge.src} has no connector")
