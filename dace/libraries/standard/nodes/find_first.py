# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""":class:`FindFirst`: the smallest index in a range at which a predicate holds.

The lowered form of an early-exit loop (``for i: if cond(i): break``), built directly by a caller
that already knows the predicate. It is an argmin over the firing indices, but a SHORT-CIRCUITING
one: the value of a find-first is that the range past the answer is never read, which a plain
:class:`~dace.libraries.standard.nodes.arg_reduce.ArgReduce` over a materialized
``i if cond(i) else N`` array cannot give -- that form pays a whole extra array and three full
sweeps to compute what one cancelling sweep already knows.

The predicate is a Python expression; the C++ expansions spell it in C++ and linearize each array
subscript by the array's strides, and the ``pure`` expansion tests it in a Python tasklet inside a
sequential loop that breaks at the first hit.

The CPU expansions call :cpp:func:`dace::find_first_index` (``dace/runtime/include/dace/detect.h``)
and differ only in whether the chunks go to OpenMP; the CUDA expansion calls
:cpp:func:`dace::find_first_index_device` (``dace/runtime/include/dace/cuda/detect.cuh``), a
cancelling block-per-tile argmin folded by ``gpucub::BlockReduce``. Each runtime function owns its
chunking, its shared cancellation state and its reduction, so there is one implementation per
machine to tune rather than a copy per expansion.

The node is opaque on purpose: the tile vectorizer never looks inside a library node, so the
search neither perturbs nor is perturbed by the tiling of the Maps around it.
"""

import ast

import dace
from dace import library, properties, symbolic
from dace.codegen.common import global_code_id
from dace.codegen.cppunparse import pyexpr2cpp
from dace.libraries.standard.helper import schedule_dispatch
from dace.libraries.standard.pure_components import chain, counted_loop, operand_array, tasklet_state
from dace.memlet import Memlet
from dace.optionals import required
from dace.sdfg import nodes
from dace.sdfg.state import BreakBlock, ConditionalBlock, ControlFlowRegion
from dace.transformation.transformation import ExpandTransformation

#: The index the predicate expression is written against.
INDEX_NAME = "__i"

#: Scalar out-connector carrying the answer (or ``end`` when the predicate never holds).
OUTPUT_CONNECTOR_NAME = "_out_idx"


class Linearize(ast.NodeTransformer):
    """Rewrites each subscript of an array connector into one offset from its base pointer."""

    def __init__(self, strides: dict[str, tuple]) -> None:
        self.strides = strides

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id not in self.strides:
            return node
        strides = self.strides[node.value.id]
        index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        if len(index) != len(strides):
            raise ValueError(f"the predicate subscripts {node.value.id} with {len(index)} of its {len(strides)} dims")
        if len(index) == 1 and strides[0] == 1:
            return node
        flat = " + ".join(
            f"({ast.unparse(dim)}) * ({symbolic.symstr(stride, cpp_mode=False)})" for dim, stride in zip(index, strides)
        )
        return ast.Subscript(node.value, ast.parse(flat, mode="eval").body, node.ctx)


class ElementReads(ast.NodeTransformer):
    """Rewrites each subscript of an array connector into a connector of its own, read through an element memlet."""

    def __init__(self, arrays: set[str]) -> None:
        self.arrays = arrays
        self.memlets: dict[str, Memlet] = {}

    def visit_Subscript(self, node: ast.Subscript) -> ast.expr:
        self.generic_visit(node)
        if not isinstance(node.value, ast.Name) or node.value.id not in self.arrays:
            return node
        index = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        name = f"__ff_read{len(self.memlets)}"
        self.memlets[name] = Memlet(f"{node.value.id}[{', '.join(ast.unparse(dim) for dim in index)}]")
        return ast.Name(name, ast.Load())


def parenthesized(cpp: str) -> str:
    """``cpp`` inside one pair of parentheses: as given when one pair already encloses all of it."""
    depth = 0
    for position, char in enumerate(cpp):
        depth += {"(": 1, ")": -1}.get(char, 0)
        if depth == 0 and position < len(cpp) - 1:
            return f"({cpp})"
    return cpp if cpp.startswith("(") else f"({cpp})"


def cpp_predicate(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> str:
    """``node.predicate`` in C++, parenthesized, each array subscript one offset by the array's strides.

    The pointer arrives on a connector and the subscript is generated here, inside the library
    expansion, which is the one place a hand-written index belongs."""
    strides = {
        edge.dst_conn: tuple(sdfg.arrays[required(edge.data.data)].strides)
        for edge in state.in_edges(node)
        if edge.dst_conn is not None and not isinstance(sdfg.arrays[required(edge.data.data)], dace.data.Scalar)
    }
    tree = Linearize(strides).visit(ast.parse(node.predicate, mode="eval"))
    return parenthesized(pyexpr2cpp(ast.unparse(tree)))


def find_first_code(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG, parallel: bool) -> str:
    """The expansion body: one :cpp:func:`dace::find_first_index` call over the predicate, which becomes
    the body of a ``[&]`` lambda taking :data:`INDEX_NAME` and reads its arrays through the node's own
    in-connectors."""
    from dace.codegen.targets.cpp import sym2cpp

    begin, end = sym2cpp(node.begin), sym2cpp(node.end)
    par = "true" if parallel else "false"
    return (
        f"{OUTPUT_CONNECTOR_NAME} = dace::find_first_index(({begin}), ({end}), "
        f"[&](long long {INDEX_NAME}) -> bool {{ return {cpp_predicate(node, state, sdfg)}; }}, {par});"
    )


def find_first_connectors(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> dict[str, object]:
    """In-connector types for the expanded tasklet: a pointer per array read, by value per scalar.

    An array is handed to the lambda whole and indexed inside it, so its connector must be a
    pointer; a loop-invariant scalar is read once and stays a value connector."""
    conns: dict[str, object] = {}
    for edge in state.in_edges(node):
        if edge.dst_conn is None:
            continue
        desc = sdfg.arrays[required(edge.data.data)]
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
    return any(
        required(sdfg.arrays[required(e.data.data)]).storage in GPU_STORAGE
        for e in state.in_edges(node)
        if e.dst_conn is not None
    )


def refuse_wrong_machine(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG, want_device: bool) -> None:
    """Refuse an expansion whose machine does not match where the predicate's inputs live."""
    on_device = find_first_reads_device_memory(node, state, sdfg)
    if on_device and not want_device:
        raise NotImplementedError(
            f"{node.label}: FindFirst reads GPU memory; set implementation='CUDA' on the node "
            "(a host expansion would dereference a device pointer)."
        )
    if want_device and not on_device:
        raise NotImplementedError(
            f"{node.label}: FindFirst(CUDA) reads host memory; use the 'CPU' "
            "implementation (a device kernel cannot dereference a host pointer)."
        )


@library.expansion
class ExpandFindFirstCPU(ExpandTransformation):
    """CPU lowering: chunks handed out ``schedule(dynamic, 1)``, cancelling on a shared hint. See
    :cpp:func:`dace::find_first_index` for why the hint's race is benign. A ``Sequential`` node, or one
    re-entered by an enclosing parallel scope (or a loop, unless the node opts out), runs the chunk loop on one thread: still
    blocked, ``simd``-scanned and cancelling between chunks."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.transformation.auto.auto_optimize import libnode_is_sequential

        node.validate(sdfg, state)
        refuse_wrong_machine(node, state, sdfg, want_device=False)
        parallel = node.schedule != dace.ScheduleType.Sequential and not libnode_is_sequential(node, state, sdfg)
        return nodes.Tasklet(
            f"{node.label}_openmp",
            find_first_connectors(node, state, sdfg),
            {OUTPUT_CONNECTOR_NAME: None},
            find_first_code(node, state, sdfg, parallel=parallel),
            language=dace.dtypes.Language.CPP,
        )


@library.expansion
class ExpandFindFirstPure(ExpandTransformation):
    """The sequential search as SDFG components: the answer starts at ``end``, and a loop over
    ``[begin, end)`` tests the predicate in a Python tasklet, writing the index and breaking at the first hit.
    Each array is the whole container, and each element the predicate reads arrives through its own memlet."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "FindFirst", parent_state: dace.SDFGState, parent_sdfg: dace.SDFG) -> dace.SDFG:
        node.validate(parent_sdfg, parent_state)
        refuse_wrong_machine(node, parent_state, parent_sdfg, want_device=False)
        nsdfg = dace.SDFG(f"{node.label}_pure")
        scalars: dict[str, Memlet] = {}
        for edge in parent_state.in_edges(node):
            if edge.dst_conn is None:
                continue
            desc = parent_sdfg.arrays[required(edge.data.data)]
            if isinstance(desc, dace.data.Scalar):
                operand_array(nsdfg, edge.dst_conn, edge, parent_sdfg)
                scalars[edge.dst_conn] = Memlet(f"{edge.dst_conn}[0]")
            else:
                nsdfg.add_array(edge.dst_conn, desc.shape, desc.dtype, strides=desc.strides, storage=desc.storage)
        elements = ElementReads(set(nsdfg.arrays) - set(scalars))
        predicate = ast.unparse(elements.visit(ast.parse(node.predicate, mode="eval")))
        out_edge = next(e for e in parent_state.out_edges(node) if e.src_conn == OUTPUT_CONNECTOR_NAME)
        operand_array(nsdfg, OUTPUT_CONNECTOR_NAME, out_edge, parent_sdfg)
        nsdfg.add_scalar("__ff_hit", dace.bool_, transient=True)
        answer = {"o": Memlet(f"{OUTPUT_CONNECTOR_NAME}[0]")}
        end = symbolic.symstr(node.end, cpp_mode=False)
        none = tasklet_state(nsdfg, "no_hit", f"o = {end}", {}, answer)
        reads = {**scalars, **elements.memlets}
        test = tasklet_state(nsdfg, "test", f"hit = {predicate}", reads, {"hit": Memlet("__ff_hit[0]")})
        found = ControlFlowRegion("found")
        chain(found, [tasklet_state(nsdfg, "hit", f"o = {INDEX_NAME}", {}, answer), BreakBlock("stop")])
        branch = ConditionalBlock("on_hit")
        branch.add_branch("__ff_hit", found)
        loop = counted_loop("search", INDEX_NAME, symbolic.symstr(node.begin, cpp_mode=False), end)
        chain(loop, [test, branch])
        chain(nsdfg, [none, loop])
        return nsdfg


@library.expansion
class ExpandFindFirstAuto(ExpandTransformation):
    """Picks ``CPU`` or ``CUDA`` from the node's schedule (:func:`schedule_dispatch`)."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG):
        return schedule_dispatch(ExpandFindFirstAuto, node, state)


def find_first_signature(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> list[tuple[str, str]]:
    """``(connector, C++ declaration)`` per in-connector, sorted by connector name.

    An array read is a pointer the predicate subscripts; a loop-invariant scalar is passed by
    value. The one list drives the functor's members, the wrapper's parameters and the call site,
    so those three cannot drift apart, and sorting keeps the emitted text independent of edge
    insertion order."""
    out: list[tuple[str, str]] = []
    for edge in state.in_edges(node):
        if edge.dst_conn is None:
            continue
        desc = sdfg.arrays[required(edge.data.data)]
        if isinstance(desc, dace.data.Scalar):
            out.append((edge.dst_conn, f"{desc.dtype.ctype} {edge.dst_conn}"))
        else:
            out.append((edge.dst_conn, f"const {desc.dtype.ctype} *{edge.dst_conn}"))
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
    environments: list[type] = []

    @staticmethod
    def expansion(node: "FindFirst", state: dace.SDFGState, sdfg: dace.SDFG) -> nodes.Tasklet:
        from dace.codegen.targets.cpp import sym2cpp

        if not ExpandFindFirstCUDA.environments:
            from dace.libraries.sort.environments.cub import DetectScratch

            ExpandFindFirstCUDA.environments = [DetectScratch]
        node.validate(sdfg, state)
        refuse_wrong_machine(node, state, sdfg, want_device=True)

        idstr = global_code_id(sdfg, state, node)
        signature = find_first_signature(node, state, sdfg)
        members = "\n".join(f"    {decl};" for _conn, decl in signature)
        params = ", ".join(decl for _conn, decl in signature)
        args = ", ".join(conn for conn, _decl in signature)
        prototype = (
            f"DACE_EXPORTED gpuError_t __dace_findfirst_{idstr}({params}, long long __ff_begin, "
            f"long long __ff_end, long long *__ff_out, gpuStream_t __ff_stream);"
        )

        sdfg.append_global_code(prototype + "\n")
        sdfg.append_global_code(
            f"struct __ff_pred_{idstr} {{\n"
            f"{members}\n"
            f"    __device__ __forceinline__ bool operator()(long long {INDEX_NAME}) const "
            f"{{ return {cpp_predicate(node, state, sdfg)}; }}\n"
            f"}};\n"
            f"{prototype}\n"
            f"gpuError_t __dace_findfirst_{idstr}({params}, long long __ff_begin, long long __ff_end, "
            f"long long *__ff_out, gpuStream_t __ff_stream) {{\n"
            f"    __ff_pred_{idstr} __ff_pred{{{args}}};\n"
            f"    return ::dace::find_first_index_device(__ff_begin, __ff_end, __ff_pred, __ff_out, __ff_stream);\n"
            f"}}\n",
            "cuda",
        )

        begin, end = sym2cpp(node.begin), sym2cpp(node.end)
        code = (
            f"long long __ff_result;\n"
            f"DACE_GPU_CHECK(__dace_findfirst_{idstr}({args}, ({begin}), ({end}), &__ff_result, "
            f"__dace_current_stream));\n"
            f"{OUTPUT_CONNECTOR_NAME} = __ff_result;"
        )
        return nodes.Tasklet(
            f"{node.label}_cuda",
            find_first_connectors(node, state, sdfg),
            {OUTPUT_CONNECTOR_NAME: None},
            code,
            language=dace.dtypes.Language.CPP,
        )


@library.node
class FindFirst(nodes.LibraryNode):
    """Smallest ``i`` in ``[begin, end)`` with ``predicate`` true, or ``begin >= end``'s ``end``.

    :cvar implementations: ``"Auto"`` (picks by schedule, the default), ``"CPU"`` (chunked OpenMP
        search), ``"CUDA"`` (the device search) and ``"pure"`` (a sequential loop of SDFG components).
    """

    implementations = {
        "Auto": ExpandFindFirstAuto,
        "CPU": ExpandFindFirstCPU,
        "CUDA": ExpandFindFirstCUDA,
        "pure": ExpandFindFirstPure,
    }
    default_implementation = "Auto"

    #: The ANSWER is a host scalar in every expansion, the CUDA one included: the device search
    #: leaves its result in CUB scratch and ``find_first_index_device`` copies it back and writes
    #: ``*out`` on the host. Promoting it to device memory makes host code write a device pointer --
    #: which validates, then corrupts. Declared so an offloader keeps it where the expansion writes.
    host_connectors = frozenset({OUTPUT_CONNECTOR_NAME})

    #: An enclosing loop is not a reason to search on one thread: a lifted early exit's chunk loop
    #: re-enters the search once per chunk of work, which the fork amortizes as it does a map's.
    serial_on_loop_reentry = False

    predicate = properties.Property(
        dtype=str,
        default="false",
        desc=f"Python predicate over the in-connectors, indexed by '{INDEX_NAME}'; an array is subscripted per "
        "dimension. True at the index the search returns.",
    )
    symbols = properties.ListProperty(
        element_type=str, default=[], desc="Symbols the predicate reads by name, so they reach the node's scope."
    )
    begin = properties.SymbolicProperty(default=0, desc="First index to test.")
    end = properties.SymbolicProperty(default=0, desc="One past the last index to test; the no-hit answer.")

    def __init__(
        self,
        name: str,
        predicate: str = "false",
        begin: symbolic.SymbolicType = 0,
        end: symbolic.SymbolicType = 0,
        location: str | None = None,
    ):
        super().__init__(name, location=location, inputs={}, outputs={OUTPUT_CONNECTOR_NAME: None})
        self.predicate = predicate
        self.begin = begin
        self.end = end

    @property
    def free_symbols(self) -> set[str]:
        return super().free_symbols | set(self.symbols)

    def validate(self, sdfg: dace.SDFG, state: dace.SDFGState) -> None:
        """Require the index output and a predicate that actually names the index.

        A predicate free of :data:`INDEX_NAME` is constant over the range, so the search would
        answer ``begin`` or ``end`` without reading anything -- always a lifting bug upstream, and
        silent if it is not refused here."""
        out_conns = {e.src_conn for e in state.out_edges(self) if e.src_conn is not None}
        if out_conns != {OUTPUT_CONNECTOR_NAME}:
            raise ValueError(
                f"{self.label}: FindFirst requires exactly one output "
                f"'{OUTPUT_CONNECTOR_NAME}', got {sorted(out_conns)}"
            )
        if INDEX_NAME not in self.predicate:
            raise ValueError(
                f"{self.label}: FindFirst predicate does not read the index '{INDEX_NAME}': {self.predicate!r}"
            )
        for edge in state.in_edges(self):
            if edge.dst_conn is None:
                raise ValueError(f"{self.label}: FindFirst input edge from {edge.src} has no connector")
