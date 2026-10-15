# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StateScan``: the sequence of states a finite-state recurrence passes through, computed as a parallel scan.

The recurrence is ``s[k+1] = f(s[k], k)`` for ``k`` in ``[begin, end)``, where ``f`` is the ``step`` SDFG: a region
that reads the symbols ``state`` and ``iterate`` and assigns ``state``. :class:`Scan` carries a value through a
fixed operator; this node carries a STATE through arbitrary code, which only composes cheaply when the state is
finite. It takes ``count`` values starting at ``base``
(:mod:`~dace.transformation.passes.analysis.finite_state` proves that), so the effect of a block of iterations is a
table of ``count`` end states, one per start state, and two tables compose by lookup.

Output ``_out[k - begin]`` is the state before iteration ``k``, and ``_out[end - begin]`` the state after the last.

Every expansion is one SDFG in three phases, scheduled for the device:

1. ``chunks`` blocks run the step from each of the ``count`` start states and record the end state: a map over
   (block, start state), each point a sequential loop over its block.
2. One thread composes the tables from ``_init``: ``starts[c + 1] = table[c, starts[c] - base]``.
3. The blocks run again, each from its true start state, and record every state.

Work is ``(count + 1) * N`` steps against ``N`` for the loop, at depth ``(count + 1) * N / chunks + chunks``.
"""

import copy

import dace
from dace import SDFG, dtypes, library, symbolic
from dace.libraries.standard.helper import schedule_dispatch
from dace.libraries.standard.pure_components import chain, counted_loop
from dace.memlet import Memlet
from dace.properties import Property, SDFGReferenceProperty, SymbolicProperty
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

INIT_CONNECTOR = "_init"
#: Threads per GPU thread block for the block maps.
GPU_BLOCK_THREADS = 128
OUT_CONNECTOR = "_out"

#: Recorded on every node: the device trade of the scan against the loop it replaces.
FINITE_STATE_SCAN_HINT = (
    "finite-state scan; canonicalization takes the parallel form.\n"
    "Alternative: the sequential loop over the state.\n"
    "CPU: the loop is usually the better of the two unless the cores outnumber the states -- every block runs "
    "once per state.\n"
    "GPU: the scan is usually the better of the two.\n"
    "Both are correct. Measure before choosing."
)


def chunk_bounds(node: "StateScan") -> tuple[str, str]:
    """The first and one-past-last iteration of block ``__c``."""
    length = symbolic.symstr(node.end - node.begin, cpp_mode=False)
    chunks = symbolic.symstr(node.chunks, cpp_mode=False)
    begin = symbolic.symstr(node.begin, cpp_mode=False)
    end = symbolic.symstr(node.end, cpp_mode=False)
    size = f"int_ceil({length}, {chunks})"
    return f"{begin} + __c * {size}", f"min({begin} + (__c + 1) * {size}, {end})"


def argument(outer: SDFG, name: str) -> dace.data.Data:
    """``outer``'s container ``name`` as a nested SDFG's non-transient argument."""
    desc = copy.deepcopy(outer.arrays[name])
    desc.transient = False
    return desc


def declare_operands(node: "StateScan", sdfg: SDFG, outer: SDFG, state: dace.SDFGState) -> None:
    """The node's tables, ``_init`` and ``_out`` as non-transient containers of ``sdfg``."""
    for edge in state.in_edges(node) + state.out_edges(node):
        name = edge.dst_conn if edge.dst is node else edge.src_conn
        sdfg.add_datadesc(name, argument(outer, edge.data.data))


def run_sdfg(node: "StateScan", label: str, outer: SDFG, start: str, record_states: bool) -> SDFG:
    """One block's sequential run of the step from ``start``: it records every state into ``_out`` when
    ``record_states``, else its end state into ``table[__c, __j]``."""
    inner = SDFG(label)
    step = copy.deepcopy(node.step)
    for name, desc in step.arrays.items():
        inner.add_datadesc(name, copy.deepcopy(desc))
    block_symbols = [("__c", dtypes.int64)] if record_states else [("__c", dtypes.int64), ("__j", dtypes.int64)]
    for name, dtype in [*step.symbols.items(), *block_symbols]:
        if name not in inner.symbols:
            inner.add_symbol(name, dtype)
    target = OUT_CONNECTOR if record_states else "table"
    inner.add_datadesc(target, argument(outer, target))
    if start == "starts":
        inner.add_datadesc("starts", argument(outer, "starts"))
    first, last = chunk_bounds(node)
    loop = counted_loop(f"{label}_blocks", node.iterate, first, last)
    body: list = []
    if record_states:
        record = dace.SDFGState(f"{label}_record", sdfg=inner)
        write = record.add_tasklet("record", {}, {"out": None}, f"out = {node.state}")
        offset = symbolic.symstr(node.begin, cpp_mode=False)
        record.add_edge(
            write, "out", record.add_write(OUT_CONNECTOR), None, Memlet(f"{OUT_CONNECTOR}[{node.iterate} - ({offset})]")
        )
        body.append(record)
    blocks = list(step.nodes())
    edges = list(step.edges())
    for block in blocks:
        loop.add_node(block, is_start_block=not body and block is step.start_block)
    for edge in edges:
        loop.add_edge(edge.src, edge.dst, edge.data)
    if body:
        loop.add_node(body[0], is_start_block=True)
        loop.add_edge(body[0], step.start_block, dace.InterstateEdge())
    entry = dace.SDFGState(f"{label}_entry", sdfg=inner)
    leave = dace.SDFGState(f"{label}_leave", sdfg=inner)
    if not record_states:
        keep = leave.add_tasklet("keep", {}, {"out": None}, f"out = {node.state}")
        leave.add_edge(keep, "out", leave.add_write("table"), None, Memlet("table[__c, __j]"))
    chain(inner, [entry, loop, leave])
    seed = "starts[__c]" if start == "starts" else f"({symbolic.symstr(node.base, cpp_mode=False)}) + __j"
    inner.out_edges(entry)[0].data.assignments[node.state] = seed
    inner.reset_cfg_list()
    return inner


def compose_sdfg(node: "StateScan", outer: SDFG) -> SDFG:
    """One thread: ``starts`` from ``_init`` and the block tables, and the final state into ``_out``."""
    inner = SDFG("compose")
    for name in ("table", "starts", INIT_CONNECTOR, OUT_CONNECTOR):
        inner.add_datadesc(name, argument(outer, name))
    if "__c" not in inner.symbols:
        inner.add_symbol("__c", dtypes.int64)
    base = symbolic.symstr(node.base, cpp_mode=False)
    chunks = symbolic.symstr(node.chunks, cpp_mode=False)
    length = symbolic.symstr(node.end - node.begin, cpp_mode=False)
    seed = dace.SDFGState("seed", sdfg=inner)
    copy_in = seed.add_tasklet("seed", {"init": None}, {"first": None}, "first = init")
    seed.add_edge(seed.add_read(INIT_CONNECTOR), None, copy_in, "init", Memlet(f"{INIT_CONNECTOR}[0]"))
    seed.add_edge(copy_in, "first", seed.add_write("starts"), None, Memlet("starts[0]"))
    loop = counted_loop("compose_blocks", "__c", "0", chunks)
    hop = loop.add_state("hop", is_start_block=True)
    look = hop.add_tasklet("look", {"tbl": None, "cur": None}, {"nxt": None}, f"nxt = tbl[__c, cur - ({base})]")
    hop.add_edge(hop.add_read("table"), None, look, "tbl", Memlet.from_array("table", inner.arrays["table"]))
    hop.add_edge(hop.add_read("starts"), None, look, "cur", Memlet("starts[__c]"))
    hop.add_edge(look, "nxt", hop.add_write("starts"), None, Memlet("starts[__c + 1]"))
    final = dace.SDFGState("final", sdfg=inner)
    last = final.add_tasklet("last", {"cur": None}, {"out": None}, "out = cur")
    final.add_edge(final.add_read("starts"), None, last, "cur", Memlet(f"starts[{chunks}]"))
    final.add_edge(last, "out", final.add_write(OUT_CONNECTOR), None, Memlet(f"{OUT_CONNECTOR}[{length}]"))
    chain(inner, [seed, loop, final])
    inner.reset_cfg_list()
    return inner


def whole(sdfg: SDFG, name: str) -> Memlet:
    return Memlet.from_array(name, sdfg.arrays[name])


def nest(
    state: dace.SDFGState,
    sdfg: SDFG,
    inner: SDFG,
    reads: list[str],
    writes: list[str],
    ranges: dict[str, str],
    schedule: dtypes.ScheduleType,
) -> None:
    """``inner`` under a map over ``ranges``, reading and writing whole containers of ``sdfg``."""
    entry, leave = state.add_map(inner.name, ranges, schedule=schedule)
    if schedule in dtypes.ALL_GPU_SCHEDULES:
        # one thread per (block, start state); the compose map has a single point
        entry.map.gpu_block_size = [GPU_BLOCK_THREADS if len(ranges) > 1 or "__c" in ranges else 1, 1, 1]
    mapping = {name: name for name in inner.free_symbols}
    call = state.add_nested_sdfg(inner, dict.fromkeys(reads), dict.fromkeys(writes), symbol_mapping=mapping)
    for name in reads:
        entry.add_in_connector(f"IN_{name}")
        entry.add_out_connector(f"OUT_{name}")
        state.add_edge(state.add_read(name), None, entry, f"IN_{name}", whole(sdfg, name))
        state.add_edge(entry, f"OUT_{name}", call, name, whole(sdfg, name))
    if not reads:
        state.add_edge(entry, None, call, None, Memlet())
    for name in writes:
        leave.add_in_connector(f"IN_{name}")
        leave.add_out_connector(f"OUT_{name}")
        state.add_edge(call, name, leave, f"IN_{name}", whole(sdfg, name))
        state.add_edge(leave, f"OUT_{name}", state.add_write(name), None, whole(sdfg, name))


def scan_sdfg(node: "StateScan", state: dace.SDFGState, outer: SDFG, schedule: dtypes.ScheduleType) -> SDFG:
    """The three-phase expansion with every map at ``schedule``."""
    sdfg = SDFG(f"{node.label}_expansion")
    declare_operands(node, sdfg, outer, state)
    on_gpu = schedule in dtypes.ALL_GPU_SCHEDULES
    storage = dtypes.StorageType.GPU_Global if on_gpu else dtypes.StorageType.CPU_Heap
    dtype = sdfg.arrays[OUT_CONNECTOR].dtype
    sdfg.add_array("table", [node.chunks, node.count], dtype, storage=storage, transient=True)
    sdfg.add_array("starts", [node.chunks + 1], dtype, storage=storage, transient=True)
    tables = sorted(edge.dst_conn for edge in state.in_edges(node) if edge.dst_conn != INIT_CONNECTOR)
    chunks = symbolic.symstr(node.chunks, cpp_mode=False)
    count = symbolic.symstr(node.count, cpp_mode=False)
    single = schedule if on_gpu else dtypes.ScheduleType.Sequential
    phases = [
        sdfg.add_state("block_tables", is_start_block=True),
        sdfg.add_state("compose"),
        sdfg.add_state("block_states"),
    ]
    sdfg.add_edge(phases[0], phases[1], dace.InterstateEdge())
    sdfg.add_edge(phases[1], phases[2], dace.InterstateEdge())
    nest(
        phases[0],
        sdfg,
        run_sdfg(node, "block_table", sdfg, "base", False),
        tables,
        ["table"],
        {"__c": f"0:{chunks}", "__j": f"0:{count}"},
        schedule,
    )
    nest(
        phases[1],
        sdfg,
        compose_sdfg(node, sdfg),
        ["table", INIT_CONNECTOR],
        ["starts", OUT_CONNECTOR],
        {"__unit": "0:1"},
        single,
    )
    nest(
        phases[2],
        sdfg,
        run_sdfg(node, "block_states", sdfg, "starts", True),
        [*tables, "starts"],
        [OUT_CONNECTOR],
        {"__c": f"0:{chunks}"},
        schedule,
    )
    return sdfg


@library.expansion
class ExpandCPU(ExpandTransformation):
    """The three phases with the block maps on ``CPU_Multicore``."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "StateScan", state: dace.SDFGState, sdfg: SDFG) -> SDFG:
        return scan_sdfg(node, state, sdfg, dtypes.ScheduleType.CPU_Multicore)


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """The three phases with the block maps on ``GPU_Device``; the GPU code generator lowers them for CUDA and
    HIP alike."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "StateScan", state: dace.SDFGState, sdfg: SDFG) -> SDFG:
        return scan_sdfg(node, state, sdfg, dtypes.ScheduleType.GPU_Device)


@library.expansion
class ExpandPure(ExpandTransformation):
    """The three phases with sequential maps, for consumers that read the SDFG."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "StateScan", state: dace.SDFGState, sdfg: SDFG) -> SDFG:
        return scan_sdfg(node, state, sdfg, dtypes.ScheduleType.Sequential)


@library.expansion
class ExpandAuto(ExpandTransformation):
    """Picks ``CPU``, ``CUDA`` or ``pure`` from the node's schedule."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "StateScan", state: dace.SDFGState, sdfg: SDFG):
        return schedule_dispatch(ExpandAuto, node, state)


@library.node
class StateScan(nodes.LibraryNode):
    """The states of the recurrence ``s[k+1] = step(s[k], k)`` over ``k`` in ``[begin, end)``.

    Inputs: one connector per array the step reads, named as the step names it, and ``_init``, the state before
    ``begin``. Output: ``_out``, ``end - begin + 1`` states.
    """

    step = SDFGReferenceProperty(
        allow_none=True,
        desc="The step: a region reading the symbols ``state`` and ``iterate`` and assigning ``state``.",
    )
    state = Property(dtype=str, default="", desc="The carried state symbol.")
    iterate = Property(dtype=str, default="", desc="The iteration symbol the step reads.")
    begin = SymbolicProperty(default=0, desc="The first iteration.")
    end = SymbolicProperty(default=0, desc="One past the last iteration.")
    base = SymbolicProperty(default=0, desc="The smallest state.")
    count = SymbolicProperty(default=1, desc="How many states there are.")
    chunks = SymbolicProperty(default=256, desc="How many blocks the iterations are cut into.")

    implementations = {"CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure, "Auto": ExpandAuto}
    default_implementation = "Auto"

    def __init__(self, name: str = "StateScan", tables: tuple[str, ...] = (), *args, **kwargs):
        super().__init__(
            name,
            *args,
            inputs=dict.fromkeys([*tables, INIT_CONNECTOR]),
            outputs=dict.fromkeys([OUT_CONNECTOR]),
            **kwargs,
        )
        self.specialization_hint = FINITE_STATE_SCAN_HINT

    def validate(self, sdfg: SDFG, state: dace.SDFGState):
        connected = {edge.dst_conn for edge in state.in_edges(self)}
        if INIT_CONNECTOR not in connected or not state.out_edges(self):
            raise ValueError(f"{self.label}: '{INIT_CONNECTOR}' and '{OUT_CONNECTOR}' must be connected")
        if self.step is None or not self.state or not self.iterate:
            raise ValueError(f"{self.label}: step, state and iterate must be set")
