# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``StateScan``: the sequence of states a loop-carried recurrence passes through, computed as a parallel scan.

The recurrence is ``s[k+1] = f(s[k], k)`` for ``k`` in ``[begin, end)``, where ``f`` is the ``step`` SDFG: a region
that reads the symbol ``state`` and the iteration symbol ``iterate`` and assigns ``state``. :class:`Scan` carries a
value through a fixed operator; this node carries a STATE through arbitrary code. That composes cheaply in two
cases, the node's ``kind``:

- ``FINITE``: the state takes ``count`` values starting at ``base``
  (:mod:`~dace.transformation.passes.analysis.finite_state`). A block of iterations is the table of its end state
  per start state, and two tables compose by lookup.
- ``GF2_AFFINE``: ``f(s, k) = A s (+) c_k`` over GF(2) with ``A`` the same at every ``k``
  (:mod:`~dace.transformation.passes.analysis.carried_state`). A block of ``l`` iterations is
  ``s -> A^l s (+) c_block``; ``A`` is read off the step at ``begin`` and ``c_block`` is the block run from zero.

Output ``_out[k - begin]`` is the state before iteration ``k``, and ``_out[end - begin]`` the state after the last.

Every expansion is one SDFG, scheduled for the device: ``chunks`` blocks summarize themselves in parallel, one
thread composes the summaries from ``_init`` into each block's start state, and the blocks run again from those
states recording every state. A finite scan does ``(count + 1) * N`` steps against ``N`` for the loop; a GF(2) scan
``2 * N + bits + 1`` steps and ``bits^2 log N`` word operations.
"""

import copy
import enum

import dace
from dace import SDFG, dtypes, library, symbolic
from dace.libraries.standard.helper import schedule_dispatch
from dace.libraries.standard.pure_components import chain, counted_loop
from dace.memlet import Memlet
from dace.properties import EnumProperty, Property, SDFGReferenceProperty, SymbolicProperty
from dace.sdfg import nodes
from dace.transformation.transformation import ExpandTransformation

INIT_CONNECTOR = "_init"
OUT_CONNECTOR = "_out"
#: Threads per GPU thread block for the block maps.
GPU_BLOCK_THREADS = 128

#: Recorded on every node: the device trade of the scan against the loop it replaces.
STATE_SCAN_HINT = (
    "state scan; canonicalization takes the parallel form.\n"
    "Alternative: the sequential loop over the state.\n"
    "CPU: the loop is usually the better of the two for a finite scan unless the cores outnumber the states -- "
    "every block runs once per state; a GF(2) scan does twice the loop's work.\n"
    "GPU: the scan is usually the better of the two.\n"
    "Both are correct. Measure before choosing."
)


class StateScanKind(enum.Enum):
    """How a :class:`StateScan` summarizes a block of iterations."""

    #: A table of ``count`` end states, one per start state.
    FINITE = "finite"
    #: The constant of ``s -> A^l s (+) c``; ``A`` is shared by every block.
    GF2_AFFINE = "gf2_affine"


def text(expr) -> str:
    return symbolic.symstr(expr, cpp_mode=False)


def chunk_bounds(node: "StateScan") -> tuple[str, str]:
    """The first and one-past-last iteration of block ``__c``."""
    size = f"int_ceil({text(node.end - node.begin)}, {text(node.chunks)})"
    return f"{text(node.begin)} + __c * {size}", f"min({text(node.begin)} + (__c + 1) * {size}, {text(node.end)})"


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


def run_sdfg(
    node: "StateScan", label: str, outer: SDFG, seed: str, bounds: tuple[str, str], sink: str | None, index: str
) -> SDFG:
    """A sequential run of the step from ``seed`` over ``bounds``. It writes its end state to ``sink[index]``, or,
    with no ``sink``, records every state into ``_out``. ``__c`` and ``__j`` are the map's symbols."""
    inner = SDFG(label)
    step = copy.deepcopy(node.step)
    for name, desc in step.arrays.items():
        inner.add_datadesc(name, argument(outer, name) if name in node.in_connectors else copy.deepcopy(desc))
    used = f"{seed} {bounds[0]} {bounds[1]} {index}"
    for name, dtype in [*step.symbols.items(), ("__c", dtypes.int64), ("__j", dtypes.int64)]:
        if name not in inner.symbols and (name in step.symbols or name in used):
            inner.add_symbol(name, dtype)
    for name in [name for name in outer.arrays if name in seed]:
        inner.add_datadesc(name, argument(outer, name))
    target = sink if sink is not None else OUT_CONNECTOR
    inner.add_datadesc(target, argument(outer, target))
    loop = counted_loop(f"{label}_iterations", node.iterate, bounds[0], bounds[1])
    for block in list(step.nodes()):
        loop.add_node(block, is_start_block=sink is not None and block is step.start_block)
    for edge in step.edges():
        loop.add_edge(edge.src, edge.dst, edge.data)
    if sink is None:
        record = dace.SDFGState(f"{label}_record", sdfg=inner)
        write = record.add_tasklet("record", {}, {"out": None}, f"out = {node.state}")
        record.add_edge(
            write,
            "out",
            record.add_write(OUT_CONNECTOR),
            None,
            Memlet(f"{OUT_CONNECTOR}[{node.iterate} - ({text(node.begin)})]"),
        )
        loop.add_node(record, is_start_block=True)
        loop.add_edge(record, step.start_block, dace.InterstateEdge())
    entry = dace.SDFGState(f"{label}_entry", sdfg=inner)
    leave = dace.SDFGState(f"{label}_leave", sdfg=inner)
    if sink is not None:
        keep = leave.add_tasklet("keep", {}, {"out": None}, f"out = {node.state}")
        leave.add_edge(keep, "out", leave.add_write(sink), None, Memlet(f"{sink}[{index}]"))
    chain(inner, [entry, loop, leave])
    inner.out_edges(entry)[0].data.assignments[node.state] = seed
    inner.reset_cfg_list()
    return inner


def finite_compose_sdfg(node: "StateScan", outer: SDFG) -> SDFG:
    """One thread: ``starts`` from ``_init`` and the block tables, and the final state into ``_out``."""
    inner = SDFG("compose")
    for name in ("table", "starts", INIT_CONNECTOR, OUT_CONNECTOR):
        inner.add_datadesc(name, argument(outer, name))
    if "__c" not in inner.symbols:
        inner.add_symbol("__c", dtypes.int64)
    seed = dace.SDFGState("seed", sdfg=inner)
    copy_in = seed.add_tasklet("seed", {"init": None}, {"first": None}, "first = init")
    seed.add_edge(seed.add_read(INIT_CONNECTOR), None, copy_in, "init", Memlet(f"{INIT_CONNECTOR}[0]"))
    seed.add_edge(copy_in, "first", seed.add_write("starts"), None, Memlet("starts[0]"))
    loop = counted_loop("compose_blocks", "__c", "0", text(node.chunks))
    hop = loop.add_state("hop", is_start_block=True)
    look = hop.add_tasklet(
        "look",
        {"tbl": None, "cur": None},
        {"nxt": None},
        f"nxt = tbl[__c * ({text(node.count)}) + cur - ({text(node.base)})]",
    )
    hop.add_edge(hop.add_read("table"), None, look, "tbl", whole(inner, "table"))
    hop.add_edge(hop.add_read("starts"), None, look, "cur", Memlet("starts[__c]"))
    hop.add_edge(look, "nxt", hop.add_write("starts"), None, Memlet("starts[__c + 1]"))
    final = dace.SDFGState("final", sdfg=inner)
    last = final.add_tasklet("last", {"cur": None}, {"out": None}, "out = cur")
    final.add_edge(final.add_read("starts"), None, last, "cur", Memlet(f"starts[{text(node.chunks)}]"))
    final.add_edge(
        last, "out", final.add_write(OUT_CONNECTOR), None, Memlet(f"{OUT_CONNECTOR}[{text(node.end - node.begin)}]")
    )
    chain(inner, [seed, loop, final])
    inner.reset_cfg_list()
    return inner


#: The GF(2) compose. ``basis_images[j]`` is the step at ``begin`` from basis state ``j`` (from zero at ``j == bits``), so
#: column ``j`` of ``A`` is ``basis_images[j] ^ basis_images[bits]``. Every block but the last is ``size`` long; ``A^size`` and the
#: last block's power come from repeated squaring on column masks.
GF2_COMPOSE = """
unsigned long long column[64], full[64], last[64], base[64], next[64];
const long long bits = {bits}, chunks = {chunks}, begin = {begin}, end = {end};
const long long size = (end - begin + chunks - 1) / chunks;
for (long long j = 0; j < bits; ++j) {{
    column[j] = (unsigned long long)basis_images[j] ^ (unsigned long long)basis_images[bits];
}}
for (long long which = 0; which < 2; ++which) {{
    unsigned long long *power = which == 0 ? full : last;
    long long exponent = which == 0 ? size : (end - begin) - size * (chunks - 1);
    if (exponent < 0) exponent = 0;
    for (long long j = 0; j < bits; ++j) {{ power[j] = 1ULL << j; base[j] = column[j]; }}
    while (exponent > 0) {{
        if (exponent & 1) {{
            for (long long j = 0; j < bits; ++j) {{
                unsigned long long x = power[j], y = 0;
                for (long long b = 0; b < bits; ++b) if ((x >> b) & 1ULL) y ^= base[b];
                next[j] = y;
            }}
            for (long long j = 0; j < bits; ++j) power[j] = next[j];
        }}
        for (long long j = 0; j < bits; ++j) {{
            unsigned long long x = base[j], y = 0;
            for (long long b = 0; b < bits; ++b) if ((x >> b) & 1ULL) y ^= base[b];
            next[j] = y;
        }}
        for (long long j = 0; j < bits; ++j) base[j] = next[j];
        exponent >>= 1;
    }}
}}
block_starts[0] = seed_state;
for (long long c = 0; c < chunks; ++c) {{
    long long lo = begin + c * size, hi = begin + (c + 1) * size;
    if (lo > end) lo = end;
    if (hi > end) hi = end;
    unsigned long long x = (unsigned long long)block_starts[c], y = 0;
    if (hi - lo == size) {{
        for (long long b = 0; b < bits; ++b) if ((x >> b) & 1ULL) y ^= full[b];
    }} else if (hi > lo) {{
        for (long long b = 0; b < bits; ++b) if ((x >> b) & 1ULL) y ^= last[b];
    }} else {{
        y = x;
    }}
    block_starts[c + 1] = ({ctype})(y ^ (unsigned long long)block_constants[c]);
}}
states[end - begin] = block_starts[chunks];
"""


def gf2_compose_sdfg(node: "StateScan", outer: SDFG) -> SDFG:
    """One thread: ``A`` from the basis runs, its block powers, ``starts`` and the final state into ``_out``."""
    inner = SDFG("compose")
    for name in ("raw", "block", "starts", INIT_CONNECTOR, OUT_CONNECTOR):
        inner.add_datadesc(name, argument(outer, name))
    state = inner.add_state("compose")
    code = GF2_COMPOSE.format(
        bits=node.bits,
        chunks=symbolic.symstr(node.chunks, cpp_mode=True),
        begin=symbolic.symstr(node.begin, cpp_mode=True),
        end=symbolic.symstr(node.end, cpp_mode=True),
        ctype=inner.arrays["starts"].dtype.ctype,
    )
    tasklet = state.add_tasklet(
        "compose",
        {"basis_images": None, "block_constants": None, "seed_state": None},
        {"block_starts": None, "states": None},
        code,
        language=dtypes.Language.CPP,
    )
    state.add_edge(state.add_read("raw"), None, tasklet, "basis_images", whole(inner, "raw"))
    state.add_edge(state.add_read("block"), None, tasklet, "block_constants", whole(inner, "block"))
    state.add_edge(state.add_read(INIT_CONNECTOR), None, tasklet, "seed_state", Memlet(f"{INIT_CONNECTOR}[0]"))
    state.add_edge(tasklet, "block_starts", state.add_write("starts"), None, whole(inner, "starts"))
    state.add_edge(tasklet, "states", state.add_write(OUT_CONNECTOR), None, whole(inner, OUT_CONNECTOR))
    return inner


def basis_sdfg(node: "StateScan", outer: SDFG) -> SDFG:
    """``seeds[j]``: the basis state ``1 << j`` for ``j < bits``, and zero at ``j == bits``."""
    inner = SDFG("basis")
    inner.add_datadesc("seeds", argument(outer, "seeds"))
    state = inner.add_state("basis")
    code = (
        f"for (long long j = 0; j <= {node.bits}; ++j) "
        f"basis_states[j] = j < {node.bits} ? ({inner.arrays['seeds'].dtype.ctype})(1ULL << j) : 0;"
    )
    tasklet = state.add_tasklet("basis", {}, {"basis_states": None}, code, language=dtypes.Language.CPP)
    state.add_edge(tasklet, "basis_states", state.add_write("seeds"), None, whole(inner, "seeds"))
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
        # one thread per map point; a single-point map gets one thread
        entry.map.gpu_block_size = [GPU_BLOCK_THREADS if ranges != {"__unit": "0:1"} else 1, 1, 1]
    mapping = {name: name for name in inner.free_symbols if name not in inner.arrays}
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


def phases(sdfg: SDFG, count: int) -> list[dace.SDFGState]:
    states = [sdfg.add_state(f"phase_{index}", is_start_block=index == 0) for index in range(count)]
    for first, second in zip(states, states[1:]):
        sdfg.add_edge(first, second, dace.InterstateEdge())
    return states


def scan_sdfg(node: "StateScan", state: dace.SDFGState, outer: SDFG, schedule: dtypes.ScheduleType) -> SDFG:
    """The expansion with every map at ``schedule``."""
    sdfg = SDFG(f"{node.label}_expansion")
    declare_operands(node, sdfg, outer, state)
    on_gpu = schedule in dtypes.ALL_GPU_SCHEDULES
    storage = dtypes.StorageType.GPU_Global if on_gpu else dtypes.StorageType.CPU_Heap
    single = schedule if on_gpu else dtypes.ScheduleType.Sequential
    dtype = sdfg.arrays[OUT_CONNECTOR].dtype
    sdfg.add_array("starts", [node.chunks + 1], dtype, storage=storage, transient=True)
    tables = sorted(edge.dst_conn for edge in state.in_edges(node) if edge.dst_conn != INIT_CONNECTOR)
    chunks = text(node.chunks)
    blocks = chunk_bounds(node)
    if node.kind is StateScanKind.FINITE:
        # flat, so no dimension of extent one is squeezed out of it
        sdfg.add_array("table", [node.chunks * node.count], dtype, storage=storage, transient=True)
        first, second, third = phases(sdfg, 3)
        nest(
            first,
            sdfg,
            run_sdfg(
                node,
                "block_table",
                sdfg,
                f"({text(node.base)}) + __j",
                blocks,
                "table",
                f"__c * ({text(node.count)}) + __j",
            ),
            tables,
            ["table"],
            {"__c": f"0:{chunks}", "__j": f"0:{text(node.count)}"},
            schedule,
        )
        nest(
            second,
            sdfg,
            finite_compose_sdfg(node, sdfg),
            ["table", INIT_CONNECTOR],
            ["starts", OUT_CONNECTOR],
            {"__unit": "0:1"},
            single,
        )
    else:
        sdfg.add_array("seeds", [node.bits + 1], dtype, storage=storage, transient=True)
        sdfg.add_array("raw", [node.bits + 1], dtype, storage=storage, transient=True)
        sdfg.add_array("block", [node.chunks], dtype, storage=storage, transient=True)
        basis, columns, summaries, second, third = phases(sdfg, 5)
        nest(basis, sdfg, basis_sdfg(node, sdfg), [], ["seeds"], {"__unit": "0:1"}, single)
        once = (text(node.begin), f"min({text(node.begin)} + 1, {text(node.end)})")
        nest(
            columns,
            sdfg,
            run_sdfg(node, "basis_step", sdfg, "seeds[__j]", once, "raw", "__j"),
            [*tables, "seeds"],
            ["raw"],
            {"__j": f"0:{node.bits + 1}"},
            schedule,
        )
        nest(
            summaries,
            sdfg,
            run_sdfg(node, "block_constant", sdfg, "0", blocks, "block", "__c"),
            tables,
            ["block"],
            {"__c": f"0:{chunks}"},
            schedule,
        )
        nest(
            second,
            sdfg,
            gf2_compose_sdfg(node, sdfg),
            ["raw", "block", INIT_CONNECTOR],
            ["starts", OUT_CONNECTOR],
            {"__unit": "0:1"},
            single,
        )
    nest(
        third,
        sdfg,
        run_sdfg(node, "block_states", sdfg, "starts[__c]", blocks, None, ""),
        [*tables, "starts"],
        [OUT_CONNECTOR],
        {"__c": f"0:{chunks}"},
        schedule,
    )
    return sdfg


@library.expansion
class ExpandCPU(ExpandTransformation):
    """The block maps on ``CPU_Multicore``."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "StateScan", state: dace.SDFGState, sdfg: SDFG) -> SDFG:
        return scan_sdfg(node, state, sdfg, dtypes.ScheduleType.CPU_Multicore)


@library.expansion
class ExpandCUDA(ExpandTransformation):
    """The block maps on ``GPU_Device``; the GPU code generator lowers them for CUDA and HIP alike."""

    environments: list[type] = []

    @staticmethod
    def expansion(node: "StateScan", state: dace.SDFGState, sdfg: SDFG) -> SDFG:
        return scan_sdfg(node, state, sdfg, dtypes.ScheduleType.GPU_Device)


@library.expansion
class ExpandPure(ExpandTransformation):
    """Sequential maps, for consumers that read the SDFG."""

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

    Inputs: one connector per container the step reads, named as the step names it, and ``_init``, the state
    before ``begin``. Output: ``_out``, ``end - begin + 1`` states.
    """

    kind = EnumProperty(dtype=StateScanKind, default=StateScanKind.FINITE, desc="How a block is summarized.")
    step = SDFGReferenceProperty(
        allow_none=True,
        desc="The step: a region reading the symbols ``state`` and ``iterate`` and assigning ``state``.",
    )
    state = Property(dtype=str, default="", desc="The carried state symbol.")
    iterate = Property(dtype=str, default="", desc="The iteration symbol the step reads.")
    begin = SymbolicProperty(default=0, desc="The first iteration.")
    end = SymbolicProperty(default=0, desc="One past the last iteration.")
    base = SymbolicProperty(default=0, desc="FINITE: the smallest state.")
    count = SymbolicProperty(default=1, desc="FINITE: how many states there are.")
    bits = Property(dtype=int, default=64, desc="GF2_AFFINE: the width of the state, at most 64.")
    chunks = SymbolicProperty(default=256, desc="How many blocks the iterations are cut into.")

    implementations = {"CPU": ExpandCPU, "CUDA": ExpandCUDA, "pure": ExpandPure, "Auto": ExpandAuto}
    default_implementation = "Auto"

    def __init__(self, name: str = "StateScan", tables: tuple[str, ...] = (), *args, **kwargs):
        super().__init__(
            name, *args, inputs=dict.fromkeys([*tables, INIT_CONNECTOR]), outputs={OUT_CONNECTOR: None}, **kwargs
        )
        self.specialization_hint = STATE_SCAN_HINT

    @property
    def free_symbols(self) -> set[str]:
        """The node's own symbols and the ones its step reads from outside: a pass that drops symbols nothing uses
        must see the step's, which live in a property it does not walk."""
        own = super().free_symbols
        if self.step is None:
            return own
        return own | (self.step.free_symbols - set(self.step.arrays) - {self.state, self.iterate})

    def validate(self, sdfg: SDFG, state: dace.SDFGState):
        connected = {edge.dst_conn for edge in state.in_edges(self)}
        if INIT_CONNECTOR not in connected or not state.out_edges(self):
            raise ValueError(f"{self.label}: '{INIT_CONNECTOR}' and '{OUT_CONNECTOR}' must be connected")
        if self.step is None or not self.state or not self.iterate:
            raise ValueError(f"{self.label}: step, state and iterate must be set")
        if self.kind is StateScanKind.GF2_AFFINE and not 0 < self.bits <= 64:
            raise ValueError(f"{self.label}: a GF(2) state is 1 to 64 bits wide, not {self.bits}")
