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
from dace.libraries.standard.pure_components import chain, counted_loop, tasklet_state
from dace.memlet import Memlet
from dace.properties import CodeBlock, EnumProperty, Property, SDFGReferenceProperty, SymbolicProperty
from dace.sdfg import nodes
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion
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
        bound = name == node.iterate
        if not bound and name not in inner.symbols and (name in step.symbols or name in used):
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


#: The uint64 scratch of the GF(2) nests: ``unit[j] = 1 << j``, the step's columns, the block powers, a product
#: staging buffer and one accumulator word. The expansion declares it with the device's storage, as it does
#: ``starts``: 64 words is too large for a kernel's registers.
GF2_SCRATCH = ("unit", "column", "full", "last", "base", "product", "word")


def unit_blocks(inner: SDFG, bits: int) -> list[ControlFlowBlock]:
    """``unit[j] = 1 << j`` by doubling, so no shift overflows a narrower literal."""
    one = tasklet_state(inner, "unit_one", "u = 1", {}, {"u": Memlet("unit[0]")})
    double = counted_loop("unit_doubling", "__j", "1", str(bits))
    double.add_node(
        tasklet_state(inner, "unit_double", "u = v + v", {"v": Memlet("unit[__j - 1]")}, {"u": Memlet("unit[__j]")}),
        is_start_block=True,
    )
    return [one, double]


def word_loop(inner: SDFG, label: str, bits: int, code: str, reads: dict[str, Memlet], writes: dict[str, Memlet]):
    """``code`` once per ``__j`` in ``[0, bits)``."""
    loop = counted_loop(label, "__j", "0", str(bits))
    loop.add_node(tasklet_state(inner, label, code, reads, writes), is_start_block=True)
    return loop


def times(inner: SDFG, label: str, bits: int, operand: str) -> list[ControlFlowBlock]:
    """``operand := base * operand`` over GF(2): column ``j`` becomes the XOR of the ``base`` columns at its set
    bits, staged through ``product``."""
    rows = counted_loop(f"{label}_columns", "__j", "0", str(bits))
    zero = tasklet_state(inner, f"{label}_zero", "w = 0", {}, {"w": Memlet("word[0]")})
    fold = counted_loop(f"{label}_bits", "__b", "0", str(bits))
    fold.add_node(
        tasklet_state(
            inner,
            f"{label}_fold",
            "w = acc ^ (col if ((x >> __b) & 1) == 1 else 0)",
            {"acc": Memlet("word[0]"), "col": Memlet("base[__b]"), "x": Memlet(f"{operand}[__j]")},
            {"w": Memlet("word[0]")},
        ),
        is_start_block=True,
    )
    store = tasklet_state(inner, f"{label}_store", "y = w", {"w": Memlet("word[0]")}, {"y": Memlet("product[__j]")})
    chain(rows, [zero, fold, store])
    back = word_loop(
        inner, f"{label}_back", bits, "y = x", {"x": Memlet("product[__j]")}, {"y": Memlet(f"{operand}[__j]")}
    )
    return [rows, back]


def power_blocks(inner: SDFG, bits: int, power: str, exponent: str) -> list[ControlFlowBlock]:
    """``power := A^exponent`` by repeated squaring of ``base``, with ``A``'s columns in ``column``."""
    start = word_loop(
        inner,
        f"{power}_start",
        bits,
        "p = u\nb = c",
        {"u": Memlet("unit[__j]"), "c": Memlet("column[__j]")},
        {"p": Memlet(f"{power}[__j]"), "b": Memlet("base[__j]")},
    )
    variable = f"__e_{power}"
    squaring = LoopRegion(
        f"{power}_squaring", f"{variable} > 0", variable, f"{variable} = {exponent}", f"{variable} = {variable} // 2"
    )
    odd = ConditionalBlock(f"{power}_odd")
    multiply = ControlFlowRegion(f"{power}_multiply", sdfg=inner)
    chain(multiply, times(inner, f"{power}_times", bits, power))
    odd.add_branch(CodeBlock(f"{variable} % 2 == 1"), multiply)
    chain(squaring, [odd, *times(inner, f"{power}_square", bits, "base")])
    return [start, squaring]


def gf2_compose_sdfg(node: "StateScan", outer: SDFG) -> SDFG:
    """One thread: ``A`` from the basis runs, its block powers, ``starts`` and the final state into ``_out``.
    ``basis_images[j]`` (``raw``) is the step at ``begin`` from basis state ``j`` (from zero at ``j == bits``), so
    column ``j`` of ``A`` is ``raw[j] ^ raw[bits]``. Every non-empty block is ``__size`` long except at most one,
    which holds the remainder; ``A^__size`` and that block's power come from repeated squaring."""
    inner = SDFG("compose")
    for name in ("raw", "block", "starts", INIT_CONNECTOR, OUT_CONNECTOR):
        inner.add_datadesc(name, argument(outer, name))
    bits = node.bits
    for name in GF2_SCRATCH:
        inner.add_datadesc(name, argument(outer, name))
    for name in ("__size", "__length"):
        inner.add_symbol(name, dtypes.int64)
    span = text(node.end - node.begin)
    columns = word_loop(
        inner,
        "columns",
        bits,
        "c = r ^ z",
        {"r": Memlet("raw[__j]"), "z": Memlet(f"raw[{bits}]")},
        {"c": Memlet("column[__j]")},
    )
    seed = tasklet_state(
        inner, "seed", "first = init", {"init": Memlet(f"{INIT_CONNECTOR}[0]")}, {"first": Memlet("starts[0]")}
    )
    hops = counted_loop("compose_blocks", "__c", "0", text(node.chunks))
    zero = tasklet_state(inner, "hop_zero", "w = 0", {}, {"w": Memlet("word[0]")})
    fold = counted_loop("hop_bits", "__b", "0", str(bits))
    fold.add_node(
        tasklet_state(
            inner,
            "hop_fold",
            "w = acc ^ ((f if __length == __size else (l if __length > 0 else u)) if ((x >> __b) & 1) == 1 else 0)",
            {
                "acc": Memlet("word[0]"),
                "f": Memlet("full[__b]"),
                "l": Memlet("last[__b]"),
                "u": Memlet("unit[__b]"),
                "x": Memlet("starts[__c]"),
            },
            {"w": Memlet("word[0]")},
        ),
        is_start_block=True,
    )
    store = tasklet_state(
        inner,
        "hop_store",
        "s = w ^ k",
        {"w": Memlet("word[0]"), "k": Memlet("block[__c]")},
        {"s": Memlet("starts[__c + 1]")},
    )
    chain(hops, [zero, fold, store])
    low, high = chunk_bounds(node)
    hops.out_edges(zero)[0].data.assignments["__length"] = f"({high}) - min({low}, {text(node.end)})"
    final = tasklet_state(
        inner,
        "final",
        "out = cur",
        {"cur": Memlet(f"starts[{text(node.chunks)}]")},
        {"out": Memlet(f"{OUT_CONNECTOR}[{span}]")},
    )
    blocks = [
        *unit_blocks(inner, bits),
        columns,
        *power_blocks(inner, bits, "full", "__size"),
        *power_blocks(inner, bits, "last", f"({span}) % max(__size, 1)"),
        seed,
        hops,
        final,
    ]
    chain(inner, blocks)
    inner.out_edges(blocks[0])[0].data.assignments["__size"] = f"int_ceil({span}, {text(node.chunks)})"
    inner.reset_cfg_list()
    return inner


def basis_sdfg(node: "StateScan", outer: SDFG) -> SDFG:
    """``seeds[j]``: the basis state ``1 << j`` for ``j < bits``, and zero at ``j == bits``."""
    inner = SDFG("basis")
    inner.add_datadesc("seeds", argument(outer, "seeds"))
    bits = node.bits
    inner.add_datadesc("unit", argument(outer, "unit"))
    copy_out = word_loop(inner, "basis_copy", bits, "s = u", {"u": Memlet("unit[__j]")}, {"s": Memlet("seeds[__j]")})
    clear = tasklet_state(inner, "basis_zero", "s = 0", {}, {"s": Memlet(f"seeds[{bits}]")})
    chain(inner, [*unit_blocks(inner, bits), copy_out, clear])
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
    # flat and one element longer than needed: a container of one element reaches a tasklet as a scalar, not an array
    sdfg.add_array("starts", [node.chunks + 1], dtype, storage=storage, transient=True)
    tables = sorted(edge.dst_conn for edge in state.in_edges(node) if edge.dst_conn != INIT_CONNECTOR)
    chunks = text(node.chunks)
    blocks = chunk_bounds(node)
    if node.kind is StateScanKind.FINITE:
        sdfg.add_array("table", [node.chunks * node.count + 1], dtype, storage=storage, transient=True)
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
        sdfg.add_array("block", [node.chunks + 1], dtype, storage=storage, transient=True)
        for name in GF2_SCRATCH:
            words = 1 if name == "word" else node.bits
            sdfg.add_array(name, [words], dtypes.uint64, storage=storage, transient=True)
        basis, columns, summaries, second, third = phases(sdfg, 5)
        nest(basis, sdfg, basis_sdfg(node, sdfg), [], ["seeds", "unit"], {"__unit": "0:1"}, single)
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
            ["starts", OUT_CONNECTOR, *GF2_SCRATCH],
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
