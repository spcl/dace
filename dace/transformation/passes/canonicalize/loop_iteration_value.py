# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""What one iteration of a normalized loop nest computes, as a closed-form expression.

The semantic lifts that recognize a BLAS operator in a hand-written nest (``LoopToSymm``, ``LoopToRankKUpdate``)
compare what the nest COMPUTES against the operator's definition, not how it is spelled. The same kernel reaches
the ``lift_semantics`` phase in several shapes: a slice statement is lowered to a chain of elementwise loops that
pass values through vector temporaries, a scalar statement to one tasklet chain in one state, an inner product to
a ``Dot`` / ``MatMul`` node or to a reduction loop. :func:`summarize` evaluates all of them to one value.

Given an outer ``LoopRegion`` over ``i``, it returns, for every element of the one non-transient array ``C`` the
body writes, the value the body stores there as a sympy expression over the values at iteration start:

- an array read is an applied undefined function ``A(i, k)`` named after the container;
- an elementwise loop (every write of the queried array indexes it by the loop variable) contributes the body
  value of the one iteration that writes the element;
- an accumulation loop (the queried element does not depend on the loop variable, and one iteration turns its
  prior value ``P`` into ``P + T(k)``) contributes ``prior + Sum(T(k), (k, lo, hi))``;
- a vector-vector ``Dot`` / ``MatMul`` contributes ``Sum(a[t] * b[t], (t, 0, L - 1))``.

Elements are queried per write DOMAIN: every axis of a write is either fixed (an expression of ``i``) or a whole
loop/map range, and a range axis becomes a fresh query symbol bounded by that range. Anything not proven exactly
-- a loop-carried read, a write that may or may not hit the queried element, a WCR, a conditional, a nested SDFG
-- raises :class:`Refused`, which the lifts treat as "no match".
"""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import NamedTuple

import sympy
from sympy.core.function import AppliedUndef

from dace import SDFG, dtypes, subsets, symbolic
from dace import data as dt
from dace import memlet as mm
from dace.sdfg import nodes
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.sdfg import InterstateEdge
from dace.sdfg.state import AbstractControlFlowRegion, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation.passes.analysis import loop_analysis
from dace.transformation.passes.canonicalize.split_statements import value_edges

#: Evaluation steps one summary may take; a body that needs more is not a BLAS kernel.
MAX_STEPS = 20000


class Refused(Exception):
    """The nest's value cannot be derived exactly."""


def plain(value: object) -> sympy.Expr:
    """``value`` as a sympy expression whose symbols are plain ``sympy.Symbol`` keyed by name only.

    DaCe symbols carry a dtype in their identity, so a loop variable read from a memlet and the same name parsed from
    a loop bound would be two symbols; by name they are one.
    """
    expr = value if isinstance(value, sympy.Basic) else symbolic.pystr_to_symbolic(str(value))
    return expr.xreplace({s: sympy.Symbol(s.name) for s in expr.free_symbols if type(s) is not sympy.Symbol})


def same(a: sympy.Expr, b: sympy.Expr) -> bool:
    return sympy.expand(a - b) == 0


def nonnegative(expr: sympy.Expr) -> bool:
    """Whether ``expr`` is a number ``>= 0``; a symbolic difference is not proven either way."""
    expr = sympy.expand(expr)
    return bool(expr.is_Number and expr >= 0)


def positive(expr: sympy.Expr) -> bool:
    expr = sympy.expand(expr)
    return bool(expr.is_Number and expr > 0)


def leaf(array: str, index: Iterable[sympy.Expr]) -> sympy.Expr:
    """The read ``array[index]`` at iteration start."""
    return sympy.Function(array)(*index)


def leaves(expr: sympy.Basic) -> list[AppliedUndef]:
    """Every array read in ``expr``, in a stable order."""
    return sorted(expr.atoms(AppliedUndef), key=sympy.default_sort_key)


def read_array(read: AppliedUndef) -> str:
    return read.func.__name__


def canonical(expr: sympy.Expr) -> sympy.Expr:
    """``expr`` expanded, with every ``Sum`` over a canonical variable, split over its terms and with the factors
    that do not depend on its variable pulled out -- so two spellings of one value compare equal."""
    expr = sympy.expand(expr)
    if not expr.has(sympy.Sum):
        return expr
    return sympy.expand(expr.replace(lambda e: isinstance(e, sympy.Sum), canonical_sum))


def canonical_sum(total: sympy.Sum) -> sympy.Expr:
    if len(total.limits) != 1:
        raise Refused("multi-limit Sum")
    (var, lo, hi), body = total.limits[0], canonical(total.function)
    depth = 1 + max((int(s.limits[0][0].name[3:]) for s in body.atoms(sympy.Sum)), default=-1)
    k = sympy.Symbol(f"__s{depth}")
    body = sympy.expand(body.xreplace({var: k}))
    terms = []
    for term in sympy.Add.make_args(body):
        factors = sympy.Mul.make_args(term)
        outside = [f for f in factors if k not in f.free_symbols]
        inside = [f for f in factors if k in f.free_symbols]
        terms.append(sympy.Mul(*outside) * sympy.Sum(sympy.Mul(*inside), (k, lo, hi)))
    return sympy.Add(*terms)


class AxisKind(Enum):
    FIXED = auto()  #: an expression free of every range variable below the query level, or a one-value range
    RANGE = auto()  #: a whole range: a bare loop/map variable, or a copied slice
    OTHER = auto()  #: anything else (``k + 1``, ``2 * j``): never proven to hit or miss


@dataclass(frozen=True, slots=True)
class Axis:
    """One axis of a write, in the coordinates the query is asked in."""

    kind: AxisKind
    lo: sympy.Expr = sympy.S.Zero
    hi: sympy.Expr = sympy.S.Zero
    var: str | None = None


def classify(index: sympy.Expr, ranges: dict[str, tuple[sympy.Expr, sympy.Expr]]) -> Axis:
    if isinstance(index, sympy.Symbol) and index.name in ranges:
        lo, hi = ranges[index.name]
        return Axis(AxisKind.FIXED if same(lo, hi) else AxisKind.RANGE, lo, hi, index.name)
    if any(s.name in ranges for s in index.free_symbols):
        return Axis(AxisKind.OTHER)
    return Axis(AxisKind.FIXED, index, index)


def point_indices(subset: subsets.Subset | None) -> list[sympy.Expr]:
    """The element index of a single-element subset; :class:`Refused` for a range."""
    if subset is None:
        raise Refused("no subset")
    out = []
    for begin, end, _ in subset.ndrange():
        if not same(plain(begin), plain(end)):
            raise Refused("range where an element is expected")
        out.append(plain(begin))
    return out


def chain(region: ControlFlowRegion) -> list[ControlFlowBlock]:
    """The region's blocks in execution order; :class:`Refused` unless they form one unconditional chain without
    interstate assignments."""
    blocks: list[ControlFlowBlock] = []
    block = region.start_block
    while block is not None:
        blocks.append(block)
        outs = region.out_edges(block)
        if len(outs) > 1:
            raise Refused("branching body")
        if not outs:
            break
        edge: InterstateEdge = outs[0].data
        if edge.assignments or not edge.is_unconditional():
            raise Refused("interstate assignment or condition in the body")
        block = outs[0].dst
        if block in blocks:
            raise Refused("cycle in the body")
    if len(blocks) != len(region.nodes()):
        raise Refused("unreachable block in the body")
    return blocks


def loop_step(loop: LoopRegion) -> int:
    """``1`` or ``-1``, the step of a counted unit-stride loop; :class:`Refused` otherwise."""
    stride = loop_analysis.get_loop_stride(loop) if loop.loop_variable else None
    try:
        step = None if stride is None else symbolic.simplify(stride)
    except Exception:
        step = None
    if step not in (1, -1):
        raise Refused("not a unit-stride counted loop")
    return int(step)


def unit_stride(loop: LoopRegion) -> bool:
    stride = loop_analysis.get_loop_stride(loop)
    try:
        return stride is not None and symbolic.simplify(stride) == 1
    except Exception:
        return False


def loop_range(loop: LoopRegion, bind: dict[sympy.Symbol, sympy.Expr]) -> tuple[sympy.Expr, sympy.Expr]:
    """``(lo, hi)``: the smallest and largest iteration of a unit-stride loop in either direction, under ``bind``."""
    step = loop_step(loop)
    init, end = loop_analysis.get_init_assignment(loop), loop_analysis.get_loop_end(loop)
    if init is None or end is None:
        raise Refused("no loop bounds")
    first, last = plain(init).xreplace(bind), plain(end).xreplace(bind)
    return (first, last) if step == 1 else (last, first)


class Mode(Enum):
    TOP = auto()  #: the summarized iteration: a read at its start is a leaf
    ELEMENTWISE = auto()  #: the one iteration that writes the queried element
    ACCUMULATE = auto()  #: one iteration of an accumulation, its prior value a placeholder


@dataclass(slots=True)
class Frame:
    """One loop body being evaluated, and how a read that reaches its start is resolved."""

    seq: list[ControlFlowBlock]
    bind: dict[sympy.Symbol, sympy.Expr]
    mode: Mode
    loop: LoopRegion | None = None
    parent: "Frame | None" = None
    pos: int = 0  #: position of ``loop`` in ``parent.seq``
    target: tuple[str, tuple[sympy.Expr, ...]] | None = None
    prior: sympy.Symbol | None = None


Write = tuple[Axis, ...]


@dataclass(slots=True)
class Domain:
    """One write domain of the output and the value the iteration stores there."""

    axes: Write
    index: tuple[sympy.Expr, ...]  #: query index: the fixed expression or a query symbol per axis
    value: sympy.Expr


@dataclass(slots=True)
class Summary:
    """What one iteration of ``loop`` stores into its only non-transient output."""

    output: str
    var: sympy.Symbol  #: the outer loop variable
    extent: sympy.Expr  #: the outer trip count; the loop visits ``0 .. extent - 1``
    ascending: bool  #: the outer loop runs upward
    domains: list[Domain]
    written: dict[str, None]  #: every container the nest writes


@dataclass(slots=True)
class IterationEvaluator:
    """Backward evaluation of element values through one outer loop body."""

    facts: dict[sympy.Symbol, tuple[sympy.Expr, sympy.Expr]] = field(default_factory=dict)
    steps: int = 0
    fresh: int = 0

    def symbol(self, prefix: str, lo: sympy.Expr, hi: sympy.Expr) -> sympy.Symbol:
        """A fresh symbol bounded by ``[lo, hi]``."""
        sym = sympy.Symbol(f"__{prefix}{self.fresh}")
        self.fresh += 1
        self.facts[sym] = (lo, hi)
        return sym

    def axis_hit(self, axis: Axis, x: sympy.Expr) -> bool | None:
        """``True``: the write covers every value ``x`` can take; ``False``: none; ``None``: not proven."""
        if axis.kind is AxisKind.OTHER:
            return None
        if axis.kind is AxisKind.FIXED and same(x, axis.lo):
            return True
        lo, hi = self.facts.get(x, (x, x)) if isinstance(x, sympy.Symbol) else (x, x)
        if nonnegative(lo - axis.lo) and nonnegative(axis.hi - hi):
            return True
        if positive(axis.lo - hi) or positive(lo - axis.hi):
            return False
        return None

    def hit(self, write: Write, index: tuple[sympy.Expr, ...]) -> dict[str, sympy.Expr] | None:
        """The range-variable binding under which ``write`` stores ``index``, or ``None`` if it never does.

        :raises Refused: when neither is proven.
        """
        if len(write) != len(index):
            raise Refused("rank mismatch")
        binding: dict[str, sympy.Expr] = {}
        unknown = False
        for axis, x in zip(write, index):
            verdict = self.axis_hit(axis, x)
            if verdict is False:
                return None
            if verdict is None:
                unknown = True
            elif axis.var is not None:
                if axis.var in binding and not same(binding[axis.var], x):
                    unknown = True
                binding[axis.var] = x
        if unknown:
            raise Refused("write may or may not hit the queried element")
        return binding

    # Writes

    def writes(
        self,
        block: ControlFlowBlock,
        array: str,
        bind: dict[sympy.Symbol, sympy.Expr],
        ranges: dict[str, tuple[sympy.Expr, sympy.Expr]],
    ) -> list[Write]:
        """Every write of ``array`` inside ``block``, with the loops and maps below ``block`` as range variables."""
        if isinstance(block, SDFGState):
            return [
                self.state_write(block, edge, bind, ranges)
                for node in block.data_nodes()
                if node.data == array
                for edge in value_edges(block.in_edges(node))
            ]
        if isinstance(block, LoopRegion):
            inner = {**ranges, block.loop_variable: loop_range(block, bind)}
            return [w for b in block.nodes() for w in self.writes(b, array, bind, inner)]
        if not isinstance(block, AbstractControlFlowRegion) or array in written_arrays(block):
            raise Refused(f"{array} written under {type(block).__name__}")
        return []

    def state_write(
        self,
        state: SDFGState,
        edge: MultiConnectorEdge[mm.Memlet],
        bind: dict[sympy.Symbol, sympy.Expr],
        ranges: dict[str, tuple[sympy.Expr, sympy.Expr]],
    ) -> Write:
        """The axes of the write ``edge`` stores into its AccessNode."""
        if edge.data.wcr is not None:
            raise Refused("WCR write")
        src = edge.src
        if isinstance(src, (nodes.MapExit, nodes.Tasklet)):
            inner = state.memlet_path(edge)[0] if isinstance(src, nodes.MapExit) else edge
            if not isinstance(inner.src, nodes.Tasklet):
                raise Refused("map exit not fed by a tasklet")
            scoped = {**ranges, **self.map_ranges(state, inner.src, bind)}
            return tuple(classify(x.xreplace(bind), scoped) for x in point_indices(inner.data.subset))
        if isinstance(src, (nodes.AccessNode, nodes.LibraryNode)):
            if state.entry_node(src) is not None:
                raise Refused("copy inside a map")
            dst = edge.data.get_dst_subset(edge, state)
            if dst is None:
                raise Refused("copy without a destination subset")
            axes = []
            for begin, end, step in dst.ndrange():
                lo, hi = plain(begin).xreplace(bind), plain(end).xreplace(bind)
                if same(lo, hi):
                    axes.append(classify(lo, ranges))
                elif same(plain(step), 1) and not any(s.name in ranges for s in lo.free_symbols | hi.free_symbols):
                    axes.append(Axis(AxisKind.RANGE, lo, hi))
                else:
                    axes.append(Axis(AxisKind.OTHER))
            return tuple(axes)
        raise Refused(f"write by {type(src).__name__}")

    def map_ranges(
        self, state: SDFGState, node: nodes.Node, bind: dict[sympy.Symbol, sympy.Expr]
    ) -> dict[str, tuple[sympy.Expr, sympy.Expr]]:
        """The parameter ranges of every map around ``node``."""
        out: dict[str, tuple[sympy.Expr, sympy.Expr]] = {}
        entry = state.entry_node(node)
        while entry is not None:
            for param, (begin, end, step) in zip(entry.map.params, entry.map.range):
                if not same(plain(step), 1):
                    raise Refused("strided map")
                out[param] = (plain(begin).xreplace(bind), plain(end).xreplace(bind))
            entry = state.entry_node(entry)
        return out

    # Backward evaluation

    def before(self, frame: Frame, pos: int, array: str, index: tuple[sympy.Expr, ...]) -> sympy.Expr:
        """The value of ``array[index]`` just before ``frame.seq[pos]`` runs."""
        for p in range(pos - 1, -1, -1):
            value = self.effect(frame, p, array, index)
            if value is not None:
                return value
        return self.at_start(frame, array, index)

    def at_start(self, frame: Frame, array: str, index: tuple[sympy.Expr, ...]) -> sympy.Expr:
        """The value of ``array[index]`` at the start of ``frame``'s iteration."""
        if frame.mode is Mode.TOP:
            return leaf(array, index)
        loop, parent = frame.loop, frame.parent
        assert loop is not None and parent is not None
        if frame.mode is Mode.ACCUMULATE:
            assert frame.target is not None
            if array == frame.target[0] and all(same(a, b) for a, b in zip(index, frame.target[1])):
                assert frame.prior is not None
                return frame.prior
        var = sympy.Symbol(loop.loop_variable)
        for write in self.writes(loop, array, parent.bind, {}):
            binding = self.hit(write, index)
            if binding is None:
                continue
            # Elementwise: only the current iteration may store this element, and it has not yet.
            if frame.mode is Mode.ACCUMULATE or loop.loop_variable not in binding:
                raise Refused(f"loop-carried read of {array}{list(index)} in {loop.label}")
            if not same(binding[loop.loop_variable], frame.bind[var]):
                raise Refused(f"loop-carried read of {array}{list(index)} in {loop.label}")
        return self.before(parent, frame.pos, array, index)

    def step(self) -> None:
        self.steps += 1
        if self.steps > MAX_STEPS:
            raise Refused("evaluation budget exhausted")

    def effect(self, frame: Frame, pos: int, array: str, index: tuple[sympy.Expr, ...]) -> sympy.Expr | None:
        """The value ``frame.seq[pos]`` leaves in ``array[index]``, or ``None`` if it does not store it."""
        self.step()
        block = frame.seq[pos]
        if isinstance(block, SDFGState):
            return self.state_effect(frame, pos, block, array, index)
        if isinstance(block, LoopRegion):
            return self.loop_effect(frame, pos, block, array, index)
        self.writes(block, array, frame.bind, {})
        return None

    def state_effect(
        self, frame: Frame, pos: int, state: SDFGState, array: str, index: tuple[sympy.Expr, ...]
    ) -> sympy.Expr | None:
        hits = [
            node
            for node in state.data_nodes()
            if node.data == array
            for edge in value_edges(state.in_edges(node))
            if self.hit(self.state_write(state, edge, frame.bind, {}), index) is not None
        ]
        if not hits:
            return None
        if len(hits) > 1:
            raise Refused(f"{array} stored twice in one state")
        return StateValues(self, frame, pos, state).value_at(hits[0], index)

    def loop_effect(
        self, frame: Frame, pos: int, loop: LoopRegion, array: str, index: tuple[sympy.Expr, ...]
    ) -> sympy.Expr | None:
        bindings = [b for w in self.writes(loop, array, frame.bind, {}) if (b := self.hit(w, index)) is not None]
        if not bindings:
            return None
        seq = chain(loop)
        var = loop.loop_variable
        if all(var in b for b in bindings):
            first = bindings[0][var]
            if not all(same(b[var], first) for b in bindings):
                raise Refused("several iterations store the element")
            child = Frame(seq, {**frame.bind, sympy.Symbol(var): first}, Mode.ELEMENTWISE, loop, frame, pos)
            return self.before(child, len(seq), array, index)
        if any(var in b for b in bindings):
            raise Refused("element stored both per iteration and by every iteration")
        lo, hi = loop_range(loop, frame.bind)
        k = sympy.Symbol(var)
        self.facts[k] = (lo, hi)
        prior = self.symbol("prior", sympy.S.Zero, sympy.S.Zero)
        child = Frame(seq, frame.bind, Mode.ACCUMULATE, loop, frame, pos, (array, index), prior)
        value = self.before(child, len(seq), array, index)
        term = sympy.expand(value - prior)
        if term.has(prior) or not value.has(prior):
            raise Refused("not an accumulation")
        start = self.before(frame, pos, array, index)
        return start if term == 0 else start + sympy.Sum(term, (k, lo, hi))


@dataclass(slots=True)
class StateValues:
    """Dataflow evaluation inside one state; a read with no producer in the state asks the blocks before it."""

    evaluator: IterationEvaluator
    frame: Frame
    pos: int
    state: SDFGState

    def value_at(self, node: nodes.AccessNode, index: tuple[sympy.Expr, ...]) -> sympy.Expr:
        self.evaluator.step()
        producers = value_edges(self.state.in_edges(node))
        if not producers:
            return self.evaluator.before(self.frame, self.pos, node.data, index)
        if len(producers) != 1:
            raise Refused(f"{node.data} has several producers")
        edge = producers[0]
        binding = self.evaluator.hit(self.evaluator.state_write(self.state, edge, self.frame.bind, {}), index)
        if binding is None:
            raise Refused(f"{node.data} read outside what the state stores")
        src = edge.src
        if isinstance(src, (nodes.MapExit, nodes.Tasklet)):
            inner = self.state.memlet_path(edge)[0] if isinstance(src, nodes.MapExit) else edge
            params = self.evaluator.map_ranges(self.state, inner.src, self.frame.bind)
            if any(p not in binding for p in params):
                raise Refused("map iterations store one element")
            return self.tasklet_value(inner.src, inner.src_conn, {sympy.Symbol(p): binding[p] for p in params})
        if isinstance(src, nodes.AccessNode):
            return self.copied(edge, index)
        if isinstance(src, nodes.LibraryNode):
            return self.inner_product(src)
        raise Refused(f"value produced by {type(src).__name__}")

    def tasklet_value(
        self, tasklet: nodes.Tasklet, out_conn: str, params: dict[sympy.Symbol, sympy.Expr]
    ) -> sympy.Expr:
        """``tasklet``'s assignment to ``out_conn``, its inputs resolved, with the map parameters bound to
        ``params``."""
        if tasklet.code.language is not dtypes.Language.Python:
            raise Refused("non-Python tasklet")
        lines = [ln.strip() for ln in (tasklet.code.as_string or "").splitlines() if ln.strip()]
        matches = [ln for ln in lines if ln.split("=", 1)[0].strip() == out_conn]
        if len(matches) != 1 or matches[0].count("=") != 1:
            raise Refused(f"no unique assignment to {out_conn!r}")
        try:
            expr = plain(symbolic.pystr_to_symbolic(matches[0].split("=", 1)[1].strip()))
        except Exception as error:  # any construct sympy cannot hold is not an arithmetic statement
            raise Refused(f"unparsable tasklet: {error}") from error
        bind = {**self.frame.bind, **params}
        values: dict[sympy.Symbol, sympy.Expr] = {}
        for edge in self.state.in_edges(tasklet):
            if edge.dst_conn is None or edge.data is None or edge.data.is_empty():
                continue
            if isinstance(edge.src, nodes.Tasklet):
                values[sympy.Symbol(edge.dst_conn)] = self.tasklet_value(edge.src, edge.src_conn, params)
                continue
            source = self.state.memlet_path(edge)[0].src
            if not isinstance(source, nodes.AccessNode) or edge.data.data != source.data:
                raise Refused("tasklet input not read from an access node")
            index = tuple(x.xreplace(bind) for x in point_indices(edge.data.subset))
            values[sympy.Symbol(edge.dst_conn)] = self.value_at(source, index)
        return expr.xreplace(values)

    def copied(self, edge: MultiConnectorEdge[mm.Memlet], index: tuple[sympy.Expr, ...]) -> sympy.Expr:
        """The value a copy stores at ``index``: the source element at the same offset into the copied region."""
        src, dst = edge.data.get_src_subset(edge, self.state), edge.data.get_dst_subset(edge, self.state)
        if src is None or dst is None:
            raise Refused("copy without both subsets")
        src_axes, dst_axes = list(src.ndrange()), list(dst.ndrange())
        bind = self.frame.bind
        moving_src = [a for a, (b, e, _) in enumerate(src_axes) if not same(plain(b), plain(e))]
        moving_dst = [a for a, (b, e, _) in enumerate(dst_axes) if not same(plain(b), plain(e))]
        if len(moving_src) != len(moving_dst):
            raise Refused("copy reshapes")
        source = [plain(b).xreplace(bind) for b, _, _ in src_axes]
        for s, d in zip(moving_src, moving_dst):
            source[s] = source[s] + index[d] - plain(dst_axes[d][0]).xreplace(bind)
        return self.value_at(edge.src, tuple(sympy.expand(x) for x in source))

    def inner_product(self, node: nodes.LibraryNode) -> sympy.Expr:
        """``Sum(a[t] * b[t])`` of a vector-vector ``Dot`` / ``MatMul``."""
        if type(node).__name__ not in ("Dot", "MatMul"):
            raise Refused(f"library node {type(node).__name__}")
        sdfg = self.state.sdfg
        operands = value_edges(self.state.in_edges(node))
        if len(operands) != 2:
            raise Refused("inner product without two operands")
        t: sympy.Symbol | None = None
        product: sympy.Expr = sympy.S.One
        for edge in operands:
            if not isinstance(edge.src, nodes.AccessNode) or len(sdfg.arrays[edge.src.data].shape) != 1:
                raise Refused("inner product of a non-vector")
            ((begin, end, step),) = edge.data.subset.ndrange()
            lo, hi = plain(begin).xreplace(self.frame.bind), plain(end).xreplace(self.frame.bind)
            if not same(plain(step), 1):
                raise Refused("strided operand")
            if t is None:
                t = self.evaluator.symbol("dot", sympy.S.Zero, hi - lo)
            elif not same(self.evaluator.facts[t][1], hi - lo):
                raise Refused("operands of different lengths")
            product = product * self.value_at(edge.src, (lo + t,))
        assert t is not None
        return sympy.Sum(product, (t, sympy.S.Zero, self.evaluator.facts[t][1]))


def written_arrays(block: ControlFlowBlock) -> dict[str, None]:
    """Every container any state inside ``block`` writes."""
    states = [block] if isinstance(block, SDFGState) else block.states()
    return dict.fromkeys(n.data for st in states for n in st.data_nodes() if st.in_degree(n) > 0)


def summarize(sdfg: SDFG, loop: LoopRegion) -> Summary:
    """What one iteration of ``loop`` stores into the only non-transient container its body writes.

    :raises Refused: when the nest is not a unit-stride loop over ``0 .. extent - 1`` (either direction) around a
                     straight-line body, writes several
                     non-transient containers or one with a write that is not a domain, or a value is not derived
                     exactly.
    """
    lo, hi = loop_range(loop, {})
    if not same(lo, 0):
        raise Refused("loop does not start at 0")
    if any(isinstance(n, nodes.NestedSDFG) for st in loop.states() for n in st.nodes()):
        raise Refused("nested SDFG in the nest")
    written = written_arrays(loop)
    outputs = [name for name in written if not sdfg.arrays[name].transient]
    if len(outputs) != 1:
        raise Refused("not exactly one non-transient output")
    output = outputs[0]
    seq = chain(loop)
    evaluator = IterationEvaluator()
    domains: list[Domain] = []
    for write in (w for block in seq for w in evaluator.writes(block, output, {}, {})):
        if any(a.kind is AxisKind.OTHER for a in write):
            raise Refused("write of the output is not a domain")
        if any(same_axes(write, d.axes) for d in domains):
            continue
        index = tuple(a.lo if a.kind is AxisKind.FIXED else evaluator.symbol("q", a.lo, a.hi) for a in write)
        domains.append(Domain(write, index, sympy.S.Zero))
    top = Frame(seq, {}, Mode.TOP)
    for domain in domains:
        domain.value = evaluator.before(top, len(seq), output, domain.index)
    var, extent = sympy.Symbol(loop.loop_variable), sympy.expand(hi + 1)
    return Summary(output, var, extent, loop_step(loop) == 1, domains, written)


def same_axes(a: Write, b: Write) -> bool:
    return all(x.kind is y.kind and same(x.lo, y.lo) and same(x.hi, y.hi) for x, y in zip(a, b))


def is_single_element(desc: dt.Data) -> bool:
    """A ``Scalar`` or a length-1 ``Array`` -- the shape a runtime coefficient has."""
    return isinstance(desc, dt.Scalar) or (isinstance(desc, dt.Array) and all(str(s) == "1" for s in desc.shape))


@dataclass(frozen=True, slots=True)
class Scale:
    """A BLAS coefficient: ``value``, times the runtime ``name[0]`` when ``name`` is set."""

    value: sympy.Number
    name: str | None = None

    def is_one(self) -> bool:
        return self.name is None and self.value == 1


def scale_of(sdfg: SDFG, expr: sympy.Expr) -> Scale | None:
    """``expr`` as a :class:`Scale`: a number, the read ``x[0]`` of a single-element container, or their product."""
    value, name = sympy.S.One, None
    for factor in sympy.Mul.make_args(sympy.expand(expr)):
        if factor.is_Number:
            value = value * factor
        elif isinstance(factor, AppliedUndef) and len(factor.args) == 1 and factor.args[0] == 0 and name is None:
            desc = sdfg.arrays.get(read_array(factor))
            if desc is None or not is_single_element(desc):
                return None
            name = read_array(factor)
        else:
            return None
    return Scale(value, name)


class OwnIndex(NamedTuple):
    """An index that visits each of ``0 .. extent - 1`` once over the outer loop."""

    index: sympy.Expr
    rising: bool  #: grows from iteration to iteration


def own_indices(summary: Summary) -> list[OwnIndex]:
    """The loop variable itself and its mirror ``extent - 1 - i``."""
    i, n = summary.var, summary.extent
    return [OwnIndex(i, summary.ascending), OwnIndex(sympy.expand(n - 1 - i), not summary.ascending)]


def shape_is(sdfg: SDFG, name: str, *extents: sympy.Expr) -> bool:
    desc = sdfg.arrays.get(name)
    return (
        desc is not None
        and len(desc.shape) == len(extents)
        and all(same(plain(s), e) for s, e in zip(desc.shape, extents))
    )


def transients_dead_outside(sdfg: SDFG, loop: LoopRegion, summary: Summary) -> bool:
    """Whether every transient the nest writes is read nowhere outside it: splicing the nest away loses them."""
    inside = dict.fromkeys(id(st) for st in loop.states())
    scratch = dict.fromkeys(n for n in summary.written if n != summary.output)
    for state in sdfg.states():
        if id(state) in inside:
            continue
        if any(dn.data in scratch and state.out_degree(dn) > 0 for dn in state.data_nodes()):
            return False
    return True


def full_memlet(sdfg: SDFG, name: str) -> mm.Memlet:
    """A whole-array memlet on ``name``, with a fresh ``Range`` (memlets never share subsets)."""
    return mm.Memlet(data=name, subset=subsets.Range([(0, s - 1, 1) for s in sdfg.arrays[name].shape]))


def replace_loop_with_state(parent: ControlFlowRegion, loop: LoopRegion, label: str) -> SDFGState:
    """Splice ``loop`` out of ``parent``, replacing it with a fresh (returned) state that inherits the loop's in/out
    interstate edges."""
    was_start = parent.start_block is loop
    in_edges = list(parent.in_edges(loop))
    out_edges = list(parent.out_edges(loop))
    state = parent.add_state(label, is_start_block=was_start)
    for edge in in_edges:
        parent.add_edge(edge.src, state, edge.data)
    for edge in out_edges:
        condition = edge.data.condition.as_string if edge.data.condition is not None else "1"
        parent.add_edge(
            state, edge.dst, InterstateEdge(condition=condition, assignments=dict(edge.data.assignments or {}))
        )
    parent.remove_node(loop)
    return state


def blas_coefficients(alpha: Scale, beta: Scale | None) -> dict[str, object]:
    """The ``alpha``/``beta`` keyword arguments of a BLAS node; ``beta=None`` means the prior ``C`` is dropped."""
    beta = beta or Scale(sympy.S.Zero)
    return dict(
        alpha=python_number(alpha.value),
        beta=python_number(beta.value),
        alpha_input=alpha.name is not None,
        beta_input=beta.name is not None,
    )


def python_number(value: sympy.Number) -> int | float:
    return int(value) if value.is_Integer else float(value)


def splice_blas_node(
    parent: ControlFlowRegion,
    loop: LoopRegion,
    node: nodes.LibraryNode,
    operands: dict[str, str],
    output: str,
    alpha: Scale,
    beta: Scale | None,
) -> None:
    """Replace ``loop`` by a state holding ``node``: whole-array ``operands`` (connector to container), the prior
    ``output`` when the node reads it, the runtime coefficients, and the written ``output``."""
    sdfg = parent.sdfg
    state = replace_loop_with_state(parent, loop, node.label)
    state.add_node(node)
    for conn, name in operands.items():
        state.add_edge(state.add_read(name), None, node, conn, full_memlet(sdfg, name))
    if "_c" in node.in_connectors:
        state.add_edge(state.add_read(output), None, node, "_c", full_memlet(sdfg, output))
    for conn, scale in (("_alpha", alpha), ("_beta", beta)):
        if scale is not None and scale.name is not None:
            state.add_edge(state.add_read(scale.name), None, node, conn, mm.Memlet(f"{scale.name}[0]"))
    state.add_edge(node, "_c", state.add_write(output), None, full_memlet(sdfg, output))


def outer_loop_candidates(sdfg: SDFG) -> list[tuple[ControlFlowRegion, LoopRegion]]:
    """Every ``(parent, loop)`` whose ``loop`` directly contains another loop: the outer loop of a nest, nested
    SDFGs included."""
    return [
        (region, block)
        for region in sdfg.all_control_flow_regions(recursive=True)
        for block in region.nodes()
        if isinstance(block, LoopRegion) and any(isinstance(b, LoopRegion) for b in block.nodes())
    ]
