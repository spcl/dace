# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
from typing import Any

from dace import SDFG, InterstateEdge, subsets, symbolic
from dace import graphlib as nx
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, ControlFlowRegion, LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation
from dace.transformation.passes import analysis as ap

#: A write or read of the whole container (the subset of every access to a scalar).
WHOLE = "whole"
#: A read whose element an expression picks at run time: covered only by a write of the whole container.
ANY_ELEMENT = "any_element"


class AccessIndex:
    """Where each data container of one SDFG is accessed; built once per pass run, refreshed per state.

    ``nodes`` holds the access nodes of a state by container, computed on a state's first query.
    ``states`` and ``edge_sources`` index the whole SDFG on first need: the states holding an access
    node of a container, and the source block of every interstate edge naming it, in the order a walk
    of ``states()`` / ``all_interstate_edges()`` meets them.

    A rewrite keeps the index exact by calling :meth:`forget` on every state it changed.
    """

    __slots__ = ("sdfg", "by_state", "states_of", "edges_of")

    def __init__(self, sdfg: SDFG) -> None:
        self.sdfg = sdfg
        self.by_state: dict[SDFGState, dict[str, list[nd.AccessNode]]] = {}
        self.states_of: dict[str, list[SDFGState]] | None = None
        self.edges_of: dict[str, list[ControlFlowBlock]] | None = None

    def nodes(self, state: SDFGState, name: str) -> list[nd.AccessNode]:
        """The access nodes of ``name`` in ``state``."""
        table = self.by_state.get(state)
        if table is None:
            table = defaultdict(list)
            for node in state.data_nodes():
                table[node.data].append(node)
            self.by_state[state] = table
        return table.get(name, [])

    def states(self, name: str) -> list[SDFGState]:
        """Every state holding an access node of ``name``."""
        if self.states_of is None:
            self.states_of = defaultdict(list)
            for state in self.sdfg.states():
                for data in dict.fromkeys(node.data for node in state.data_nodes()):
                    self.states_of[data].append(state)
        return self.states_of.get(name, [])

    def edge_sources(self, name: str) -> list[ControlFlowBlock]:
        """The source block of every interstate edge whose free symbols name ``name``."""
        if self.edges_of is None:
            self.edges_of = defaultdict(list)
            for edge in self.sdfg.all_interstate_edges():
                for symbol_name in dict.fromkeys(str(s) for s in edge.data.free_symbols):
                    self.edges_of[symbol_name].append(edge.src)
        return self.edges_of.get(name, [])

    def interstate_names(self) -> set[str]:
        """Every name free on some interstate edge."""
        self.edge_sources("")
        return set(self.edges_of)

    def confined_to(self, name: str, states: set[SDFGState]) -> bool:
        """Whether every access of ``name`` lies in ``states`` and no interstate edge names it."""
        return all(state in states for state in self.states(name)) and not self.edge_sources(name)

    def forget(self, *states: SDFGState) -> None:
        """Drop what the index knows about ``states`` and the whole-SDFG tables, after a rewrite changed them."""
        for state in states:
            self.by_state.pop(state, None)
        self.states_of = None
        self.edges_of = None


def innermost_loop(block: ControlFlowBlock) -> LoopRegion | None:
    """The innermost ``LoopRegion`` enclosing ``block``, or ``None``."""
    region = block.parent_graph
    while region is not None:
        if isinstance(region, LoopRegion):
            return region
        region = region.parent_graph
    return None


def rename_memlet_tree(state: SDFGState, edge, old: str, new: str, node: nd.AccessNode) -> None:
    """Rename ``old`` -> ``new`` on every memlet of ``edge``'s memlet tree that belongs to ``node``'s access.

    The tree, not one path: an outer edge can fan out to several consumers inside a scope. It stops at
    another access node of ``old``: a staging copy puts two separately versioned accesses in one tree.
    ``_is_data_src`` is re-derived from the endpoints, since versioning a staging copy gives its two sides
    different names.
    """
    for pe in state.memlet_tree(edge):
        if pe.data is None or pe.data.data != old:
            continue
        if any(n is not node and isinstance(n, nd.AccessNode) and n.data == old for n in (pe.src, pe.dst)):
            continue
        pe.data.data = new
        if isinstance(pe.src, nd.AccessNode) and pe.src.data == new:
            pe.data._is_data_src = True
        elif isinstance(pe.dst, nd.AccessNode) and pe.dst.data == new:
            pe.data._is_data_src = False


def rename_access(state: SDFGState, node: nd.AccessNode, aliases: set[str], new: str) -> None:
    """Point ``node`` and every memlet on its edges that names one of ``aliases`` at ``new``.

    :param aliases: The original container name plus every version already minted for it: a node renamed
        twice in one pass no longer carries the original name on its memlets.
    """
    node.data = new
    for edge in list(state.in_edges(node)) + list(state.out_edges(node)):
        if edge.data.data in aliases:
            rename_memlet_tree(state, edge, edge.data.data, new, node)


def rename_into_nested_sdfgs(states, old: str, new: str) -> None:
    """Carry a rename ``old`` -> ``new`` across every ``NestedSDFG`` in ``states`` that an access renamed to
    ``new`` feeds or drains through a connector named ``old``.

    Renames the connector, the outer edge, the inner descriptor, inner access nodes, memlets and interstate
    edges, and a ``symbol_mapping`` key, then recurses into deeper nests. A connector is renamed only when
    its outer access node already carries ``new``: under per-scope versioning one nest is visited once per
    scope, and each scope owns only its own edges.
    """
    if not states or old == new:
        return
    for state in states:
        for n in state.nodes():
            if not isinstance(n, nd.NestedSDFG):
                continue
            in_match = old in n.in_connectors and any(
                e.dst_conn == old and isinstance(e.src, nd.AccessNode) and e.src.data == new for e in state.in_edges(n)
            )
            out_match = old in n.out_connectors and any(
                e.src_conn == old and isinstance(e.dst, nd.AccessNode) and e.dst.data == new for e in state.out_edges(n)
            )
            if not (in_match or out_match or old in n.symbol_mapping):
                continue
            if in_match:
                n.in_connectors[new] = n.in_connectors.pop(old)
                for e in state.in_edges(n):
                    if e.dst_conn == old and isinstance(e.src, nd.AccessNode) and e.src.data == new:
                        e.dst_conn = new
                        if e.data is not None and e.data.data == old:
                            e.data.data = new
            if out_match:
                n.out_connectors[new] = n.out_connectors.pop(old)
                for e in state.out_edges(n):
                    if e.src_conn == old and isinstance(e.dst, nd.AccessNode) and e.dst.data == new:
                        e.src_conn = new
                        if e.data is not None and e.data.data == old:
                            e.data.data = new
            # By hand, not ``SDFG.replace_dict``: its symbolic replacement trips on the scalar's
            # free-symbol typeclass coercion.
            inner = n.sdfg
            if old in inner.arrays:
                inner.arrays[new] = inner.arrays.pop(old)
                for inner_state in inner.states():
                    for inn in inner_state.data_nodes():
                        if inn.data == old:
                            inn.data = new
                    for ie in inner_state.edges():
                        if ie.data is not None and ie.data.data == old:
                            ie.data.data = new
            for ise in inner.all_interstate_edges():
                if any(str(s) == old for s in ise.data.free_symbols):
                    ise.data.replace_dict({old: new})
            if old in n.symbol_mapping:
                n.symbol_mapping[new] = n.symbol_mapping.pop(old)
            rename_into_nested_sdfgs(set(inner.states()), old, new)


def has_wcr_write(state: SDFGState, node: nd.AccessNode) -> bool:
    """Whether ``node`` is written through a write-conflict resolution: a read-modify-write, not a kill.

    A ``Reduce`` with no identity counts too (:func:`~dace.transformation.passes.analysis.analysis.has_wcr_in_edge`).
    """
    return ap.has_wcr_in_edge(state, node)


def axis_equal(a, b) -> bool:
    """Whether two ``(begin, end, step)`` axis ranges are provably identical."""
    return all(symbolic_equal(x, y) for x, y in zip(a, b))


def symbolic_equal(a, b) -> bool:
    """Whether ``a`` and ``b`` are provably equal (their difference simplifies to 0)."""
    try:
        return bool(symbolic.simplify(a - b) == 0)
    except (TypeError, AttributeError):
        return False


def subsets_cover(writes: list, read, facts: symbolic.Facts) -> bool:
    """Whether ``writes`` together provably cover ``read``.

    Either one write covers it (``covers_precise``: a strided write does not cover the dense box it spans),
    or several unit-step writes tile it as exactly adjacent chunks along one axis
    (``buf[0:M] = ...; buf[M:2*M] = ...`` read as ``buf[0:2*M]``). Chunk boundaries are matched by
    symbolic equality only.
    """
    if not isinstance(read, subsets.Range):
        return False
    dims = read.dims()
    if any(isinstance(w, subsets.Range) and w.dims() == dims and w.covers_precise(read, facts) for w in writes):
        return True
    chunks = [w for w in writes if isinstance(w, subsets.Range) and w.dims() == dims]
    if len(chunks) < 2:
        return False
    read_ranges = list(read.ranges)
    varying = None
    for ax in range(dims):
        if all(axis_equal(c.ranges[ax], read_ranges[ax]) for c in chunks):
            continue
        if varying is not None:
            return False
        varying = ax
    if varying is None:
        return False
    rb, re_, rs = read_ranges[varying]
    if rs != 1:
        return False
    intervals = []
    for c in chunks:
        b, e, st = c.ranges[varying]
        if st != 1:
            return False
        intervals.append((b, e))
    frontier = rb
    progressed = True
    while progressed:
        progressed = False
        for i, (b, e) in enumerate(intervals):
            if symbolic_equal(b, frontier):
                frontier = e + 1
                intervals.pop(i)
                progressed = True
                break
    return symbolic_equal(frontier, re_ + 1)


def point_subsets_equal(a: subsets.Range, b: subsets.Range) -> bool:
    """Equality of two single-point ranges, by the lower bound of each axis."""
    if len(a.ranges) != len(b.ranges):
        return False
    return all(
        symbolic.pystr_to_symbolic(ar[0]) == symbolic.pystr_to_symbolic(br[0]) for ar, br in zip(a.ranges, b.ranges)
    )


def edge_subset(state: SDFGState, edge, node: nd.AccessNode):
    """The subset of ``edge``'s memlet on ``node``'s side (by the memlet's own direction, so a self-copy
    ``A -> A`` reports the read on its source and the write on its destination)."""
    memlet = edge.data
    if memlet is None:
        return None
    return memlet.get_src_subset(edge, state) if edge.src is node else memlet.get_dst_subset(edge, state)


class Accesses:
    """What the access nodes of one container read and definitely write, as :class:`MustDefinition` asks.

    This base class is the whole-container view (scalars): a node with a non-empty incoming memlet
    writes the container, a node with an outgoing edge reads it, and any write covers any read.
    :class:`SlotAccesses` and :class:`SubsetAccesses` narrow it to one element and to memlet subsets.
    """

    def __init__(self, name: str) -> None:
        self.name = name

    def writes(self, state: SDFGState, node: nd.AccessNode) -> list:
        return [WHOLE] if any(not e.data.is_empty() for e in state.in_edges(node)) else []

    def reads(self, state: SDFGState, node: nd.AccessNode) -> list:
        return [WHOLE] if state.out_degree(node) > 0 else []

    def covered(self, writes: list, read) -> bool:
        return bool(writes)

    def key(self, write) -> Any:
        """Identity of a write in the must-def set, so a join keeps the writes every path made."""
        return write if write is WHOLE else str(write)

    def read_by(self, names: set[str]) -> list:
        """The reads an expression over ``names`` makes."""
        return [WHOLE] if self.name in names else []


class SlotAccesses(Accesses):
    """One constant element ``name[slot]``: only accesses at that point count, and only a write at a
    state's top scope is definite (a map with an empty range never runs its body)."""

    def __init__(self, name: str, slot: subsets.Range) -> None:
        super().__init__(name)
        self.slot = slot

    def touches(self, edges) -> bool:
        return any(
            e.data is not None
            and e.data.data == self.name
            and isinstance(e.data.subset, subsets.Range)
            and point_subsets_equal(e.data.subset, self.slot)
            for e in edges
        )

    def writes(self, state: SDFGState, node: nd.AccessNode) -> list:
        return [WHOLE] if state.entry_node(node) is None and self.touches(state.in_edges(node)) else []

    def reads(self, state: SDFGState, node: nd.AccessNode) -> list:
        return [WHOLE] if self.touches(state.out_edges(node)) else []

    def read_by(self, names: set[str]) -> list:
        return [ANY_ELEMENT] if self.name in names else []

    def covered(self, writes: list, read) -> bool:
        return read is not ANY_ELEMENT and bool(writes)


class SubsetAccesses(Accesses):
    """Memlet subsets of an array: every incoming memlet writes its destination subset, every outgoing one
    reads its source subset, and a read is covered when the writes together tile it (:func:`subsets_cover`)."""

    def __init__(self, name: str, facts: symbolic.Facts) -> None:
        super().__init__(name)
        self.facts = facts

    def writes(self, state: SDFGState, node: nd.AccessNode) -> list:
        found = (edge_subset(state, e, node) for e in state.in_edges(node) if not e.data.is_empty())
        return [w for w in found if w is not None]

    def reads(self, state: SDFGState, node: nd.AccessNode) -> list:
        found = (edge_subset(state, e, node) for e in state.out_edges(node) if not e.data.is_empty())
        return [r for r in found if r is not None]

    def read_by(self, names: set[str]) -> list:
        return [ANY_ELEMENT] if self.name in names else []

    def covered(self, writes: list, read) -> bool:
        return read is not ANY_ELEMENT and subsets_cover(writes, read, self.facts)


class MustDefinition:
    """Forward must-definition of one container over the control-flow tree, for the question "is every read
    preceded, on every path from the region's entry, by a write that covers it?".

    The region tree is walked structurally (no ``states()`` list per query). Within a state a read is covered by
    the writes of its own node and of every node with a dataflow path to it. A ``ConditionalBlock`` defines what
    all its branches define when it has an ``else``; a loop may run zero times and defines nothing. Cycles in a
    region are seeded pessimistically. Branch and loop-header conditions, and interstate edges, read the names
    they mention. One instance per container and pass run: what a block defines is cached.
    """

    def __init__(self, accesses: Accesses, index: AccessIndex) -> None:
        self.accesses = accesses
        self.index = index
        self.gen_cache: dict[ControlFlowBlock, dict] = {}

    def exposed(self, region: ControlFlowRegion, defined: dict | None = None) -> bool:
        """Whether a read in ``region`` may see a value from before ``region`` was entered.

        :param defined: The writes known to have happened on entry, by :meth:`Accesses.key`.
        """
        return self.analyze(region, defined or {})[1]

    def defined_at_exit(self, region: ControlFlowRegion, defined: dict | None = None) -> dict:
        """The writes every path through ``region`` has made at its exit."""
        return self.analyze(region, defined or {})[0]

    def analyze(self, region: ControlFlowRegion, entry: dict) -> tuple[dict, bool]:
        blocks = list(region.nodes())
        if not blocks:
            return dict(entry), False
        gen = {b: self.gen(b) for b in blocks}
        start = region.start_block
        defs: dict[ControlFlowBlock, dict] = {b: {} for b in blocks}
        defs[start] = dict(entry)
        for _ in range(len(blocks) + 1):
            changed = False
            for b in blocks:
                if b is start:
                    continue
                preds = [e.src for e in region.in_edges(b)]
                new = self.meet([{**defs[p], **gen[p]} for p in preds]) if preds else {}
                if new.keys() != defs[b].keys():
                    defs[b] = new
                    changed = True
            if not changed:
                break
        exposed = any(self.block_exposed(b, defs[b]) for b in blocks)
        if not exposed:
            for edge in region.edges():
                reads = self.accesses.read_by({str(s) for s in edge.data.free_symbols})
                after = list({**defs[edge.src], **gen[edge.src]}.values())
                if any(not self.accesses.covered(after, r) for r in reads):
                    exposed = True
                    break
        sinks = [b for b in blocks if region.out_degree(b) == 0]
        exit_defs = self.meet([{**defs[b], **gen[b]} for b in sinks]) if sinks else {}
        return exit_defs, exposed

    @staticmethod
    def meet(sets: list[dict]) -> dict:
        if not sets:
            return {}
        first, rest = sets[0], sets[1:]
        return {k: v for k, v in first.items() if all(k in other for other in rest)}

    def gen(self, block: ControlFlowBlock) -> dict:
        """The writes every path through ``block`` makes, independent of what held on entry."""
        cached = self.gen_cache.get(block)
        if cached is not None:
            return cached
        self.gen_cache[block] = {}
        result: dict = {}
        if isinstance(block, SDFGState):
            for node in self.index.nodes(block, self.accesses.name):
                for w in self.accesses.writes(block, node):
                    result[self.accesses.key(w)] = w
        elif isinstance(block, ConditionalBlock):
            if any(cond is None for cond, _ in block.branches):
                result = self.meet([self.analyze(branch, {})[0] for _, branch in block.branches])
        elif isinstance(block, ControlFlowRegion) and not isinstance(block, LoopRegion):
            result = self.analyze(block, {})[0]
        self.gen_cache[block] = result
        return result

    def code_reads(self, codes) -> list:
        names: set[str] = set()
        for code in codes:
            if code is not None:
                names |= code.get_free_symbols()
        return self.accesses.read_by(names)

    def block_exposed(self, block: ControlFlowBlock, defined: dict) -> bool:
        acc = self.accesses
        held = list(defined.values())
        if isinstance(block, SDFGState):
            return self.state_exposed(block, held)
        if isinstance(block, (ConditionalBlock, ControlFlowRegion)):
            if any(not acc.covered(held, r) for r in self.code_reads(block.get_meta_codeblocks())):
                return True
            if isinstance(block, ConditionalBlock):
                return any(self.analyze(branch, defined)[1] for _, branch in block.branches)
            return self.analyze(block, defined)[1]
        # Break, continue and return blocks hold no dataflow; an unknown block is conservatively a read.
        return any(not acc.covered(held, r) for r in acc.read_by({acc.name}))

    def state_exposed(self, state: SDFGState, held: list) -> bool:
        acc = self.accesses
        accessed = self.index.nodes(state, acc.name)
        if not accessed:
            return False
        writes = {node: acc.writes(state, node) for node in accessed}
        for node in accessed:
            for read in acc.reads(state, node):
                if acc.covered(held + writes[node], read):
                    continue
                before = [
                    w
                    for other in accessed
                    if other is not node and writes[other] and nx.has_path(state.nx, other, node)
                    for w in writes[other]
                ]
                if not before or not acc.covered(held + writes[node] + before, read):
                    return True
        return False


@transformation.explicit_cf_compatible
class ScalarFission(ppl.Pass):
    """
    Fission transient scalars or arrays of size 1 that are dominated by a write into separate data containers.
    """

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.AccessNodes

    def shadow_analysis(self) -> type:
        """The write-shadow-scope analysis whose result drives this pass.

        :returns: The analysis ``Pass`` subclass to read ``pipeline_results`` from.
        """
        return ap.ScalarWriteShadowScopes

    def accepts(self, desc) -> bool:
        """Whether ``desc`` is a container this pass may fission.

        Scalars accept every size-1 transient: any write to one is a full write, so the analysis'
        dominance-only shadowing is already a must-def. :class:`ArrayFission` overrides both this
        and :func:`shadow_analysis`.

        :param desc: The data descriptor to test.
        :returns: ``True`` if the container may be fissioned.
        """
        return desc.transient and desc.total_size == 1

    def depends_on(self):
        return [self.shadow_analysis()]

    def apply_pass(self, sdfg: SDFG, pipeline_results: dict[str, Any]) -> dict[str, set[str]] | None:
        """
        Rename scalars and arrays of size 1 based on dominated scopes.

        :param sdfg: The SDFG to modify.
        :param pipeline_results: If in the context of a ``Pipeline``, a dictionary that is populated with prior Pass
                                 results as ``{Pass subclass name: returned object from pass}``. If not run in a
                                 pipeline, an empty dictionary is expected.
        :return: A dictionary mapping the original name to a set of all new names created for each data container.
        """
        results: dict[str, set[str]] = defaultdict(lambda: set())

        shadow_scope_dict: ap.WriteScopeDict = pipeline_results[self.shadow_analysis().__name__][sdfg.cfg_id]

        # A loop header (init, condition, update) READS every data container it names, at points
        # the write-shadow analysis does not model, so fission would rename the writer and orphan
        # the header's reference. Leave such scalars untouched. A branch condition is modelled: it
        # reads on entry to its conditional and is renamed with its scope.
        cond_referenced: set[str] = set()
        # Read by a branch condition: versioned with its dominating write only. The undominated
        # privatization below proves its legality from access nodes, which a condition read is not.
        branch_referenced: set[str] = set()
        for region in sdfg.all_control_flow_regions():
            names = {s for cb in region.get_meta_codeblocks() for s in cb.get_free_symbols()}
            if isinstance(region, ConditionalBlock):
                branch_referenced |= names
            else:
                cond_referenced |= names
        cond_referenced &= set(sdfg.arrays.keys())

        # A scalar threaded through a single ``NestedSDFG`` inout connector is a
        # loop-carried value whose read and write share one storage location (the
        # connector); versioning its two sides into different containers is invalid
        # (see ``_inout_nsdfg_carried``). Leave it untouched, like ``cond_referenced``.
        inout_carried: set[str] = self._inout_nsdfg_carried(sdfg)

        index = AccessIndex(sdfg)
        for name, write_scope_dict in shadow_scope_dict.items():
            desc = sdfg.arrays[name]

            # Only containers this pass owns; a non-transient one may be used externally.
            if not self.accepts(desc):
                continue

            # A scalar read by a loop header is left whole (see ``cond_referenced``).
            if name in cond_referenced:
                continue

            # A scalar read AND written through a single NestedSDFG inout connector is
            # a carried value shared across both sides of that one connector; fission
            # would give the read side (``beta_0``) a different version from the write
            # side (``beta_1``) while the connector must name ONE array on both sides,
            # producing an invalid inout connector. Leave it (see ``_inout_nsdfg_carried``).
            if name in inout_carried:
                continue

            # Privatize the undominated (``None``) scope -- accesses whose reads are
            # not dominated by a single write, e.g. a scalar written in every
            # branch of a conditional and read after the merge (the cloudsc
            # zcor/zfac/zqe pattern). These are loop-local (no upward-exposed use),
            # so each enclosing loop gets its own copy -- scalar privatization.
            if None in write_scope_dict and name not in branch_referenced:
                # The index is filled lazily, after any renames made before. A rename only rewrites the
                # container being processed, so the entries of the rest stay exact.
                self._privatize_loop_local_undominated(sdfg, name, write_scope_dict[None], results, index)

            # A write-conflict-resolution write does not KILL the previous value, it reads it
            # (see :func:`has_wcr_write`), so it cannot start a new single-assignment version.
            # Skipping just that one scope is not enough: the write before it would still be
            # versioned while the WCR write kept the old name, cutting the chain between them. So a
            # container with a version-starting WCR write is left whole. (The undominated
            # privatization above is unaffected: it renames a loop's entire access group at once,
            # a pure alpha-rename that stays correct for an accumulator.)
            if any(w is not None and has_wcr_write(w[0], w[1]) for w in write_scope_dict):
                continue

            # One dominating scope needs no fission unless undominated accesses share its name: a dead
            # initializer before a loop that rewrites the temporary every iteration keeps the loop's
            # accesses tied to the outer scope, and LoopToMap refuses them.
            dominated = [w for w in write_scope_dict if w is not None]
            if len(dominated) == 0 or (len(dominated) == 1 and None not in write_scope_dict):
                continue

            # Versioning gives each access ONE name, so an access in two scopes (a read after a
            # conditional whose branches write in nested loops, which the analysis cannot merge into
            # one scope) would read only one branch's version. Leave such a container whole.
            seen: set[int] = set()
            overlapping = False
            for accesses in write_scope_dict.values():
                for _block, access in accesses:
                    overlapping |= id(access) in seen
                    seen.add(id(access))
            if overlapping:
                continue

            for write, shadowed_reads in write_scope_dict.items():
                if write is not None and len(shadowed_reads) > 0:
                    newdesc = desc.clone()
                    # Versions already minted count too: a node can be renamed twice in one pass.
                    aliases = {name} | set(results[name])
                    newname = sdfg.add_datadesc(name, newdesc, find_new_name=True)

                    # Replace the write and any connected memlets with writes to the new data container.
                    rename_access(write[0], write[1], aliases, newname)

                    # Replace all dominated reads and connected memlets.
                    affected_states: set[SDFGState] = {write[0]} if isinstance(write[0], SDFGState) else set()
                    for read in shadowed_reads:
                        if isinstance(read[1], nd.AccessNode):
                            rename_access(read[0], read[1], aliases, newname)
                            if isinstance(read[0], SDFGState):
                                affected_states.add(read[0])
                        elif isinstance(read[1], InterstateEdge):
                            read[1].replace_dict({name: newname})
                        elif isinstance(read[1], ConditionalBlock):
                            read[1].replace_meta_accesses({name: newname})

                    # Propagate the rename across every NestedSDFG boundary
                    # touched by the renamed accesses: connector names, inner
                    # arrays catalog, inner accesses + memlets, symbol_mapping.
                    rename_into_nested_sdfgs(affected_states, name, newname)
                    results[name].add(newname)
        return results

    def report(self, pass_retval: Any) -> str | None:
        return f"Renamed {len(pass_retval)} scalars: {pass_retval}."

    @staticmethod
    def _inout_nsdfg_carried(sdfg: SDFG) -> set[str]:
        """Names threaded through a single ``NestedSDFG`` inout connector.

        A connector present in BOTH ``in_connectors`` and ``out_connectors`` reads
        and writes ONE inner array through one boundary; when both its outer sides
        reference the same array (``beta`` in / ``beta`` out) that array is a
        loop-carried value shared across the connector, not a privatizable
        per-iteration temporary. The write-shadow analysis, blind to that join,
        would give the read a different version (``beta_0``) from the write
        (``beta_1``); renaming the boundary memlets then leaves the connector --
        which must name the SAME array on both sides -- reading ``beta_0`` and
        writing ``beta_1``, an invalid inout connector (``NestedSDFG.validate``).
        This is aggravated when the NSDFG sits inside a Map: its edges terminate at
        the MapEntry/MapExit, so ``_propagate_rename_into_nsdfgs`` (which renames the
        connector only when the edge's own endpoint is the renamed AccessNode) never
        fires, while ``_rename_memlet_path`` still renames the pass-through memlets.
        Collect such names so the caller leaves them for the carry handling.
        """
        carried: set[str] = set()
        for state in sdfg.states():
            for n in state.nodes():
                if not isinstance(n, nd.NestedSDFG):
                    continue
                for conn in n.in_connectors.keys() & n.out_connectors.keys():
                    in_arrays = {
                        e.data.data
                        for e in state.in_edges(n)
                        if e.dst_conn == conn and e.data is not None and e.data.data
                    }
                    out_arrays = {
                        e.data.data
                        for e in state.out_edges(n)
                        if e.src_conn == conn and e.data is not None and e.data.data
                    }
                    carried |= in_arrays & out_arrays
        return carried

    # #
    #  Privatization of undominated (None-scope) loop-local scalars
    # #

    def _privatize_loop_local_undominated(
        self, sdfg: SDFG, name: str, accesses: set[tuple], results, index: AccessIndex | None = None
    ):
        """Give a separate container to each loop's copy of a scalar whose reads are not dominated by a
        single write (the ``None`` write-scope), when it is provably loop-local: no upward-exposed use in
        any loop touching it (see :meth:`_carrier_free`). A scalar that may be read before it is written
        (a non-exhaustive ``if``) is loop-carried and is left alone.

        :param name: The size-1 transient scalar.
        :param accesses: The ``None``-scope ``(block, node-or-edge)`` accesses.
        :param results: Accumulator mapping the original name to new names.
        :param index: Where each container is accessed, shared by a run of queries; ``None`` builds one.
        """
        index = index or AccessIndex(sdfg)
        if not self._carrier_free(sdfg, name, index):
            return

        by_loop: dict[LoopRegion, list[tuple]] = defaultdict(list)
        for block, node in accesses:
            by_loop[innermost_loop(block)].append((block, node))

        for loop_accesses in by_loop.values():
            newname = sdfg.add_datadesc(name, sdfg.arrays[name].clone(), find_new_name=True)
            affected_states: set[SDFGState] = set()
            for block, node in loop_accesses:
                if isinstance(node, nd.AccessNode):
                    rename_access(block, node, {name}, newname)
                    if isinstance(block, SDFGState):
                        affected_states.add(block)
                elif isinstance(node, InterstateEdge):
                    node.replace_dict({name: newname})
            rename_into_nested_sdfgs(affected_states, name, newname)
            results[name].add(newname)

    def _carrier_free(self, sdfg: SDFG, name: str, index: AccessIndex | None = None) -> bool:
        """Whether no value of ``name`` can flow from one loop to another, or out to non-loop scope.

        The undominated (``None``) write scope is one equivalence class with no def-use guarantee, so
        privatizing its loop groups independently is value-preserving only when every access of ``name``
        lives inside a loop that definitely defines it before reading it, and no such loop nests in
        another (a nested loop's last value is live out into its parent). Otherwise the producing group
        could be renamed away from its consumer (npbench ``vadv``).

        :param index: Where each container is accessed, built once for a run of queries; ``None`` builds one.
        """
        index = index or AccessIndex(sdfg)
        loops: set[LoopRegion] = set()
        for block in (*index.states(name), *index.edge_sources(name)):
            loop = innermost_loop(block)
            if loop is None:
                return False
            loops.add(loop)
        for loop in loops:
            outer = innermost_loop(loop)
            while outer is not None:
                if outer in loops:
                    return False
                outer = innermost_loop(outer)
        must_def = MustDefinition(Accesses(name), index)
        return not any(must_def.exposed(loop) for loop in loops)


@transformation.explicit_cf_compatible
class PrivatizeScalars(ppl.Pipeline):
    """Give every write-before-read transient scalar its own name (scalar privatization).

    A self-contained pipeline that runs :class:`ScalarFission` together with the
    analysis it depends on (``ScalarWriteShadowScopes`` and that pass's own
    dependencies), so it can be applied on its own --
    ``PrivatizeScalars().apply_pass(sdfg, {})`` -- or dropped in as a single
    element of a larger pipeline such as simplify. ``ScalarFission`` declares
    those dependencies but cannot resolve them outside a pipeline.

    A size-1 transient reused as a per-iteration temporary is written before it
    is read on every iteration, so the iterations share only its *name*, not a
    real value. Splitting each dominating write into its own container removes
    that false write-after-write, which is what otherwise makes a shared
    loop-local scalar block ``LoopToMap``.
    """

    def __init__(self):
        super().__init__([ScalarFission()])


@transformation.explicit_cf_compatible
class ArrayFission(ScalarFission):
    """
    Fission transient ARRAYS that every write provably overwrites in full, into separate data
    containers -- the array analogue of :class:`ScalarFission`.

    The renaming machinery is identical to the scalar case; what differs is the premise it needs.
    Versioning a container per dominating write is only value-preserving when that write is a
    must-def of the whole container, which is automatic for a scalar and has to be *proven* for an
    array. The proof lives entirely in :class:`~dace.transformation.passes.analysis.analysis.
    ArrayWriteShadowScopes`, which reports an array only when every one of its writes covers the
    full extent (all dimensions, unit stride, no WCR, not dynamic, not through a NestedSDFG
    connector). A partially written array, an array written only under a condition, or one whose
    extent cannot be simplified to the declared shape is never reported and so is never touched.
    """

    def shadow_analysis(self) -> type:
        return ap.ArrayWriteShadowScopes

    def accepts(self, desc) -> bool:
        # Size-1 containers stay with ``ScalarFission``; the analysis has already discharged the
        # full-overwrite proof for everything else it reports.
        return desc.transient and desc.total_size != 1

    def report(self, pass_retval: Any) -> str | None:
        return f"Renamed {len(pass_retval)} arrays: {pass_retval}."


@transformation.explicit_cf_compatible
class PrivatizeArrays(ppl.Pipeline):
    """Give every provably fully-overwritten transient array its own name (array privatization).

    The array sibling of :class:`PrivatizeScalars`: a self-contained pipeline running
    :class:`ArrayFission` together with the analysis it depends on, so it can be applied on its own
    -- ``PrivatizeArrays().apply_pass(sdfg, {})`` -- or dropped into a larger recipe.

    A transient array reused as a per-iteration temporary (``Tz = np.zeros(...)`` at the top of a
    loop body) is fully written before it is read on every iteration, so the iterations share only
    its *name*. Splitting each dominating write into its own container removes that false
    write-after-write, which is what otherwise makes a shared temporary look loop-carried. Unlike
    the scalar case the sharing is only false when the write covers the entire array, so the
    underlying analysis refuses every array it cannot prove that for.
    """

    def __init__(self):
        super().__init__([ArrayFission()])
