# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``UniqueLoopIterators``: give every ``LoopRegion`` iterator a name no other loop uses.

Loops from independent sources can share an iterator name; merged into one SDFG, the shared name aliases them. Each
loop variable is renamed to ``_loop_it_<N>`` in the loop, the descriptors it sizes and the symbol mappings of nested
SDFGs. With ``assign_loop_iterator_post_value``, the original name is assigned its exit value after the loop when a
later block reads it, as Fortran leaves the iterator of a counted ``DO`` loop.
"""

import dace
from dace import nodes, symbolic
from dace.sdfg.replace import replace_properties_dict
from dace.sdfg.state import ControlFlowBlock, ControlFlowRegion, LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import ControlFlowBlockReachability, loop_analysis
from dace.transformation.transformation import explicit_cf_compatible

#: Prefix of the renamed iterators.
LOOP_ITERATOR_PREFIX = "_loop_it_"
#: Label prefix of the state assigning an iterator its exit value.
POST_VALUE_STATE_PREFIX = "loop_iter_post_value"


def iterator_id(name: str) -> int:
    """The ``<N>`` of ``_loop_it_<N>``, or -1 for any other name."""
    suffix = name[len(LOOP_ITERATOR_PREFIX) :]
    return int(suffix) if name.startswith(LOOP_ITERATOR_PREFIX) and suffix.isdigit() else -1


def first_free_id(sdfg: dace.SDFG) -> int:
    """One past the largest ``<N>`` any loop iterator or map parameter of the SDFG tree uses.

    Map parameters count because ``LoopToMap`` keeps a loop's iterator name as its map parameter.
    """
    used = [-1]
    for nested in sdfg.all_sdfgs_recursive():
        used += [iterator_id(r.loop_variable) for r in nested.all_control_flow_regions() if isinstance(r, LoopRegion)]
        for state in nested.states():
            used += [iterator_id(p) for n in state.nodes() if isinstance(n, nodes.MapEntry) for p in n.map.params]
    return max(used) + 1


def rename_iterator(cfg: ControlFlowRegion, old: str, new: str) -> None:
    """Rename ``old`` to ``new`` in ``cfg``, the descriptors its states access and the nested SDFGs importing it."""
    repl = {old: new}
    cfg.replace_meta_accesses(repl)
    cfg.replace_dict(repl)
    if not isinstance(cfg, dace.SDFG):  # ``SDFG.replace_dict`` renames its own descriptors
        accessed = {n.data for state in cfg.states() for n in state.data_nodes()}
        for name in accessed:
            desc = cfg.sdfg.arrays[name]
            if old in {str(s) for s in desc.free_symbols}:
                replace_properties_dict(desc, repl)
    for state in cfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            # ``replace_dict`` renamed the mapped value; the key is the nested SDFG's own name for it.
            if old in node.symbol_mapping:
                node.symbol_mapping[new] = node.symbol_mapping.pop(old)
            if old in node.sdfg.symbols:
                rename_iterator(node.sdfg, old, new)


def exit_value(loop: LoopRegion) -> symbolic.SymbolicType | None:
    """The iterator value after a counted loop: one step past the last value it takes."""
    end = loop_analysis.get_loop_end(loop)
    if end is None:
        return None
    step = loop_analysis.get_loop_stride(loop)
    init = loop_analysis.get_init_assignment(loop)
    if step is None:
        return end
    if init is None:
        return end + step
    return init + symbolic.int_floor(end - init + step, step) * step


def symbol_readers(sdfg: dace.SDFG, names: set[str]) -> dict[str, set[ControlFlowBlock]]:
    """The blocks of the SDFG tree that read each of ``names``; an interstate edge reads in its destination."""
    readers: dict[str, set[ControlFlowBlock]] = {}
    for cfr in sdfg.all_control_flow_regions(recursive=True):
        for block in cfr.nodes():
            for name in block.free_symbols & names:
                readers.setdefault(name, set()).add(block)
        for edge in cfr.edges():
            for name in edge.data.read_symbols() & names:
                readers.setdefault(name, set()).add(edge.dst)
    return readers


def exit_value_is_read(loop: LoopRegion, name: str, readers: dict[str, set[ControlFlowBlock]], reach: dict) -> bool:
    """Whether ``name`` may be read after ``loop``, so its exit value has to be assigned.

    The one case it is not: every reader lies outside the loop and none after it. ``name`` is then an input that only
    shares the iterator's name, and assigning it would declare a local shadowing that input.
    """
    blocks = readers.get(name)
    if not blocks:
        return True
    inside = {loop}
    for cfr in loop.all_control_flow_regions(recursive=True):
        inside.add(cfr)
        inside.update(cfr.nodes())
    if blocks <= inside:
        return True
    after = reach.get(loop.parent_graph.cfg_id, {}).get(loop)
    return not after or not blocks.isdisjoint(after)


@dace.properties.make_properties
@explicit_cf_compatible
class UniqueLoopIterators(ppl.Pass):
    """Rename the iterator of every ``LoopRegion`` in the SDFG tree to a unique ``_loop_it_<N>``.

    ``<N>`` counts from past every ``_loop_it_<N>`` already in the tree, so the pass is idempotent and deterministic.
    """

    assign_loop_iterator_post_value = dace.properties.Property(
        dtype=bool,
        default=True,
        desc="Assign the original iterator name its exit value after the loop when a later block reads it",
    )

    def __init__(self, assign_loop_iterator_post_value: bool = True):
        super().__init__()
        self.assign_loop_iterator_post_value = assign_loop_iterator_post_value

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: dace.SDFG, _) -> set[str] | None:
        """:returns: The new iterator names, or ``None`` if every iterator was unique already."""
        self.next_id = first_free_id(sdfg)
        self.readers: dict[str, set[ControlFlowBlock]] = {}
        self.reach: dict = {}
        if self.assign_loop_iterator_post_value:
            names = {
                r.loop_variable
                for r in sdfg.all_control_flow_regions(recursive=True)
                if isinstance(r, LoopRegion) and r.loop_variable
            }
            if names:
                # Renaming adds no blocks the queries depend on, so both are computed once, up front.
                self.reach = ControlFlowBlockReachability().apply_pass(sdfg, {})
                self.readers = symbol_readers(sdfg, names)
        return self.rename_loops(sdfg) or None

    def rename_loops(self, sdfg: dace.SDFG) -> set[str]:
        renamed: set[str] = set()
        iterators = [
            r.loop_variable for r in sdfg.all_control_flow_regions() if isinstance(r, LoopRegion) and r.loop_variable
        ]
        # Loop fission copies a loop with its already unique name, so a duplicate is renamed again.
        duplicated = {name for name in iterators if iterators.count(name) > 1}
        for loop in sdfg.all_control_flow_regions():
            if not isinstance(loop, LoopRegion) or not loop.loop_variable:
                continue
            old = loop.loop_variable
            if iterator_id(old) >= 0 and old not in duplicated:
                continue
            new = f"{LOOP_ITERATOR_PREFIX}{self.next_id}"
            rename_iterator(loop, old, new)
            renamed.add(new)
            if self.assign_loop_iterator_post_value:
                value = exit_value(loop) if exit_value_is_read(loop, old, self.readers, self.reach) else None
                if value is not None:
                    loop.parent_graph.add_state_after(
                        loop,
                        f"{POST_VALUE_STATE_PREFIX}_{self.next_id}",
                        assignments={old: f"({symbolic.symstr(value, arrayexprs=frozenset(sdfg.arrays))})"},
                    )
            elif old in sdfg.symbols and old not in sdfg.used_symbols(all_symbols=False):
                # Nothing reads the old name any more; its declaration would leak as a free symbol of the SDFG.
                sdfg.remove_symbol(old)
            self.next_id += 1

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG):
                    renamed |= self.rename_loops(node.sdfg)
        return renamed
