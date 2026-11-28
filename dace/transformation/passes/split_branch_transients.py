# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give each branch its own copy of a transient that only the branches of one conditional use.

Fortran allocates a local array in each arm of an ``IF`` with the arm's own size (QE's ``fwfft_orbital_gamma``:
``ALLOCATE(psio(ngk(1), incr))`` in one arm, ``ALLOCATE(psio(ngk(1), brange))`` in the other). The frontend gives both
one container sized by symbols each arm assigns, and one allocation cannot follow two sizes. Each arm uses its own
allocation only, so each arm gets its own container: the allocation of each then follows its arm's size.

Allocations one after another take a container each the same way (QE's ``p_psi``: ``ALLOCATE(coeff(nbnd, m))`` ...
``DEALLOCATE(coeff)`` in one ``IF``, then ``ALLOCATE(coeff(ibnd_2 - ibnd_1 + 1, m))`` in the next): when every block
that uses it assigns its size before each of its uses, no value crosses from one block to the next.
"""

import copy

from dace import data, dtypes, properties
from dace.codegen.targets.framecode import contains, precedes
from dace.ordered import OrderedSet
from dace.sdfg import SDFG
from dace.sdfg.state import AbstractControlFlowRegion, ConditionalBlock, ControlFlowBlock, SDFGState
from dace.transformation import pass_pipeline as ppl


def enclosing_branch(block: ControlFlowBlock) -> tuple[ConditionalBlock, int] | None:
    """The innermost conditional around ``block`` and the index of the branch holding it."""
    child, parent = block, block.parent_graph
    # Climb until the region that is a branch of a conditional
    while parent is not None and not isinstance(parent, SDFG):
        if isinstance(parent, ConditionalBlock):
            # Find which branch the child is
            for index, branch in enumerate(region for condition, region in parent.branches):
                if branch is child:
                    return parent, index
        child, parent = parent, parent.parent_graph
    return None


def branch_of(block: ControlFlowBlock, conditional: ConditionalBlock) -> int | None:
    """The index of the branch of ``conditional`` that holds ``block``, at any depth."""
    found = enclosing_branch(block)
    # Climb conditional by conditional until the one asked for
    while found is not None:
        if found[0] is conditional:
            return found[1]
        found = enclosing_branch(found[0])
    return None


@properties.make_properties
class SplitBranchTransients(ppl.Pass):
    """One container per branch for a transient used only inside two or more branches of one conditional, whose size a
    branch assigns, so each copy is allocated where its branch assigns its size."""

    CATEGORY: str = "Simplification"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Descriptors | ppl.Modifies.AccessNodes | ppl.Modifies.Memlets

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def apply_pass(self, sdfg: SDFG, pipeline_results) -> int | None:
        """:returns: How many containers were split, or ``None`` if none were."""
        split = 0
        # Each split can expose another inside a copy (two allocations in a row in one arm): repeat until none
        while (found := self.split_once(sdfg)) > 0:
            split += found
        return split or None

    def split_once(self, sdfg: SDFG) -> int:
        """Split every candidate once; :returns: how many containers were split."""
        split = 0
        # Each SDFG owns its transients
        for sd in list(sdfg.all_sdfgs_recursive()):
            users: dict[str, list[SDFGState]] = {}
            # Record the states that use each container
            for state in sd.states():
                # Every access node is a use
                for node in state.data_nodes():
                    users.setdefault(node.data, []).append(state)
            read_on_edges = OrderedSet(name for edge in sd.all_interstate_edges() for name in edge.data.free_symbols)
            # Each candidate splits when its users span two or more branches of one conditional
            for name, states in users.items():
                desc = sd.arrays[name]
                if not self.candidate(desc) or name in read_on_edges:
                    continue
                found = enclosing_branch(states[0])
                # Widen to the conditional whose branches hold every user
                while found is not None:
                    arms = OrderedSet(branch_of(state, found[0]) for state in states)
                    if None not in arms:
                        break
                    found = enclosing_branch(found[0])
                if found is not None and len(arms) >= 2 and self.sized_in(desc, found[0], arms):
                    self.split(sd, name, {f"branch{index}": found[0].branches[index][1] for index in sorted(arms)})
                    split += 1
                    continue
                blocks = self.sequential_blocks(name, desc, states)
                if blocks:
                    self.split(sd, name, {f"part{index}": block for index, block in enumerate(blocks)})
                    split += 1
        return split

    @staticmethod
    def candidate(desc: data.Data) -> bool:
        """A transient array sized by symbols, allocated by the code that uses it."""
        return (
            isinstance(desc, data.Array)
            and not isinstance(desc, data.View)
            and desc.transient
            and bool(desc.free_symbols)
            and desc.lifetime == dtypes.AllocationLifetime.Scope
        )

    @staticmethod
    def sized_in(desc: data.Data, conditional: ConditionalBlock, arms: OrderedSet[int]) -> bool:
        """Whether a branch assigns a symbol the size of ``desc`` reads: a size fixed before the conditional keeps one
        allocation for all branches."""
        sizes = OrderedSet(str(symbol) for symbol in desc.free_symbols)
        return any(
            sizes & edge.data.assignments.keys()
            for index in arms
            for edge in conditional.branches[index][1].all_interstate_edges()
        )

    @staticmethod
    def sequential_blocks(name: str, desc: data.Data, states: list[SDFGState]) -> list[ControlFlowBlock] | None:
        """The blocks of the innermost region holding every use of ``name``, one per group of uses, when there are two
        or more and each assigns the size before every use it holds and writes the container before every read: no
        value crosses from one block to the next. Else ``None``."""
        region = states[0].parent_graph
        # Climb to the innermost region, not a conditional, that holds every use
        while not isinstance(region, SDFG) and (
            isinstance(region, ConditionalBlock) or not all(s is region or contains(region, s) for s in states)
        ):
            region = region.parent_graph
        blocks: list[ControlFlowBlock] = []
        # Over the uses: the block of the region each lies in
        for state in states:
            block = state
            while block.parent_graph is not region:
                block = block.parent_graph
            if not any(block is other for other in blocks):
                blocks.append(block)
        if len(blocks) < 2:
            return None
        sizes = OrderedSet(str(symbol) for symbol in desc.free_symbols)
        # Over the blocks: each must assign the size, on an edge into it or inside it, ahead of every use it holds
        for block in blocks:
            inner = block.all_interstate_edges() if isinstance(block, AbstractControlFlowRegion) else []
            sized = [e.dst for e in inner if sizes & e.data.assignments.keys()]
            if any(sizes & e.data.assignments.keys() for e in region.in_edges(block)):
                sized.append(block)
            uses = [s for s in states if s is block or contains(block, s)]
            if not sized or not all(any(b is s or contains(b, s) or precedes(b, s) for b in sized) for s in uses):
                return None
            writes = [s for s in uses if any(s.in_degree(n) > 0 for n in s.data_nodes() if n.data == name)]
            reads = [s for s in uses if any(s.out_degree(n) > 0 for n in s.data_nodes() if n.data == name)]
            # Over the reads: each needs a write of this block ahead of it, else it reads the previous block's values
            if not all(any(precedes(w, r) for w in writes) for r in reads):
                return None
        return blocks

    @staticmethod
    def split(sdfg: SDFG, name: str, blocks: dict[str, ControlFlowBlock]) -> None:
        """Rename ``name`` in each of ``blocks`` to a copy of its own, named by the block's key."""
        desc = sdfg.arrays[name]
        # Each block gets its own copy of the descriptor
        for suffix, block in blocks.items():
            new_name = sdfg.add_datadesc(f"{name}_{suffix}", copy.deepcopy(desc), find_new_name=True)
            # Rename the container in every state of this block
            for state in [block] if isinstance(block, SDFGState) else block.states():
                # Access nodes take the new name
                for node in state.data_nodes():
                    if node.data == name:
                        node.data = new_name
                # Memlets take the new name
                for edge in state.edges():
                    if edge.data.data == name:
                        edge.data.data = new_name
        sdfg.remove_data(name, validate=False)
