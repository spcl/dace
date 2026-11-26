# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Give each branch its own copy of a transient that only the branches of one conditional use.

Fortran allocates a local array in each arm of an ``IF`` with the arm's own size (QE's ``fwfft_orbital_gamma``:
``ALLOCATE(psio(ngk(1), incr))`` in one arm, ``ALLOCATE(psio(ngk(1), brange))`` in the other). The frontend gives both
one container sized by symbols each arm assigns, and one allocation cannot follow two sizes. Each arm uses its own
allocation only, so each arm gets its own container: the allocation of each then follows its arm's size.
"""

import copy

from dace import data, dtypes, properties
from dace.sdfg import SDFG
from dace.sdfg.state import ConditionalBlock, ControlFlowBlock, SDFGState
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
        # Each SDFG owns its transients
        for sd in list(sdfg.all_sdfgs_recursive()):
            users: dict[str, list[SDFGState]] = {}
            # Record the states that use each container
            for state in sd.states():
                # Every access node is a use
                for node in state.data_nodes():
                    users.setdefault(node.data, []).append(state)
            read_on_edges = {name for edge in sd.all_interstate_edges() for name in edge.data.free_symbols}
            # Each candidate splits when its users span two or more branches of one conditional
            for name, states in users.items():
                desc = sd.arrays[name]
                if not self.candidate(desc) or name in read_on_edges:
                    continue
                found = enclosing_branch(states[0])
                # Widen to the conditional whose branches hold every user
                while found is not None:
                    arms = {branch_of(state, found[0]) for state in states}
                    if None not in arms:
                        break
                    found = enclosing_branch(found[0])
                if found is None or len(arms) < 2 or not self.sized_in(desc, found[0], arms):
                    continue
                self.split(sd, name, found[0], arms)
                split += 1
        return split or None

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
    def sized_in(desc: data.Data, conditional: ConditionalBlock, arms: set[int]) -> bool:
        """Whether a branch assigns a symbol the size of ``desc`` reads: a size fixed before the conditional keeps one
        allocation for all branches."""
        sizes = {str(symbol) for symbol in desc.free_symbols}
        return any(
            sizes & edge.data.assignments.keys()
            for index in arms
            for edge in conditional.branches[index][1].all_interstate_edges()
        )

    @staticmethod
    def split(sdfg: SDFG, name: str, conditional: ConditionalBlock, arms: set[int]) -> None:
        """Rename ``name`` in each used branch of ``conditional`` to a copy of its own."""
        desc = sdfg.arrays[name]
        # Each used branch gets its own copy of the descriptor
        for index in sorted(arms):
            branch = conditional.branches[index][1]
            new_name = sdfg.add_datadesc(f"{name}_branch{index}", copy.deepcopy(desc), find_new_name=True)
            # Rename the container in every state of this branch
            for state in branch.states():
                # Access nodes take the new name
                for node in state.data_nodes():
                    if node.data == name:
                        node.data = new_name
                # Memlets take the new name
                for edge in state.edges():
                    if edge.data.data == name:
                        edge.data.data = new_name
        sdfg.remove_data(name, validate=False)
