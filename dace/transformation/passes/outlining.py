# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Planning which control flow blocks code generation emits as functions of their own, by wrapping them in
``CodeGeneratorFunctionRegion``s.

Very large generated functions defeat compilers: past internal size limits they skip optimizations (e.g., gcc runs
no partial redundancy elimination in functions with 4000 or more basic blocks) or give up on alias analysis, so loops
stop vectorizing, and one huge function also serializes compilation. Dividing the code into functions (and
translation units) fixes both, as long as each cut separates code between which the compiler would not have found
an optimization anyway.

The planner works on the frozen, optimized SDFG, just before code generation:

1. A cost model estimates the size of each control flow block as the compiler sees it (basic blocks, statements).
2. Within each maximal chain of consecutive blocks, each possible cut gets a weight: the values live across it that
   a function boundary would force through memory (scalars, and symbols assigned before the cut and read after it).
3. A dynamic program divides each chain into functions within the size budget, minimizing the weight of the cuts
   plus a fixed cost per function. Blocks larger than the budget are not outlined themselves; the planner divides
   their contents instead (e.g., the body of a large loop).
4. The functions are spread over translation units, balancing their sizes, so they compile in parallel.
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set

from dace import SDFG, data, dtypes, properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutils
from dace.sdfg.state import (AbstractControlFlowRegion, CodeGeneratorFunctionRegion, ConditionalBlock, ControlFlowBlock,
                             ControlFlowRegion, LoopRegion, SDFGState)
from dace.transformation import helpers as xfh
from dace.transformation import pass_pipeline as ppl
from dace.transformation import transformation


@dataclass(frozen=True)
class BlockCost:
    """ The estimated size of a block of generated code. """

    basic_blocks: int  #: Basic blocks of the compiler's control flow graph
    statements: int  #: Statements (tasklet code lines, copies, library calls)

    def __add__(self, other: 'BlockCost') -> 'BlockCost':
        return BlockCost(self.basic_blocks + other.basic_blocks, self.statements + other.statements)


@dataclass
class FunctionPlan:
    """ A chain of consecutive blocks of one region, to be emitted as one function. """

    blocks: List[ControlFlowBlock]
    cost: BlockCost
    translation_unit: str = ''


_CONDITIONAL = re.compile(r'\bif\b|\?')


class CostModel:
    """
    Estimates the size of control flow blocks in the generated code. The estimates are coarse (e.g., a map dimension
    costs a loop header and latch); they only need to rank blocks and keep functions clear of compiler limits.
    """

    def __init__(self):
        self._cache: Dict[ControlFlowBlock, BlockCost] = {}

    def state(self, state: SDFGState) -> BlockCost:
        basic_blocks = 1
        statements = 0
        for node in state.nodes():
            if isinstance(node, nodes.MapEntry):
                basic_blocks += 2 * len(node.map.params)
            elif isinstance(node, nodes.Tasklet):
                code = node.code.as_string
                statements += max(1, code.count('\n') + 1)
                basic_blocks += 2 * len(_CONDITIONAL.findall(code))
            elif isinstance(node, nodes.NestedSDFG):
                inner = self.block(node.sdfg)
                basic_blocks += inner.basic_blocks
                statements += inner.statements
            elif isinstance(node, nodes.LibraryNode):
                basic_blocks += 2
                statements += 1
        for edge in state.edges():
            # A copy between containers is a loop nest of its own
            if isinstance(edge.src, nodes.AccessNode) and isinstance(edge.dst, nodes.AccessNode):
                statements += 1
                basic_blocks += 2 * len(edge.data.subset) if edge.data.subset is not None else 0
        return BlockCost(basic_blocks, statements)

    def block(self, block: ControlFlowBlock) -> BlockCost:
        """ The estimated size of a block, including everything nested in it. """
        cached = self._cache.get(block)
        if cached is not None:
            return cached
        if isinstance(block, SDFGState):
            cost = self.state(block)
        elif isinstance(block, CodeGeneratorFunctionRegion):
            cost = BlockCost(1, 1)  # Already a function of its own: a call
        elif isinstance(block, ConditionalBlock):
            cost = BlockCost(2 * len(block.branches), len(block.branches))
            for _, branch in block.branches:
                cost += self.block(branch)
        elif isinstance(block, AbstractControlFlowRegion):
            cost = BlockCost(3 if isinstance(block, LoopRegion) else 0, 0)
            for inner in block.nodes():
                cost += self.block(inner)
            cost += BlockCost(sum(1 for e in block.edges() if not e.data.is_unconditional()), 0)
        else:
            cost = BlockCost(1, 1)  # Break, continue, return
        self._cache[block] = cost
        return cost


def maximal_chains(region: ControlFlowRegion) -> List[List[ControlFlowBlock]]:
    """
    Divides the blocks of a region into maximal chains: sequences in which each block flows unconditionally into the
    next one, which nothing else enters.

    :param region: The region.
    :return: The chains, each in execution order. Every block of the region is in exactly one chain.
    """

    def links(block: ControlFlowBlock) -> Optional[ControlFlowBlock]:
        out_edges = region.out_edges(block)
        if len(out_edges) != 1 or not out_edges[0].data.is_unconditional():
            return None
        successor = out_edges[0].dst
        return successor if region.in_degree(successor) == 1 and successor is not block else None

    successors = {block: links(block) for block in region.nodes()}
    has_predecessor = {succ for succ in successors.values() if succ is not None}
    chains = []
    visited: Set[ControlFlowBlock] = set()
    for block in region.bfs_nodes(region.start_block) if region.number_of_nodes() > 0 else []:
        if block in has_predecessor or block in visited:
            continue
        chain = [block]
        while successors[chain[-1]] is not None and successors[chain[-1]] not in visited:
            chain.append(successors[chain[-1]])
        visited.update(chain)
        chains.append(chain)
    # Blocks only reachable through a cycle of links (no chain start), e.g., unreachable code
    for block in region.nodes():
        if block not in visited:
            visited.add(block)
            chains.append([block])
    return chains


class OutliningPlanner:
    """
    Plans which chains of blocks become functions. See the module documentation for the method.

    :param max_basic_blocks: The size budget of a function, in estimated basic blocks.
    :param max_statements: The size budget of a function, in estimated statements (None for no limit).
    :param min_basic_blocks: Chains smaller than this stay in their caller.
    :param scalar_weight: The weight of a scalar (or register array) live across a cut.
    :param symbol_weight: The weight of a symbol assigned before a cut and read after it.
    :param function_weight: The weight of each function (a cut must save more than this to be worth it).
    """

    def __init__(self,
                 max_basic_blocks: int,
                 min_basic_blocks: int = 0,
                 max_statements: Optional[int] = None,
                 scalar_weight: float = 4.0,
                 symbol_weight: float = 2.0,
                 function_weight: float = 1.0):
        self.max_basic_blocks = max_basic_blocks
        self.min_basic_blocks = min_basic_blocks
        self.max_statements = max_statements
        self.scalar_weight = scalar_weight
        self.symbol_weight = symbol_weight
        self.function_weight = function_weight
        self.costs = CostModel()
        self._used_symbols: Dict[ControlFlowBlock, Set[str]] = {}
        self._opaque: Dict[ControlFlowBlock, bool] = {}

    def _free_symbols(self, block: ControlFlowBlock) -> Set[str]:
        """ The symbols (and data names) a block reads before assigning them, cached. """
        syms = self._used_symbols.get(block)
        if syms is None:
            syms = set(block.used_symbols(all_symbols=True))
            self._used_symbols[block] = syms
        return syms

    ##########################################################################
    # Cut weights

    @staticmethod
    def _register_data(sdfg: SDFG, names: Set[str]) -> Set[str]:
        """ The containers among ``names`` that the compiler could keep in registers (scalars, register arrays). """
        result = set()
        for name in names:
            desc = sdfg.arrays.get(name)
            if desc is None or not desc.transient or isinstance(desc, data.View):
                continue
            if isinstance(desc, data.Scalar) or desc.storage == dtypes.StorageType.Register:
                result.add(name)
        return result

    def _accessed_data(self, block: ControlFlowBlock) -> Set[str]:
        sdfg = block.sdfg
        names = set()
        states = [block] if isinstance(
            block, SDFGState) else (block.all_states() if isinstance(block, AbstractControlFlowRegion) else [])
        for state in states:
            names |= {node.data.split('.')[0] for node in state.data_nodes()}
        names |= self._free_symbols(block) & sdfg.arrays.keys()
        return self._register_data(sdfg, names)

    @staticmethod
    def _assigned_symbols(block: ControlFlowBlock) -> Set[str]:
        assigned = set()
        if isinstance(block, AbstractControlFlowRegion):
            for edge in block.all_interstate_edges():
                assigned |= edge.data.assignments.keys()
            for inner in block.all_control_flow_blocks():
                if isinstance(inner, LoopRegion) and inner.loop_variable:
                    assigned.add(inner.loop_variable)
            if isinstance(block, LoopRegion) and block.loop_variable:
                assigned.add(block.loop_variable)
        return assigned

    def cut_weights(self, chain: List[ControlFlowBlock]) -> List[float]:
        """
        The weight of each cut of a chain: entry ``k`` is the cut between ``chain[k]`` and ``chain[k + 1]``.

        :param chain: The chain.
        :return: The ``len(chain) - 1`` cut weights.
        """
        if len(chain) < 2:
            return []
        region = chain[0].parent_graph
        edges = [region.out_edges(block)[0] for block in chain[:-1]]
        n = len(chain)
        accessed = [self._accessed_data(block) for block in chain]
        assigned = [self._assigned_symbols(block) for block in chain]
        # Symbols read before being assigned in each block (the free symbols), and on the edges of the chain
        read = [self._free_symbols(block) for block in chain]
        edge_read = [set(edge.data.free_symbols) for edge in edges] + [set()]
        edge_assigned = [set(edge.data.assignments.keys()) for edge in edges]

        # What the blocks (and edges) after each cut access and read: suffix unions
        data_after: List[Set[str]] = [set()] * n
        read_after: List[Set[str]] = [set()] * n
        for k in range(n - 2, -1, -1):
            data_after[k] = accessed[k + 1] | (data_after[k + 1] if k + 1 < n - 1 else set())
            read_after[k] = read[k + 1] | edge_read[k + 1] | (read_after[k + 1] if k + 1 < n - 1 else set())

        weights = []
        data_before: Set[str] = set()
        assigned_before: Set[str] = set()
        for k in range(n - 1):
            data_before |= accessed[k]
            assigned_before |= assigned[k]
            # An edge's assignments stay with the caller if the cut is at that edge, and its reads are passed in
            if k > 0:
                assigned_before |= edge_assigned[k - 1]
            weights.append(self.scalar_weight * len(data_before & data_after[k]) +
                           self.symbol_weight * len(assigned_before & read_after[k]))
        return weights

    ##########################################################################
    # Planning

    def fits(self, cost: BlockCost) -> bool:
        """ Whether code of the given size fits in one function. """
        return cost.basic_blocks <= self.max_basic_blocks and (self.max_statements is None
                                                               or cost.statements <= self.max_statements)

    def _divide_run(self, run: List[ControlFlowBlock], weights: List[float]) -> List[List[ControlFlowBlock]]:
        """
        Divides a run of consecutive blocks (each within the budget) into segments within the budget, minimizing the
        weights of the cuts plus the weight of each segment.

        :param run: The blocks.
        :param weights: The weights of the cuts between consecutive blocks of the run.
        :return: The segments, in order.
        """
        costs = [self.costs.block(block) for block in run]
        n = len(run)
        best = [0.0] + [float('inf')] * n  # best[j]: the cost of dividing the first j blocks
        start = [0] * (n + 1)
        for j in range(1, n + 1):
            size = BlockCost(0, 0)
            for i in range(j, 0, -1):  # Segment run[i - 1:j]
                size += costs[i - 1]
                if not self.fits(size):
                    break
                cost = best[i - 1] + self.function_weight + (weights[i - 2] if i > 1 else 0.0)
                if cost < best[j]:
                    best[j] = cost
                    start[j] = i - 1
        segments = []
        j = n
        while j > 0:
            segments.append(run[start[j]:j])
            j = start[j]
        return segments[::-1]

    def plan_region(self, region: ControlFlowRegion) -> List[FunctionPlan]:
        """
        Plans the functions within a region whose code is too large for one function.

        :param region: The region.
        :return: The planned functions, which do not overlap.
        """
        plans: List[FunctionPlan] = []
        for chain in maximal_chains(region):
            weights = self.cut_weights(chain)
            run: List[ControlFlowBlock] = []
            run_weights: List[float] = []

            def flush():
                for segment in (self._divide_run(run, run_weights) if run else []):
                    cost = BlockCost(0, 0)
                    for block in segment:
                        cost += self.costs.block(block)
                    if cost.basic_blocks >= self.min_basic_blocks:
                        plans.append(FunctionPlan(segment, cost))
                run.clear()
                run_weights.clear()

            for k, block in enumerate(chain):
                too_large = not self.fits(self.costs.block(block))
                opaque = self._calls_opaque_code(block)
                if too_large or opaque or isinstance(block,
                                                     CodeGeneratorFunctionRegion) or xfh.control_flow_exit(block):
                    # Not a part of any function: divide its contents instead if it is too large or calls opaque code
                    flush()
                    if too_large or opaque:
                        plans.extend(self._plan_inside(block))
                    continue
                if run:
                    run_weights.append(weights[k - 1])
                run.append(block)
            flush()
        return plans

    def _calls_opaque_code(self, block: ControlFlowBlock) -> bool:
        """
        Whether a block calls opaque code (e.g., a callback). Such blocks stay in their caller: a function containing
        one would have to reach persistent data through the state struct, which compilers treat as possibly aliased.
        """
        cached = self._opaque.get(block)
        if cached is None:
            states = [block] if isinstance(block, SDFGState) else (
                list(block.all_states()) if isinstance(block, AbstractControlFlowRegion) else [])
            cached = sdutils.calls_opaque_code(block.sdfg, states)
            self._opaque[block] = cached
        return cached

    def _plan_inside(self, block: ControlFlowBlock) -> List[FunctionPlan]:
        if isinstance(block, ConditionalBlock):
            return [plan for _, branch in block.branches for plan in self.plan_region(branch)]
        if isinstance(block, ControlFlowRegion) and not isinstance(block, CodeGeneratorFunctionRegion):
            return self.plan_region(block)
        return []  # A state too large for a function stays as it is

    def plan(self, sdfg: SDFG) -> List[FunctionPlan]:
        """
        Plans the functions of an SDFG. An SDFG within the budget is left as one function.

        :param sdfg: The SDFG.
        :return: The planned functions.
        """
        if self.fits(self.costs.block(sdfg)):
            return []
        return self.plan_region(sdfg)


def assign_translation_units(plans: List[FunctionPlan], units: int, min_unit_statements: int = 0) -> None:
    """
    Spreads functions over translation units, balancing the estimated statements per unit (largest first, each into
    the currently smallest unit).

    :param plans: The planned functions, whose ``translation_unit`` is set.
    :param units: The largest number of translation units.
    :param min_unit_statements: Fewer units are used if they would hold fewer statements than this on average: each
                                unit also parses the common preamble (the runtime headers and the state struct).
    """
    if min_unit_statements > 0:
        total = sum(plan.cost.statements for plan in plans)
        units = min(units, total // min_unit_statements)
    load = [0] * max(1, units)
    for plan in sorted(plans, key=lambda p: p.cost.statements + p.cost.basic_blocks, reverse=True):
        unit = min(range(len(load)), key=lambda u: load[u])
        load[unit] += plan.cost.statements + plan.cost.basic_blocks
        plan.translation_unit = f'unit_{unit}'


@properties.make_properties
@transformation.explicit_cf_compatible
class OutlineFunctions(ppl.Pass):
    """
    Wraps chains of control flow blocks into ``CodeGeneratorFunctionRegion``s, so that code generation divides a large
    program into functions and translation units. Runs on the final SDFG right before code generation, since
    simplification passes may inline the regions again.
    """

    CATEGORY: str = 'Code Generation'

    max_basic_blocks = properties.Property(dtype=int,
                                           default=2000,
                                           desc='Size budget of a function, in estimated basic blocks')
    min_basic_blocks = properties.Property(dtype=int,
                                           default=16,
                                           desc='Chains smaller than this (in estimated basic blocks) stay in their '
                                           'caller')
    max_statements = properties.Property(dtype=int,
                                         default=600,
                                         allow_none=True,
                                         desc='Size budget of a function, in estimated statements (None for no limit)')
    translation_units = properties.Property(dtype=int,
                                            default=8,
                                            desc='Largest number of translation units to spread the functions over')
    min_unit_statements = properties.Property(dtype=int,
                                              default=2500,
                                              desc='Fewer translation units are used if they would hold fewer '
                                              'statements than this on average, since each one parses the common '
                                              'preamble')
    function_placement = properties.EnumProperty(dtype=dtypes.FunctionPlacement,
                                                 default=dtypes.FunctionPlacement.SeparateUnit,
                                                 desc='Where to emit the functions')
    inlining = properties.EnumProperty(dtype=dtypes.FunctionInlining,
                                       default=dtypes.FunctionInlining.NoInline,
                                       desc='The inlining hint of the functions. Without a hint, compilers may inline '
                                       'a function called once back into its caller')

    def __init__(self,
                 max_basic_blocks: Optional[int] = None,
                 min_basic_blocks: Optional[int] = None,
                 max_statements: Optional[int] = None,
                 translation_units: Optional[int] = None,
                 min_unit_statements: Optional[int] = None,
                 function_placement: Optional[dtypes.FunctionPlacement] = None,
                 inlining: Optional[dtypes.FunctionInlining] = None):
        """ Creates the pass. Arguments left as None keep the defaults of the corresponding properties. """
        super().__init__()
        if max_basic_blocks is not None:
            self.max_basic_blocks = max_basic_blocks
        if min_basic_blocks is not None:
            self.min_basic_blocks = min_basic_blocks
        if max_statements is not None:
            self.max_statements = max_statements
        if translation_units is not None:
            self.translation_units = translation_units
        if min_unit_statements is not None:
            self.min_unit_statements = min_unit_statements
        if function_placement is not None:
            self.function_placement = function_placement
        if inlining is not None:
            self.inlining = inlining

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.CFG

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self) -> set:
        return set()

    def plan(self, sdfg: SDFG) -> List[FunctionPlan]:
        """
        Plans the functions of an SDFG without changing it.

        :param sdfg: The SDFG.
        :return: The planned functions, with translation units assigned if placed in separate units.
        """
        planner = OutliningPlanner(self.max_basic_blocks, self.min_basic_blocks, self.max_statements)
        plans = planner.plan(sdfg)
        if self.function_placement == dtypes.FunctionPlacement.SeparateUnit:
            assign_translation_units(plans, self.translation_units, self.min_unit_statements)
        return plans

    def apply_pass(self, sdfg: SDFG, pipeline_results: Dict[str, Any]) -> Optional[int]:
        """
        :return: The number of functions outlined, or None if none was.
        """
        plans = self.plan(sdfg)
        for i, plan in enumerate(plans):
            xfh.wrap_in_function_region(plan.blocks,
                                        f'outlined_{i}',
                                        function_placement=self.function_placement,
                                        translation_unit=plan.translation_unit,
                                        inlining=self.inlining,
                                        reset_cfg_list=False)
        if plans:
            sdfg.reset_cfg_list()
        return len(plans) or None
