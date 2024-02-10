# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.

import ast
from collections import defaultdict

import sympy
from dace.memlet import Memlet
from dace.sdfg.state import LoopRegion
from dace.subsets import Subset, SubsetUnion
from dace.transformation import pass_pipeline as ppl
from dace import SDFG, properties, symbolic
from typing import Dict, Optional, Set, Any, Tuple

from dace.transformation.pass_pipeline import Pass
from dace.transformation.passes.control_flow_region_analysis import CFGDataDependence


@properties.make_properties
class LoopCarryDependencyAnalysis(ppl.Pass):
    """
    Analyze the data dependencies between loop iterations for loop regions.
    """

    CATEGORY: str = 'Analysis'

    _non_analyzable_loops: Set[LoopRegion]

    def __init__(self):
        self._non_analyzable_loops = set()
        super().__init__()

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.CFG

    def depends_on(self) -> Set[type[Pass] | Pass]:
        return {CFGDataDependence}

    def _intersects(self, loop: LoopRegion, write_subset: Subset, read_subset: Subset, update: sympy.Basic) -> bool:
        """
        Check if a write subset intersects a read subset after being offset by the loop stride. The offset is performed
        based on the symbolic loop update assignment expression.
        """
        offset = update - symbolic.symbol(loop.loop_variable)
        offset_list = []
        for i in range(write_subset.dims()):
            if loop.loop_variable in write_subset.get_free_symbols_by_indices([i]):
                offset_list.append(offset)
            else:
                offset_list.append(0)
        offset_write = write_subset.offset_new(offset_list, True)
        return offset_write.intersects(read_subset)

    def apply_pass(self, top_sdfg: SDFG,
                   pipeline_results: Dict[str, Any]) -> Dict[int, Dict[LoopRegion, Dict[Memlet, Set[Memlet]]]]:
        """
        :return: For each SDFG, a dictionary mapping loop regions to a dictionary that resolves reads to writes in the
                 same loop, from which they may carry a RAW dependency.
        """
        results = defaultdict(lambda: defaultdict(dict))

        cfg_dependency_dict: Dict[int, Tuple[Dict[str, Set[Memlet]], Dict[str, Set[Memlet]]]] = pipeline_results[
            CFGDataDependence.__name__
        ]
        for cfg in top_sdfg.all_control_flow_regions(recursive=True):
            if isinstance(cfg, LoopRegion):
                loop_inputs, loop_outputs = cfg_dependency_dict[cfg.cfg_id]
                update_assignment = None
                loop_dependencies: Dict[Memlet, Set[Memlet]] = dict()

                for data in loop_inputs:
                    if not data in loop_outputs:
                        continue

                    for input in loop_inputs[data]:
                        read_subset = input.src_subset
                        dep_candidates: Set[Memlet] = set()
                        if cfg.loop_variable and cfg.loop_variable in input.free_symbols:
                            # If the iteration variable is involved in an access, we need to first offset it by the loop
                            # stride and then check for an overlap/intersection. If one is found after offsetting, there
                            # is a RAW loop carry dependency.
                            for output in loop_outputs[data]:
                                # Get and cache the update assignment for the loop.
                                if update_assignment is None and not cfg in self._non_analyzable_loops:
                                    update_assignment = _get_update_assignment(cfg)
                                    if update_assignment is None:
                                        self._non_analyzable_loops(cfg)

                                if isinstance(output.subset, SubsetUnion):
                                    if any([self._intersects(cfg, s, read_subset, update_assignment)
                                            for s in output.subset.subset_list]):
                                        dep_candidates.add(output)
                                elif self._intersects(cfg, output.subset, read_subset, update_assignment):
                                    dep_candidates.add(output)
                        else:
                            # Check for basic overlaps/intersections in RAW loop carry dependencies, when there is no
                            # iteration variable involved.
                            for output in loop_outputs[data]:
                                if isinstance(output.subset, SubsetUnion):
                                    if any([s.intersects(read_subset) for s in output.subset.subset_list]):
                                        dep_candidates.add(output)
                                elif output.subset.intersects(read_subset):
                                    dep_candidates.add(output)
                        loop_dependencies[input] = dep_candidates
                results[cfg.sdfg.cfg_id][cfg] = loop_dependencies

        return results


class FindAssignment(ast.NodeVisitor):

    assignments: Dict[str, str]
    multiple: bool

    def __init__(self):
        self.assignments = {}
        self.multiple = False

    def visit_Assign(self, node: ast.Assign) -> Any:
        for tgt in node.targets:
            if isinstance(tgt, ast.Name):
                if tgt.id in self.assignments:
                    self.multiple = True
                self.assignments[tgt.id] = ast.unparse(node.value)
        return self.generic_visit(node)


def _get_update_assignment(loop: LoopRegion) -> Optional[sympy.Basic]:
    """
    Parse a loop region's update statement to identify the exact update assignment expression.
    """
    update_stmt = loop.update_statement
    if update_stmt is None:
        return None

    update_codes_list = update_stmt.code if isinstance(update_stmt.code, list) else [update_stmt.code]
    assignments: Dict[str, str] = {}
    for code in update_codes_list:
        visitor = FindAssignment()
        visitor.visit(code)
        if visitor.multiple:
            return None
        for assign in visitor.assignments:
            if assign in assignments:
                return None
            assignments[assign] = visitor.assignments[assign]

    if loop.loop_variable in assignments:
        return symbolic.pystr_to_symbolic(assignments[loop.loop_variable])

    return None
