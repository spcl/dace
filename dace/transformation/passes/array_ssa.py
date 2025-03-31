# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

import copy
from dataclasses import dataclass
from dace.sdfg.state import (
    ControlFlowBlock,
    ControlFlowRegion,
    ConditionalBlock,
    LoopRegion,
)
from dace.transformation import pass_pipeline as ppl, transformation
from dace import SDFG, properties, SDFGState
from typing import Dict, Set, Optional
from dace import data as dt
from dace.sdfg.nodes import AccessNode
from dace.subsets import Range
from dace.sdfg import utils as sdutils
import re
from functools import lru_cache


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class StrictArraySSA(ppl.Pass):
    """
    Performs SSA on arrays in a strict manner, i.e. only considers full writes to arrays and does not introduce phi nodes.
    Patterns such as:
    write A -> read A -> write A -> read A
    are split up to:
    write A -> read A -> write B -> read B.
    If the second write fully overwrites the first write's data.
    """

    CATEGORY: str = "Optimization Preparation"

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Edges | ppl.Modifies.Nodes

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        # If anything was modified, reapply
        return modified != ppl.Modifies.Nothing

    def apply_pass(self, sdfg: SDFG, _) -> Optional[Set[str]]:
        # For the fixed-point analysis we maintain a table of mappings from original array name to the new name, represented by a counter.

        # Get all CFG blocks present in the SDFG
        all_cfgb = dict()
        for node, parent in sdfg.all_nodes_recursive():
            if isinstance(node, ControlFlowBlock) and parent.sdfg == sdfg:
                all_cfgb[node] = parent

        # For each CFG Block maintain a dict of incoming and outgoing tables
        in_table = {cfgb: {} for cfgb in all_cfgb.keys()}
        out_table = {cfgb: {} for cfgb in all_cfgb.keys()}

        # Perform a forward fixed-point iteration to build tables
        changed = True
        while changed:
            changed = False

            # Update incoming table
            for cfgb, parent in all_cfgb.items():
                new_in_table = self._get_in_table(cfgb, parent, in_table, out_table)

                # Check if the incoming table have changed
                if new_in_table != in_table[cfgb]:
                    changed = True
                    in_table[cfgb] = new_in_table

            # Update outgoing table
            for cfgb, parent in all_cfgb.items():
                new_out_table = self._get_out_table(
                    sdfg, cfgb, parent, in_table, out_table
                )

                # Check if the outgoing table have changed
                if new_out_table != out_table[cfgb]:
                    changed = True
                    out_table[cfgb] = new_out_table

        # Create replacement mappings
        in_array_names = {cfgb: {} for cfgb in all_cfgb.keys()}
        out_array_names = {cfgb: {} for cfgb in all_cfgb.keys()}
        id_to_new_name = {}

        for cfgb, parent in all_cfgb.items():
            for orig, cnt in in_table[cfgb].items():
                assert cnt is not None
                self._copy_array(sdfg, orig, cnt, id_to_new_name)
                in_array_names[cfgb][orig] = id_to_new_name[orig][cnt]

            for orig, cnt in out_table[cfgb].items():
                assert cnt is not None
                self._copy_array(sdfg, orig, cnt, id_to_new_name)
                out_array_names[cfgb][orig] = id_to_new_name[orig][cnt]

        # Perform replacement of the arrays
        for cfgb, parent in all_cfgb.items():
            self._replace(sdfg, cfgb, parent, in_array_names, out_array_names)
        return set()

    # Given a CFGB, builds the incoming table
    def _get_in_table(
        self,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_table: Dict[ControlFlowBlock, Dict[str, int]],
        out_table: Dict[ControlFlowBlock, Dict[str, int]],
    ) -> Dict[str, int]:
        new_in_table = {}

        # Combine the outgoing tables of all incoming edges
        for i, pred in enumerate(parent.predecessors(cfgb)):
            sym_table = copy.deepcopy(out_table[pred])
            if i == 0:
                new_in_table = sym_table
            else:
                new_in_table = self._combine_entries(new_in_table, sym_table)

        # Nested starting CFGBs should inherit from their parent
        # Ignore SDFGs as nested SDFGs have inputs and outputs
        if (parent.start_block == cfgb and not isinstance(parent, SDFG)) or (
            isinstance(parent, ConditionalBlock) and cfgb in parent.sub_regions()
        ):
            assert new_in_table == {}
            new_in_table = copy.deepcopy(in_table[parent])

        # If we have disagreeing entries, we would need to add a phi node
        for orig, cnt in new_in_table.items():
            assert cnt is not None
        return new_in_table

    # Given a CFGB, builds the outgoing table
    def _get_out_table(
        self,
        sdfg: SDFG,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_table: Dict[ControlFlowBlock, Dict[str, int]],
        out_table: Dict[ControlFlowBlock, Dict[str, int]],
    ) -> Dict[str, int]:
        if isinstance(cfgb, LoopRegion):
            # Combine all outgoing tables of the sink nodes and the incoming table of the loop (as the loop might not be taken)
            new_out_table = copy.deepcopy(in_table[cfgb])
            for n in cfgb.sink_nodes():
                assert isinstance(n, ControlFlowBlock)
                new_out_table = self._combine_entries(new_out_table, out_table[n])
            return new_out_table

        elif isinstance(cfgb, ConditionalBlock):
            # Combine all outgoing tables of the branches
            new_out_table = copy.deepcopy(out_table[cfgb.sub_regions()[0]])
            for b in cfgb.sub_regions():
                new_out_table = self._combine_entries(new_out_table, out_table[b])

            # If no else branch is present, also combine the incoming table (implicit else branch)
            has_non_conds = any([c is None for c, _ in cfgb.branches])
            if not has_non_conds:
                new_out_table = self._combine_entries(new_out_table, in_table[cfgb])

            return new_out_table

        elif isinstance(cfgb, SDFGState):
            # If we are in a loop or conditional block, only rename if not already renamed
            in_loop_or_cond = False
            parent_cpy = parent
            while parent_cpy is not None and not isinstance(parent_cpy, SDFG):
                if isinstance(parent_cpy, LoopRegion) or isinstance(
                    parent_cpy, ConditionalBlock
                ):
                    in_loop_or_cond = True
                    break
                parent_cpy = parent_cpy.parent_graph

            # Any array that is completely written can be renamed
            new_out_table = copy.deepcopy(in_table[cfgb])
            for arr in StrictArraySSA._get_overwritten_arrays(sdfg, cfgb):
                if in_loop_or_cond and arr in in_table[cfgb]:
                    continue

                if arr not in out_table[cfgb]:  # not already renamed
                    in_count = 0 if arr not in in_table[cfgb] else in_table[cfgb][arr]
                    new_out_table[arr] = in_count + 1

                else:
                    new_out_table[arr] = out_table[cfgb][arr]
            return new_out_table

        else:
            # Combine all sinks
            sink_nodes = [
                n
                for n in cfgb.nodes()
                if cfgb.out_degree(n) == 0 and isinstance(n, ControlFlowBlock)
            ]
            if len(sink_nodes) == 0:
                return in_table[cfgb]

            new_out_table = copy.deepcopy(out_table[sink_nodes[0]])
            for n in sink_nodes:
                new_out_table = self._combine_entries(new_out_table, out_table[n])
            return new_out_table

    # Combines two table entries
    def _combine_entries(
        self, e1: Dict[str, int], e2: Dict[str, int]
    ) -> Dict[str, int]:
        out = {}
        for orig, cnt in e1.items():
            if orig not in e2:
                out[orig] = cnt
            elif cnt != e2[orig]:
                out[orig] = None
            else:
                out[orig] = cnt
        return out

    # Checks if an access node is completely written
    @staticmethod  # To make it cacheable
    @lru_cache(maxsize=None)
    def _is_overwritten(sdfg: SDFG, state: SDFGState, node: AccessNode) -> bool:
        # NOTE: Only consider scalars and views to avoid increasing the memory footprint too much
        if not isinstance(sdfg.arrays[node.data], dt.Scalar) and not isinstance(
            sdfg.arrays[node.data], dt.View
        ):
            return False

        # Turn array into a range
        node_range = Range.from_array(sdfg.arrays[node.data])

        # Check coverage
        for edge in state.in_edges(node):
            if (
                edge.data
                and edge.data.dst_subset
                and edge.data.dst_subset.covers_precise(node_range)
            ):
                return True
        return False

    # Given a state, returns all arrays that are completely written
    @staticmethod  # To make it cacheable
    @lru_cache(maxsize=None)
    def _get_overwritten_arrays(sdfg: SDFG, state: SDFGState) -> Set[str]:
        overwritten = set()
        for node in state.data_nodes():
            assert isinstance(node, AccessNode)
            # Non-transients cannot be renamed
            if sdfg.arrays[node.data].transient != True:
                continue

            if StrictArraySSA._is_overwritten(sdfg, state, node):
                overwritten.add(node.data)
        return overwritten

    # Given a CFGB, replaces the arrays with the new names
    def _replace(
        self,
        sdfg: SDFG,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_array_names: Dict[ControlFlowBlock, Dict[str, str]],
        out_array_names: Dict[ControlFlowBlock, Dict[str, str]],
    ) -> None:
        # Replace depending on type
        if isinstance(cfgb, LoopRegion):
            # Replace meta accesses
            cfgb.replace_meta_accesses(in_array_names[cfgb])
        elif isinstance(cfgb, ConditionalBlock):
            # Replace meta accesses
            cfgb.replace_meta_accesses(in_array_names[cfgb])
        elif isinstance(cfgb, SDFGState):
            # Replace complete writes with the out_array_names
            for node in cfgb.data_nodes():
                assert isinstance(node, AccessNode)
                if (
                    StrictArraySSA._is_overwritten(sdfg, cfgb, node)
                    and node.data in out_array_names[cfgb]
                ):
                    old_data = node.data
                    new_data = out_array_names[cfgb][node.data]
                    repl_pattern = r"\b" + re.escape(old_data) + r"\b"

                    for dfs_node in sdutils.dfs_topological_sort(cfgb, node):
                        if isinstance(dfs_node, AccessNode):
                            dfs_node.data = re.sub(
                                repl_pattern, new_data, dfs_node.data
                            )

                        for i_edge in cfgb.in_edges(dfs_node) + cfgb.out_edges(
                            dfs_node
                        ):
                            i_edge.data.replace({old_data: new_data})
                            i_edge.data.data = re.sub(
                                repl_pattern, new_data, i_edge.data.data
                            )

            # Replace any other access nodes with the in_table
            cfgb.replace_dict(in_array_names[cfgb])
        else:
            # Don't replace, as the nested CFBGs should inherit the symbols from their parent
            pass

        # Also replace all symbols in the outgoing edges with their values
        for edge in parent.out_edges(cfgb):
            edge.data.replace_dict(out_array_names[cfgb], replace_keys=False)

    # Creates a copy of an array with a new name if not already present
    def _copy_array(
        self,
        sdfg: SDFG,
        array_name: str,
        cnt: str,
        id_to_new_name: Dict[str, Dict[str, str]],
    ) -> None:
        # Check if the array is already present
        if array_name in id_to_new_name:
            if cnt in id_to_new_name[array_name]:
                return
        else:
            id_to_new_name[array_name] = {}

        array_desc = copy.deepcopy(sdfg.arrays[array_name])
        new_name = sdfg.add_datadesc(
            f"{array_name}_SSA{cnt}", array_desc, find_new_name=True
        )
        id_to_new_name[array_name][cnt] = new_name
