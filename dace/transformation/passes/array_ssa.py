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
from dace.sdfg import InterstateEdge
from dace.memlet import Memlet
from dace.subsets import Range


@dataclass(unsafe_hash=True)
@properties.make_properties
@transformation.explicit_cf_compatible
class StrictArraySSA(ppl.Pass):
    """
    Performs SSA on arrays in a strict manner, i.e. only considers full writes to arrays.
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

        # Also maintain a table of required phi nodes
        phi_table = {cfgb: set() for cfgb in all_cfgb.keys()}

        # Global counter for the new names
        self._counter = 0
        self._cnt_to_name = {}

        # Perform a backwards fixed-point iteration to build tables
        changed = True
        while changed:
            changed = False

            # Update incoming table
            for cfgb, parent in all_cfgb.items():
                new_in_table = self._get_in_table(
                    cfgb, parent, in_table, out_table, phi_table
                )

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

        # Create a mapping from the counter to the new name
        cnt_to_new_name = {}
        for c, n in self._cnt_to_name.items():
            cnt_to_new_name[c] = self._copy_array(sdfg, n, str(c))

        # Create replacement mappings
        in_array_names = {cfgb: {} for cfgb in all_cfgb.keys()}
        out_array_names = {cfgb: {} for cfgb in all_cfgb.keys()}
        for cfgb, parent in all_cfgb.items():
            for orig, cnt in in_table[cfgb].items():
                in_array_names[cfgb][orig] = cnt_to_new_name[cnt]

            for orig, cnt in out_table[cfgb].items():
                if cnt is not None:
                    out_array_names[cfgb][orig] = cnt_to_new_name[cnt]

        # Perform replacement of the arrays
        for cfgb, parent in all_cfgb.items():
            self._replace(
                sdfg, cfgb, parent, in_array_names, out_array_names, phi_table
            )
        return set()

    # Given a CFGB, builds the incoming table
    def _get_in_table(
        self,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_table: Dict[ControlFlowBlock, Dict[str, int]],
        out_table: Dict[ControlFlowBlock, Dict[str, int]],
        phi_table: Dict[ControlFlowBlock, Set[str]],
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
            new_in_table = in_table[parent]

            # For LoopRegions, combine with sink nodes
            if isinstance(parent, LoopRegion):
                for n in parent.sink_nodes():
                    assert isinstance(n, ControlFlowBlock)
                    new_in_table = self._combine_entries(new_in_table, out_table[n])

        # If we have disagreeing entries, we need to add a phi node
        for orig, cnt in new_in_table.items():
            if cnt is not None:
                continue

            # Already added a phi node?
            if orig in phi_table[cfgb]:
                new_in_table[orig] = in_table[cfgb][orig]
                continue

            # Add a phi node
            phi_table[cfgb].add(orig)
            new_in_table[orig] = self._counter
            self._cnt_to_name[self._counter] = orig
            self._counter += 1

        # There should be no None entries in the table
        for orig, cnt in new_in_table.items():
            assert cnt is not None, f"Entry {orig} in {cfgb} is None"
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
            # Any array that is completely written can be renamed
            new_out_table = copy.deepcopy(in_table[cfgb])
            for arr in self._get_overwritten_arrays(sdfg, cfgb):
                if arr not in out_table[cfgb]:  # not already renamed
                    new_out_table[arr] = self._counter
                    self._cnt_to_name[self._counter] = arr
                    self._counter += 1
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
    def _overwritten(self, sdfg: SDFG, state: SDFGState, node: AccessNode) -> bool:
        # Only consider end-of-chain writes (simplifies replacement)
        if state.out_degree(node) != 0:
            return False

        # Only consider scalars
        if not isinstance(sdfg.arrays[node.data], dt.Scalar):
            return False

        if not node.data == "maxvcfl":
            return False

        # Turn array into a range
        node_range = Range.from_array(sdfg.arrays[node.data])

        # Check coverage
        for edge in state.in_edges(node):
            if edge.data.dst_subset.covers_precise(node_range):
                return True
        return False

    # Given a state, returns all arrays that are completely written
    def _get_overwritten_arrays(self, sdfg: SDFG, state: SDFGState) -> Set[str]:
        # Cache
        if not hasattr(self, "_goa_cache"):
            self._goa_cache = {}
        if state in self._goa_cache:
            return self._goa_cache[state]
        self._goa_cache[state] = set()

        # Compute result
        overwritten = set()
        for node in state.data_nodes():
            assert isinstance(node, AccessNode)
            # Non-transients cannot be renamed
            if sdfg.arrays[node.data].transient != True:
                continue

            if self._overwritten(sdfg, state, node):
                overwritten.add(node.data)

        # Cache the result
        self._goa_cache[state] = overwritten
        return overwritten

    # Given a CFGB, replaces the arrays with the new names
    def _replace(
        self,
        sdfg: SDFG,
        cfgb: ControlFlowBlock,
        parent: ControlFlowRegion,
        in_array_names: Dict[ControlFlowBlock, Dict[str, str]],
        out_array_names: Dict[ControlFlowBlock, Dict[str, str]],
        phi_table: Dict[ControlFlowBlock, Set[str]],
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
                    self._overwritten(sdfg, cfgb, node)
                    and node.data in out_array_names[cfgb]
                ):
                    node.data = out_array_names[cfgb][node.data]

            # Replace any other access nodes with the in_table
            cfgb.replace_dict(in_array_names[cfgb])
        else:
            # Don't replace, as the nested CFBGs should inherit the symbols from their parent
            pass

        # Also replace all symbols in the outgoing edges with their values
        for edge in parent.out_edges(cfgb):
            edge.data.replace_dict(out_array_names[cfgb], replace_keys=False)

        # Insert any necessary phi nodes
        for orig in phi_table[cfgb]:
            # For each predecessor, add a state in between
            for pred in parent.predecessors(cfgb):
                if isinstance(pred, ConditionalBlock):
                    # We need to add a phi node for each branch
                    for b in pred.sub_regions():
                        self._add_phi_node(
                            cfgb,
                            b,
                            orig,
                            in_array_names,
                            out_array_names,
                        )

                else:
                    self._add_phi_node(
                        cfgb,
                        pred,
                        orig,
                        in_array_names,
                        out_array_names,
                    )

    # Creates a copy of an array with a new name
    def _copy_array(self, sdfg: SDFG, array_name: str, cnt: str) -> str:
        array = sdfg.arrays[array_name]
        if isinstance(array, dt.Scalar):
            new_name, _ = sdfg.add_scalar(
                f"{array_name}_SSA{cnt}",
                dtype=array.dtype,
                storage=array.storage,
                transient=array.transient,
                lifetime=array.lifetime,
                debuginfo=array.debuginfo,
                find_new_name=True,
            )
        else:
            new_name, _ = sdfg.add_array(
                f"{array_name}_SSA{cnt}",
                shape=array.shape,
                dtype=array.dtype,
                storage=array.storage,
                location=array.location,
                transient=array.transient,
                strides=array.strides,
                offset=array.offset,
                lifetime=array.lifetime,
                debuginfo=array.debuginfo,
                allow_conflicts=array.allow_conflicts,
                total_size=array.total_size,
                find_new_name=True,
                alignment=array.alignment,
                may_alias=array.may_alias,
            )
        return new_name

    # Given two CFGBs, adds a phi node between them
    def _add_phi_node(
        self,
        cfgb: ControlFlowBlock,
        pred: ControlFlowBlock,
        orig: str,
        in_array_names: Dict[ControlFlowBlock, Dict[str, str]],
        out_array_names: Dict[ControlFlowBlock, Dict[str, str]],
    ):
        outer_cfg = cfgb.parent_graph
        assert len(outer_cfg.edges_between(pred, cfgb)) == 1

        # Add inbetween state
        old_edge = outer_cfg.edges_between(pred, cfgb)[0]
        between_state = outer_cfg.add_state("PHI State")
        outer_cfg.add_edge(pred, between_state, copy.deepcopy(old_edge.data))
        outer_cfg.add_edge(between_state, cfgb, InterstateEdge())
        outer_cfg.remove_edge(old_edge)

        # Add new state to the tables
        in_array_names[between_state] = copy.deepcopy(out_array_names[pred])
        out_array_names[between_state] = copy.deepcopy(in_array_names[cfgb])

        # Find the names of the arrays
        in_name = in_array_names[between_state][orig]
        out_name = out_array_names[between_state][orig]

        # Add copy nodes
        in_node = between_state.add_read(in_name)
        out_node = between_state.add_write(out_name)
        between_state.add_edge(in_node, out_node, Memlet(out_name))
