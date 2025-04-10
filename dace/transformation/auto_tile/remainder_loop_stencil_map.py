# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


from typing import List, Set, Union
from dace.data import Property
from dace.sdfg import SDFG, SDFGState
from dace.properties import CodeBlock, make_properties
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.sdfg import ConditionalBlock, ControlFlowBlock
from dace.sdfg.state import ControlFlowRegion
from dace.transformation import transformation
from dace import dtypes
from dace import symbolic
import dace
import copy
from dace.subsets import Range
from dace.sdfg.analysis.cutout import SDFGCutout


remainder_loop_stencil_map_counter = 0

@make_properties
class RemainderLoopStencilMap(transformation.SingleStateTransformation):
    inner_work_map_entry = transformation.PatternNode(nodes.MapEntry)
    tblock_type = Property(dtype=dtypes.ScheduleType, default=dtypes.ScheduleType.GPU_ThreadBlock, allow_none=False)


    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.inner_work_map_entry)]

    def can_be_applied(self, graph, expr_index, sdfg, permissive=False):
        return True

    def update_names():
        pass

    def has_remainder(self, numerator, denominator):
        if (isinstance(numerator, int) or numerator.is_integer) and \
                (isinstance(denominator, int) or denominator.is_integer):
            return numerator % denominator != 0
        else:
            return True


    def _copy_range(self, state: SDFGState, map_entry: nodes.MapEntry):
        # Deep copy inner body_nodes
        inner_body_nodes = []
        inner_body_edges = []
        node_map = dict()
        _map_entry = None
        _map_exit = None
        for node in list(state.all_nodes_between(map_entry, state.exit_node(map_entry))) + [map_entry, state.exit_node(map_entry)]:
            n2 = copy.deepcopy(node)
            if node == map_entry:
                _map_entry = (node, n2)
            if node == state.exit_node(map_entry):
                _map_exit = (node, n2)
            node_map[node] = n2
            inner_body_nodes.append(n2)
        for edge in state.all_edges(*state.all_nodes_between(map_entry, state.exit_node(map_entry))):
            e2 = (node_map[edge.src], edge.src_conn, node_map[edge.dst], edge.dst_conn, copy.deepcopy(edge.data))
            inner_body_edges.append(e2)

        return inner_body_nodes, inner_body_edges, _map_entry, _map_exit

    def apply(self, state: SDFGState, sdfg: SDFG):
        global remainder_loop_stencil_map_counter
        # Workflow:
        # We have a map with a coarsened map inside
        # create two nested SDFGs with 1- completely unrolled inner map, 2- one with bound checks

        # db, de, ds = device_map_range
        # tb, te, ts = thread_block_map_range
        # ib, ie, is = inner_work_map_range

        # 1- completely unrolled one is (ib, ie, is)
        # 2- one with bound checks is (ib, Min(ie-1, de-1)+1, is)
        # 3- the condition for 1 is ((ie - ib) <= (de + 1 - ie))

        inner_work_map_entry = self.inner_work_map_entry
        map_entry = self.inner_work_map_entry
        dev_entry = None
        while map_entry:
            dev_entry = map_entry
            if dev_entry.map.schedule == dtypes.ScheduleType.GPU_Device:
                break
            map_entry = state.entry_node(map_entry)
        assert (dev_entry.map.schedule == dtypes.ScheduleType.GPU_Device)
        map_entry = inner_work_map_entry

        # Deepcopy nodes and edges twice for each kernels
        inedges, outedges = state.in_edges(map_entry), state.out_edges(state.exit_node(map_entry))
        inner_body_nodes, inner_body_edges, map_entry1, map_exit1 = self._copy_range(state, map_entry)
        inner_body_nodes2, inner_body_edges2, map_entry2, map_exit2 = self._copy_range(state, map_entry)

        nodes_to_remove = list(state.all_nodes_between(map_entry, state.exit_node(map_entry))) + [map_entry, state.exit_node(map_entry)]

        # Collect symbols and data we need in the inner SDFG
        symbols_defined_at_entry = state.symbols_defined_at(map_entry)
        inputs  = [ie.data.data for ie in state.in_edges(map_entry) if ie.data is not None and ie.data.data is not None]
        outputs = [oe.data.data for oe in state.in_edges(state.exit_node(map_entry)) if oe.data is not None and oe.data.data is not None]
        used_arrs = set(inputs).union(set(outputs))
        sym_map = dict()
        sym_and_type_map = dict()
        for k in symbols_defined_at_entry.keys():
            sym_map[k] = k
            sym_and_type_map[k] = symbols_defined_at_entry[k].dtype if symbols_defined_at_entry[k] is not None else dace.int64
        # Need end range of device map for bound checks
        # Need to save the type for nested SDFG (add symbol)
        sym_and_types = dict()
        for (db, _de, ds) in dev_entry.map.range:
            for de in [db, _de, ds]:
                if isinstance(de, dace.symbolic.symbol):
                    sym_map[str(de)] = str(de)
                    sym_and_types[str(de)] = de.dtype
                elif hasattr(de, 'free_symbols'):
                    for fe in de.free_symbols:
                        sym_map[str(fe)] = str(fe)
                        sym_and_types[str(fe)] = fe.dtype

        # Generated conditions
        # db, de, ds = device_map_range
        # tb, te, ts = thread_block_map_range
        # ib, ie, is = inner_work_map_range

        # 1- completely unrolled one is (ib, ie, is)
        # 2- one with bound checks is (ib, Min(ie-1, de-1)+1, is)
        # 3- the condition for 1 is ((ie - ib) <= (de + 1 - ie))
        inner_cond = []
        for (db, de, ds), (ib, ie, _is) in zip(dev_entry.map.range, inner_work_map_entry.map.range):
            inner_cond.append(f"(({ie} + 1 - {ib}) <= ({de} + 1 - {ie}))")
        inner_cond = " and ".join(inner_cond)
        remainder_cond = f"not ({inner_cond})"

        # Inner SDFG, add data from descriptors
        inner_sdfg = SDFG(f"tiled_kernel_body_{remainder_loop_stencil_map_counter}")
        for name in used_arrs:
            inner_sdfg.add_datadesc(name, copy.deepcopy(sdfg.arrays[name]))

        # Add nSDFG node and symbols
        nsdfg = nodes.NestedSDFG(
            label=f"tiled_kernel_{remainder_loop_stencil_map_counter}",
            sdfg=inner_sdfg,
            inputs=inputs,
            outputs=outputs,
            symbol_mapping=sym_map,
            schedule=dace.dtypes.ScheduleType.Sequential
        )
        for sym, symtype in sym_and_types.items():
            inner_sdfg.add_symbol(sym, symtype)
        for sym, symtype in sym_and_type_map.items():
            inner_sdfg.add_symbol(sym, symtype)

        # Add the CFGs for inner and remainder kernel body
        if_block = ConditionalBlock(label=f"cond_tiled_kernel_body_{remainder_loop_stencil_map_counter}", sdfg=inner_sdfg, parent=inner_sdfg)
        inner_cfg = ControlFlowRegion(label=f"inner_body_{remainder_loop_stencil_map_counter}", sdfg=inner_sdfg, parent=if_block)
        remainder_cfg = ControlFlowRegion(label=f"remainder_body_{remainder_loop_stencil_map_counter}", sdfg=inner_sdfg, parent=if_block)
        inner_state = inner_cfg.add_state(f"inner_body_state_{remainder_loop_stencil_map_counter}")
        remainder_state = remainder_cfg.add_state(f"remainder_body_state_{remainder_loop_stencil_map_counter}")
        if_block.add_branch(CodeBlock(inner_cond), inner_cfg)
        if_block.add_branch(CodeBlock(remainder_cond), remainder_cfg)
        inner_sdfg.add_node(if_block, is_start_block=True)

        # Add nSDFG and connections in and out
        state.add_node(nsdfg)
        for ie in state.in_edges(map_entry):
            if ie.data.data is not None:
                state.add_edge(ie.src, ie.src_conn, nsdfg, ie.data.data,
                            dace.memlet.Memlet.from_array(ie.data.data, sdfg.arrays[ie.data.data]) if ie.data.data is not None else dace.memlet.Memlet(None))
        for oe in state.out_edges(state.exit_node(map_entry)):
            if oe.data.data is not None:
                state.add_edge(nsdfg, oe.data.data, oe.dst, oe.dst_conn,
                            dace.memlet.Memlet.from_array(oe.data.data, sdfg.arrays[oe.data.data]) if oe.data.data is not None else dace.memlet.Memlet(None))

        # Copy over all nodes and edges to inner SDFGs
        for node in inner_body_nodes:
            if isinstance(node, dace.nodes.NestedSDFG):
                node.sdfg.parent_graph = inner_state
                node.sdfg.parent_sdfg = inner_cfg.sdfg
            inner_state.add_node(node)
        for edge in inner_body_edges:
            inner_state.add_edge(*edge)
        for node in inner_body_nodes2:
            if isinstance(node, dace.nodes.NestedSDFG):
                node.sdfg.parent_graph = remainder_state
                node.sdfg.parent_sdfg = remainder_cfg.sdfg
            remainder_state.add_node(node)
        for edge in inner_body_edges2:
            remainder_state.add_edge(*edge)
        for ie in inedges:
            if ie.dst == map_entry1[0]:
                if ie.data.data is not None:
                    an = inner_state.add_access(ie.data.data)
                    inner_state.add_edge(an, None, map_entry1[1], ie.dst_conn,
                                        copy.deepcopy(ie.data))
            if ie.dst == map_entry2[0]:
                if ie.data.data is not None:
                    an = remainder_state.add_access(ie.data.data)
                    remainder_state.add_edge(an, None, map_entry2[1], ie.dst_conn,
                                        copy.deepcopy(ie.data))
        for oe in outedges:
            if oe.src == map_exit1[0]:
                if oe.data.data is not None:
                    an = inner_state.add_access(oe.data.data)
                    inner_state.add_edge(map_exit1[1], oe.src_conn, an, None,
                                        copy.deepcopy(oe.data))
            if oe.src == map_exit2[0]:
                if oe.data.data is not None:
                    an = remainder_state.add_access(oe.data.data)
                    remainder_state.add_edge(map_exit2[1], oe.src_conn, an, None,
                                        copy.deepcopy(oe.data))
        # Add missing datadaesc
        def add_missing_desc(sdfg, inner_sdfg, inner_state):
            for n in inner_state.nodes():
                if isinstance(n, nodes.AccessNode):
                    if n.data not in inner_sdfg.arrays:
                        inner_sdfg.add_datadesc(n.data, copy.deepcopy(sdfg.arrays[n.data]))
        add_missing_desc(sdfg, inner_sdfg, inner_state)
        add_missing_desc(sdfg, inner_sdfg, remainder_state)


        # If we have the remainder condition, we need to add the bound checks
        remainder_maps = []
        for n in remainder_state.nodes():
            if isinstance(n, nodes.MapEntry):
                remainder_maps.append(n)
        assert len(remainder_maps) == 1
        remainder_map = remainder_maps[0]
        # Add the bound checks
        new_remainder_range = []
        for (db, de, ds), (ib, ie, _is) in zip(dev_entry.map.range, remainder_map.map.range):
            # 2- one with bound checks is (ib, Min(ie-1, de-1)+1, is)
            new_remainder_range.append((ib, symbolic.SymExpr(f"Min({ie}, {de})"), _is))
        remainder_map.map.range = dace.subsets.Range(new_remainder_range)

        remainder_loop_stencil_map_counter  += 1

        # Remove the old nodes
        for node in nodes_to_remove:
            state.remove_node(node)

        sdfg.reset_cfg_list()
        sdfg.validate()



    @staticmethod
    def annotates_memlets():
        return True
