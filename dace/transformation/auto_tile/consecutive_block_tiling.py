# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes and functions that implement the grid-strided map tiling
    transformation."""


import sympy
from dace.memlet import Memlet
from dace.sdfg import SDFG, SDFGState
from dace.properties import make_properties, Property
from dace.sdfg import nodes
from dace.sdfg import utils as sdutil
from dace.sdfg.propagation import (
    propagate_memlet,
    propagate_memlets_scope,
    propagate_memlets_state,
)
from dace.transformation import transformation
from dace import dtypes, subsets
from dace.symbolic import SymExpr, symbol
import dace
import copy

from dace.transformation.dataflow.tiling import MapTiling
import re


def extract_number_from_string(s):
    match = re.search(r"(\d+)$", s)
    if match:
        return int(match.group(1))
    return None


@make_properties
class ConsecutiveBlockTiling(transformation.SingleStateTransformation):
    """
    Block tiling is applied twice - data hoisting pattern changes.
    It block tiles an already block tiled map.

    Let's say we have block tiled the map k=0:K to
    k=0:K:64 and tk=0:64
    then the consecutive block tiling map will make to for example
    k=0:K:64 and tk=0:64, and tk2=0:32

    This requires
    """

    block_tiled_map_entry = transformation.PatternNode(nodes.MapEntry)

    # Properties
    block_tile_factor = Property(dtype=tuple, default=(2,), desc="")
    level = Property(dtype=int, default=1, desc="")

    global_application_number = 0

    unroll = Property(dtype=bool, default=True, desc="Unroll the map")

    @classmethod
    def expressions(cls):
        return [sdutil.node_path_graph(cls.block_tiled_map_entry)]

    def can_be_applied(self, state, expr_index, sdfg, permissive=False):
        return True

    def find_next_inner_work_map_entry(self, state, start_node, _str):
        visited = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)

            if (isinstance(node, dace.sdfg.nodes.MapEntry) and
                node.label.startswith(_str)):
                return node
            for succ in state.successors(node):
                stack.append(succ)

        return None

    def replace_subsets(self, state: SDFGState, map_node, match_subset, replace_subset):
        edges_to_check = state.out_edges(map_node)
        while len(edges_to_check) > 0:
            edge = edges_to_check.pop()
            u, u_conn, v, v_conn, memlet = edge
            new_range_list = []
            if memlet.subset:
                for _range in memlet.subset:
                    if _range == match_subset:
                        new_range_list.append(replace_subset)
                    else:
                        new_range_list.append(_range)
            else:
                new_range_list = None
            new_memlet = Memlet(
                data=memlet.data,
                subset=(
                    subsets.Range(new_range_list)
                    if new_range_list != None
                    else new_range_list
                ),
                wcr=memlet.wcr,
                wcr_nonatomic=memlet.wcr_nonatomic,
                allow_oob=memlet.allow_oob,
                debuginfo=memlet.debuginfo,
            )
            state.remove_edge(edge)
            state.add_edge(u, u_conn, v, v_conn, new_memlet)
            # print("AE8", u, u_conn, v, v_conn, new_memlet)
            edges_to_check += [
                e for e in state.out_edges(v) if e != state.exit_node(map_node)
            ]

    def apply(self, state: SDFGState, sdfg: SDFG):
        map_entry = self.block_tiled_map_entry

        # Just tiling does not work

        assert len(self.block_tile_factor) == len(map_entry.map.params)

        param_size_new_size = [
            (param, map_range[2] - map_range[0], (map_range[2] - map_range[0]) / factor)
            for param, map_range, factor in zip(
                map_entry.map.params, map_entry.map.range, self.block_tile_factor
            )
        ]
        for param, old_size, new_size in param_size_new_size:
            print(param, old_size, new_size)

        # Add the new level of work map
        nl_map_entry, nl_map_exit = state.add_map(
            name=f"OuterWorkMapLevel{self.level}No{ConsecutiveBlockTiling.global_application_number}",
            ndrange=dict(
                [
                    (f"{param}_bl{self.level}", f"0:{old_size}:{new_size}")
                    for param, old_size, new_size in param_size_new_size
                ]
            ),
            schedule=dace.ScheduleType.Sequential,
            unroll=self.unroll,
        )

        # Re-route all outgoing edges from previous work map to this work map
        edges_to_rm = []
        ranges_to_repl = []
        # First for map entry
        for oe in state.out_edges(map_entry):
            edges_to_rm.append(oe)
            nl_in_conn = oe.src_conn.replace("OUT_", "IN_") if oe.src_conn else None
            nl_out_conn = oe.src_conn
            state.add_edge(
                oe.src, oe.src_conn, nl_map_entry, nl_in_conn, copy.deepcopy(oe.data)
            )
            # The outgoing data changes, if k is in the subset then divide it by the factor,
            if oe.data is not None and oe.data.subset is not None:
                m2 = oe.data.subset
                old_ranges = []
                new_ranges = []
                for r in m2.ranges:
                    split = -1
                    for _id, param in enumerate(map_entry.params):
                        print(
                            param,
                            r,
                            r[0].free_symbols,
                            r[1].free_symbols,
                            r[2].free_symbols,
                        )
                        if (
                            dace.symbol(param) in r[0].free_symbols
                            or dace.symbol(param) in r[1].free_symbols
                            or dace.symbol(param) in r[2].free_symbols
                        ):
                            # Then we need to split
                            split = _id
                            break
                    if split != -1:
                        print(split)
                        l = r[1] - r[0] + 1
                        l = l // self.block_tile_factor[split]
                        ns = dace.symbol(f"{param}_bl{self.level}")
                        new_ranges.append((r[0] + ns, r[0] + l - 1 + ns, 1))
                        print(r, type(r))
                        ranges_to_repl.append((r, (r[0] + ns, r[0] + l - 1 + ns, 1)))
                        old_ranges.append(r)
                    else:
                        new_ranges.append(r)
                        old_ranges.append(r)
                m3 = dace.memlet.Memlet(
                    data=oe.data.data,
                    subset=subsets.Range(new_ranges),
                )
            else:
                m3 = dace.memlet.Memlet(None)
            print(m3)
            state.add_edge(nl_map_entry, nl_out_conn, oe.dst, oe.dst_conn, m3)
            if nl_in_conn is not None and nl_in_conn not in nl_map_entry.in_connectors:
                nl_map_entry.add_in_connector(nl_in_conn)
            if (
                nl_out_conn is not None
                and nl_out_conn not in nl_map_entry.out_connectors
            ):
                nl_map_entry.add_out_connector(nl_out_conn)

        # Then for map exit
        for ie in state.in_edges(state.exit_node(map_entry)):
            edges_to_rm.append(ie)
            nl_in_conn = ie.dst_conn
            nl_out_conn = ie.dst_conn.replace("IN_", "OUT_") if ie.dst_conn else None
            state.add_edge(
                ie.src, ie.src_conn, nl_map_exit, nl_in_conn, copy.deepcopy(ie.data)
            )
            state.add_edge(
                nl_map_exit, nl_out_conn, ie.dst, ie.dst_conn, copy.deepcopy(ie.data)
            )
            if nl_in_conn is not None and nl_in_conn not in nl_map_exit.in_connectors:
                nl_map_exit.add_in_connector(nl_in_conn)
            if (
                nl_out_conn is not None
                and nl_out_conn not in nl_map_exit.out_connectors
            ):
                nl_map_exit.add_out_connector(nl_out_conn)

        # Update the range of inner work map
        num = extract_number_from_string(map_entry.label)
        for n in state.nodes():
            if n.label == f"InnerWorkMapNo{num}" and isinstance(n, nodes.MapEntry):
                new_ranges = []
                for _i, r in enumerate(n.map.range):
                    print("BR", r[1],  (r[1]) // self.block_tile_factor[_i] )
                    assert r[0] == 0 and r[2] == 1
                    new_ranges.append((0, (r[1]) // self.block_tile_factor[_i], 1))
                n.map.range = dace.subsets.Range(new_ranges)

        for e in edges_to_rm:
            state.remove_edge(e)

        for rold, rnew in ranges_to_repl:
            print("RO", rold, "RN", rnew)
            self.replace_subsets(state, nl_map_entry, rold, rnew)

        for oe in state.out_edges(nl_map_entry):
            print("Prop", propagate_memlet(state, oe.data, nl_map_entry, False))

        # propagate_memlets_scope()
        # Outer Work Loop is K=0, K<?, K+=STEP
        # Intermediate Work Loop is k_bl1=0, k_bl1<STEP, k_bl1+= STEP/factor
        # Inner Work Loop is tk=0, tk<STEP/factor, tk+=1
        # If any access involves k, we need to make it to (k+k_bl1)

        inner_work_map_entry = self.find_next_inner_work_map_entry(state, nl_map_entry, "InnerWorkMapNo")

        new_params = [f"{param}_bl{self.level}" for param in map_entry.map.params]
        print("NP", new_params)
        edges_to_check = set(state.out_edges(inner_work_map_entry))
        edges_to_rm = set()
        edges_to_add = set()
        while edges_to_check:
            edge = edges_to_check.pop()
            u, u_conn, v, v_conn, memlet = edge
            if u == nl_map_entry:
                edges_to_check = edges_to_check.union(set(state.out_edges(v)))
                continue
            new_ranges = []
            if memlet is not None and memlet.subset is not None:
                for beg, end, step in memlet.subset:
                    for param, new_param in zip(map_entry.params, new_params):
                        param_symbol = symbol(param)
                        new_param_symbol = symbol(new_param)

                        _beg = beg.subs(param_symbol, SymExpr(param_symbol + new_param_symbol))
                        _end = end.subs(param_symbol, SymExpr(param_symbol + new_param_symbol))
                        _step = step.subs(param_symbol, new_param_symbol)

                    new_ranges.append((_beg, _end, _step))
                new_memlet = Memlet(subset=subsets.Range(new_ranges), data=memlet.data, wcr=memlet.wcr, wcr_nonatomic=memlet.wcr_nonatomic, allow_oob=memlet.allow_oob, debuginfo=memlet.debuginfo)
                #state.remove_edge(edge)
                edges_to_rm.add(edge)
                #state.add_edge(u, u_conn, v, v_conn, new_memlet)
                edges_to_add.add((u, u_conn, v, v_conn, new_memlet))
                #print("AE1:",u, u_conn, v, v_conn, new_memlet)
                if v != nl_map_exit:
                    edges_to_check.union(set(state.out_edges(v)))
        for e in edges_to_rm:
            state.remove_edge(e)
        for e in edges_to_add:
            state.add_edge(*e)


    @staticmethod
    def annotates_memlets():
        return True
