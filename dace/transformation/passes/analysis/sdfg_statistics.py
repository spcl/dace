# Copyright 2019-2024 ETH Zurich and the DaCe authors. All rights reserved.
from collections import defaultdict
import re
from typing import Any, Dict, Optional, Set

import dace
from dace import SDFG, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.sdfg import utils as sdutil

class SDFGStatistics(ppl.Pass):
    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nothing

    def should_reapply(self, modified: ppl.Modifies) -> bool:
        return modified & ppl.Modifies.Symbols | ppl.Modifies.Edges | ppl.Modifies.Nodes | ppl.Modifies.States

    def depends_on(self):
        return {}

    def _print_defaultdict(self, d: defaultdict, tab_count = 0):
        def camel_to_title(s: str) -> str:
            s = re.sub(r'(_)', ' ', s)
            s = re.sub(r'(?<!^)(?=[A-Z])', ' ', s)
            return s.title().replace("Sdfg", "SDFG")

        retstr  = ""
        for key, value in d.items():
            readable_key = camel_to_title(key)

            if isinstance(value, set) and len(value) == 1:
                formatted_value = next(iter(value))
            elif isinstance(value, set) or isinstance(value, str):
                formatted_value = value
            elif isinstance(value, defaultdict):
                formatted_value = "\n" + self._print_defaultdict(value, tab_count + 1)[:-1]
            else:
                formatted_value = value

            retstr += (f"{tab_count * '\t'}{readable_key}: {formatted_value}\n")
        return retstr

    def _count_nested_sdfgs(self, sdfg: SDFG, level: int = 0, counts: Dict[int, int] = None) -> Dict[int, int]:
        if counts is None:
            counts = defaultdict(int)

        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, dace.nodes.NestedSDFG):
                    counts[level] += 1
                    self._count_nested_sdfgs(node.sdfg, level + 1, counts)

        return counts


    def _count_maps(self, sdfg: SDFG):
        num_outer_maps = 0
        maps = []
        for s in sdfg.states():
            sdict = s.scope_dict()
            for n in s.nodes():
                if isinstance(n, dace.nodes.MapEntry):
                    if sdict[n] is None:
                        num_outer_maps += 1
                    map_entry = n
                    map_exit = s.exit_node(n)
                    num_inner_maps = 0
                    inner_map_info = []
                    for inner_node in sdutil.dfs_topological_sort(s, map_entry):
                        if inner_node != map_entry and isinstance(inner_node, dace.nodes.MapEntry):
                            num_inner_maps += 1
                            inner_map_info.append((inner_node.map.label, inner_node.map.range))
                        if inner_node == map_exit:
                            break
                    maps.append((n.map.label, n.map.range, num_inner_maps, inner_map_info))
        return num_outer_maps, maps

    def apply_pass(self, sdfg: SDFG, _: Dict[str, Any], recursive: bool = False) -> Optional[Dict[str, Set[str]]]:
        statistics_dict: Dict[str, Set[str]] = defaultdict(lambda: set())

        states = sdfg.states()
        interstate_edges = sdfg.edges()

        statistics_dict["number_of_states"] = {len(states)}
        statistics_dict["number_of_interstate_edges"] = {len(interstate_edges)}

        conditional_interstate_edges = set()
        assignment_interstate_edges = set()

        for interstate_edge in interstate_edges:
            if not interstate_edge.data.is_unconditional:
                conditional_interstate_edges.add(interstate_edge)
            if len(interstate_edge.data.assignments) > 0:
                assignment_interstate_edges.add(interstate_edge)

        conditional_assignment_interstate_edges = set.intersection(conditional_interstate_edges, assignment_interstate_edges)

        statistics_dict["number_of_empty_interstate_edges"] = {len(interstate_edges)}
        statistics_dict["number_of_interstate_edges_with_assignments"] = {len(assignment_interstate_edges)}
        statistics_dict["number_of_interstate_edges_with_conditions"] = {len(conditional_interstate_edges)}
        statistics_dict["number_of_interstate_edges_with_conditions_and_assignments"] = {len(conditional_assignment_interstate_edges)}

        counts = self._count_nested_sdfgs(sdfg, 0, defaultdict(int))
        formatted_counts = defaultdict()
        for k, v in counts.items():
            formatted_counts[f"Level {k}"] = str(v)

        if len(formatted_counts) > 0:
            statistics_dict["number_of_nested_sdfgs_per_depth"] = formatted_counts
        else:
            statistics_dict["number_of_nested_sdfgs_per_depth"] = None

        statistics_dict["maximum_nested_sdfg_depth"] = {len(formatted_counts)}

        num_outer_maps, map_info = self._count_maps(sdfg)
        statistics_dict["number_of_outer_maps"] = num_outer_maps

        map_str = ""
        # (n.map.label, n.map.range, num_inner_maps, inner_map_info) = map_info
        for outer_map, outer_range, num_inner_maps, inner_map_info in map_info:
            map_str += f"\t{outer_map}: {outer_range}\n"
            map_str += f"\tNum Inner Maps: {num_inner_maps}\n"
            map_str += "\n".join([f"\t\t{f}: {s}" for f, s in inner_map_info])

        statistics_dict["map_information"] = "\n" + map_str if map_str != "" else None


        retstr = self._print_defaultdict(statistics_dict)[:-1]
        print(retstr)

        return statistics_dict

