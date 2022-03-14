# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" This module contains classes that implement the expansion transformation.
"""

from dace import dtypes, registry, symbolic, subsets
from dace.sdfg import nodes
from dace.memlet import Memlet
from dace.sdfg import replace, SDFG, dynamic_map_inputs
from dace.sdfg.graph import SubgraphView
from dace.transformation import transformation
from dace.properties import make_properties, Property
from dace.symbolic import symstr
from dace.sdfg.propagation import propagate_memlets_sdfg
from dace.transformation.subgraph import helpers
from collections import defaultdict

from copy import deepcopy as dcpy
from typing import List, Union

import itertools
import dace.libraries.standard as stdlib

import warnings
import sys

from dace.transformation.dataflow import map_expansion, map_collapse

@make_properties
class SplitMaps(transformation.SubgraphTransformation):

    debug = Property(dtype=bool, desc="Debug Mode", default=False)

    def can_be_applied(self, sdfg: SDFG, subgraph: SubgraphView) -> bool:
        graph = subgraph.graph

        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
        if len(map_entries) <= 1:
            return False

        return True

    def apply(self, sdfg, subgraph):
        graph = subgraph.graph

        map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)

        ranges = {}
        for i, map_entry in enumerate(map_entries):
            for j, param in enumerate(map_entry.map.params):
                start, stop, step = map_entry.map.range.ranges[j]
                # TODO: Fix for non-normalized maps
                if start > 0 or step > 1:
                    raise ValueError("Improve")

                if not param in ranges:
                    ranges[param] = []

                ranges[param].append(stop)

        divisors = {param: min(values) for param, values in ranges.items()}
        divisors = dict(sorted(divisors.items(), key=lambda item: item[1]))

        for param, divisor in divisors.items():
            current_map_entries = helpers.get_outermost_scope_maps(sdfg, graph, subgraph)
            for map_entry in current_map_entries:
                current_map = map_entry.map
                if param not in current_map.params:
                    continue

                param_index = current_map.params.index(param)
                start, stop, step = current_map.ranges.range(param_index)

                main_range = subsets.Range(start, divisor, step)
                main_ranges = dcpy(current_map.ranges)
                main_ranges.pop(param_index)
                main_ranges.insert(param_index, main_range)
                main_map = nodes.Map(map_entry.label, current_map.params, main_ranges, schedule=current_map.schedule)
                main_entry = nodes.MapEntry(main_map)
                main_exist = nodes.MapExit(main_map)

                remainder_range = subsets.Range(divisor, stop - divisor, step)
                remainder_ranges = dcpy(current_map.ranges)
                remainder_ranges.pop(param_index)
                remainder_ranges.insert(param_index, remainder_range)

                remainer_map = nodes.Map(map_entry.label + "_remainder", current_map.params, remainder_ranges, schedule=current_map.schedule)
                remainder_entry = nodes.MapEntry(remainer_map)
                remainder_exit = nodes.MapExit(remainer_map)

                # TODO: Divide memlets w.r.t current param
                # TODO: Remove map_entry and add main & remainder