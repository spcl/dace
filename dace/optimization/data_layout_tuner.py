# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import copy
import math

import itertools

from typing import Generator, Tuple, Dict, List, Sequence, Set

from dace import data as dt, SDFG, dtypes
from dace.optimization import cutout_tuner
from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter

try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x: x


class DataLayoutTuner(cutout_tuner.CutoutTuner):
    def __init__(self, sdfg: SDFG, measurement: dtypes.InstrumentationType = dtypes.InstrumentationType.Timer) -> None:
        super().__init__(sdfg=sdfg)
        self.instrument = measurement

    def cutouts(self) -> Generator[Tuple[dace.SDFGState, dace.nodes.Node], None, None]:
        for node, state in self._sdfg.all_nodes_recursive():
            if xfh.get_parent_map(state, node) is not None:
                continue

            if isinstance(node, dace.nodes.MapEntry):
                yield state, node
            elif isinstance(node, (dace.nodes.LibraryNode, dace.nodes.Tasklet)):
                yield state, node

    def space(self, cutout_sdfg: dace.SDFG, groups: List[Set[str]] = None) -> Generator[Set[str], None, None]:
        # Make a copy of the original arrays
        arrays = copy.deepcopy(cutout_sdfg.arrays)

        # Tuning groups - if None, each array is in its own group
        group_dims: List[int] = []
        if groups is None:
            groups = [{k} for k, v in arrays.items() if not v.transient]
            group_dims = [len(v.shape) for v in arrays.values() if not v.transient]
        else:
            # Verify all groups have the same dimensionality
            for group in groups:
                ndims = None
                for member in group:
                    if ndims is not None and len(arrays[member].shape) != ndims:
                        raise ValueError(
                            f'Group "{group}" contains arrays with different dimensions. Cannot tune together')
                    ndims = len(arrays[member].shape)
                group_dims.append(ndims)

        # Create the number of configurations - all combinations of all permutations of each group
        group_layouts = [itertools.permutations(list(range(dims))) for dims in group_dims]
        configurations = itertools.product(*group_layouts)

        for config in tqdm(list(configurations)):
            config: Sequence[Sequence[int]]

            # Reset arrays
            cutout_sdfg._arrays = copy.deepcopy(arrays)

            # Set array strides
            modified_arrays = set()
            for group, group_config in zip(groups, config):
                for member in group:
                    desc = cutout_sdfg.arrays[member]
                    strides, total_size = desc.strides_from_layout(*group_config)
                    cutout_sdfg.arrays[member].strides = strides
                    cutout_sdfg.arrays[member].total_size = total_size
                    modified_arrays.add(member)

            # Yield configuration
            yield modified_arrays

    def optimize(self, apply: bool = True, group_inputs: bool = True, measurements: int = 30) -> Dict:
        dreport = self._sdfg.get_instrumented_data()

        tuning_report = {}
        for state, node in self.cutouts():
            subgraph_nodes = state.scope_subgraph(node).nodes() if isinstance(node, dace.nodes.MapEntry) else [node]
            cutout = cutter.cutout_state(state, *subgraph_nodes)
            cutout.instrument = self.instrument

            # Prepare original arguments to sub-SDFG from instrumented data report
            arguments: Dict[str, dt.ArrayLike] = {}
            groups = [set(), set()] if group_inputs else None
            for cstate in cutout.nodes():
                for dnode in cstate.data_nodes():
                    if cutout.arrays[dnode.data].transient:
                        continue
                    # Set tuning groups as necessary
                    if group_inputs and dnode.data not in arguments:
                        if cstate.in_degree(dnode) > 0:
                            groups[1].add(dnode.data)  # Outputs
                        else:
                            groups[0].add(dnode.data)  # Inputs

                    arguments[dnode.data] = dreport.get_first_version(dnode.data)

            results = {}
            best_choice = None
            best_runtime = math.inf
            for modified_arrays in self.space(cutout_sdfg=cutout, groups=groups):
                # Modify data layout prior to calling
                for marray in modified_arrays:
                    arguments[marray] = dt.make_array_from_descriptor(cutout.arrays[marray], arguments[marray])

                layout = '\n'.join([f'  {k}: {v.strides}' for k, v in cutout.arrays.items() if not v.transient])

                runtime = self.measure(cutout, arguments, repetitions=measurements)
                results[layout] = runtime

                if runtime < best_runtime:
                    best_choice = modified_arrays
                    best_runtime = runtime

            if apply and best_choice is not None:
                # TODO:
                pass

            tuning_report[node.label] = results

        return tuning_report
