# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.

from typing import Any, Dict, List, Sequence, Set, Tuple
import copy
import itertools
import numpy as np

import dace
from dace.transformation import helpers as xfh
from dace.transformation.auto.auto_optimize import auto_optimize

# Cutout and instrumentation
from dace.sdfg.analysis.cutout import cutout_state
from dace.codegen.instrumentation.report import InstrumentationReport
from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport, ArrayLike

# Optional progress bar
try:
    from tqdm import tqdm
except (ImportError, ModuleNotFoundError):
    tqdm = lambda x: x


def instrument_data(sdfg: dace.SDFG, *args, **kwargs):
    """
    Checks if program has an instrumented data report and if not, instruments and runs the entire SDFG to save all
    data. If a report exists, but uses different sizes, the current report is also cleared.
    :param sdfg: The SDFG to call.
    :param args: Arguments to call the program with.
    :param kwargs: Keyword arguments to call the program with.
    """
    # Update kwargs with args
    kwargs.update({aname: a for aname, a in zip(sdfg.arg_names, args)})

    # Check existing instrumented data for shape mismatch
    dreport: InstrumentedDataReport = sdfg.get_instrumented_data()
    if dreport is not None:
        for data in dreport.keys():
            rep_arr = dreport.get_first_version(data)
            sdfg_arr = sdfg.arrays[data]
            # Potential shape mismatch
            if rep_arr.shape != sdfg_arr.shape:
                # Check given data first
                if hasattr(kwargs[data], 'shape') and rep_arr.shape != kwargs[data].shape:
                    sdfg.clear_data_reports()
                    dreport = None
                    break

    # If there is no valid instrumented data available yet, run in data instrumentation mode
    if dreport is None:
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.AccessNode) and not node.desc(sdfg).transient:
                node.instrument = dace.DataInstrumentationType.Save

        sdfg(**kwargs)

        # Disable data instrumentation from now on
        for node, _ in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.AccessNode):
                node.instrument = dace.DataInstrumentationType.No_Instrumentation


def configs(sdfg: dace.SDFG, groups: List[Set[str]] = None):
    """
    Exhaustively tune data layout configurations in each tuning group, or each individual array if groups not given.
    Yields the arrays that differ from the original arrays.
    """
    # Make a copy of the original arrays
    arrays = copy.deepcopy(sdfg.arrays)

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
                    raise ValueError(f'Group "{group}" contains arrays with different dimensions. Cannot tune together')
                ndims = len(arrays[member].shape)
            group_dims.append(ndims)

    # Create the number of configurations - all combinations of all permutations of each group
    group_layouts = [itertools.permutations(list(range(dims))) for dims in group_dims]
    configurations = itertools.product(*group_layouts)

    for config in tqdm(list(configurations)):
        config: Sequence[Sequence[int]]

        # Reset arrays
        sdfg._arrays = copy.deepcopy(arrays)

        # Set array strides
        modified_arrays = set()
        for group, group_config in zip(groups, config):
            for member in group:
                desc = sdfg.arrays[member]
                strides, total_size = desc.strides_from_layout(*group_config)
                sdfg.arrays[member].strides = strides
                sdfg.arrays[member].total_size = total_size
                modified_arrays.add(member)

        # Yield configuration
        yield modified_arrays


def make_array_from_descriptor(original_array: ArrayLike, descriptor: dace.data.Array) -> ArrayLike:
    """ Creates an array that matches the given data descriptor, and copies the original array to it. """
    # Make numpy array from data descriptor
    npdtype = descriptor.dtype.as_numpy_dtype()
    buffer = np.ndarray([descriptor.total_size], dtype=npdtype)
    view = np.ndarray(descriptor.shape, npdtype, buffer=buffer, strides=descriptor.strides)
    view[:] = original_array
    return view


def cutout_candidates(sdfg: dace.SDFG) -> List[Tuple[dace.SDFGState, List[dace.nodes.Node]]]:
    """
    Select a set of candidates for cut out of a program. In this case, the candidates are all top-level 
    map scopes and library nodes.
    :param sdfg: The SDFG to search in.
    :return: A list of tuples with (state, list of participating nodes to cut out)
    """
    result = []
    for node, state in sdfg.all_nodes_recursive():
        # Skip nodes that are in scopes
        if xfh.get_parent_map(state, node) is not None:
            continue

        # A map scope is taken with its contents
        if isinstance(node, dace.nodes.MapEntry):
            result.append((state, state.scope_subgraph(node).nodes()))
        # A library node is taken as is
        elif isinstance(node, (dace.nodes.LibraryNode, dace.nodes.Tasklet)):
            result.append((state, [node]))

    return result


def tune_local_data_layouts(sdfg: dace.SDFG, group_inputs: bool = True, repetitions: int = 100):
    """
    Reports the best data layout for each individual sub-computation.
    :param sdfg: The SDFG to tune.
    :param group_inputs: If True, tunes only two groups at the same time - inputs and outputs.
    """
    dreport: InstrumentedDataReport = sdfg.get_instrumented_data()
    printout: List[Tuple[float, str]] = []

    for state, subgraph in cutout_candidates(sdfg):
        cut_sdfg = cutout_state(state, *subgraph)

        # Instrument SDFG for timing
        cut_sdfg.instrument = dace.InstrumentationType.Timer

        # Prepare original arguments to sub-SDFG from instrumented data report
        arguments: Dict[str, ArrayLike] = {}
        groups = [set(), set()] if group_inputs else None
        for cstate in cut_sdfg.nodes():
            for dnode in cstate.data_nodes():
                if cut_sdfg.arrays[dnode.data].transient:
                    continue
                # Set tuning groups as necessary
                if group_inputs and dnode.data not in arguments:
                    if cstate.in_degree(dnode) > 0:
                        groups[1].add(dnode.data)  # Outputs
                    else:
                        groups[0].add(dnode.data)  # Inputs

                arguments[dnode.data] = dreport.get_first_version(dnode.data)

        # Use configurations to change data layouts
        for modified_arrays in configs(cut_sdfg, groups):
            # Modify data layout prior to calling
            for marray in modified_arrays:
                arguments[marray] = make_array_from_descriptor(arguments[marray], cut_sdfg.arrays[marray])

            # Call instrumented SDFG multiple times and create one report
            with dace.config.set_temporary('debugprint', value=False):  # Clean printouts
                with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
                    csdfg = cut_sdfg.compile()
                    for _ in range(repetitions):
                        csdfg(**arguments)
                    csdfg.finalize()  # Creates instrumentation report

            # Create result
            report: InstrumentationReport = cut_sdfg.get_latest_report()
            layout = '\n'.join([f'  {k}: {v.strides}' for k, v in cut_sdfg.arrays.items() if not v.transient])
            # Get duration from report (first element, first value)
            durations = next(iter(next(iter(report.durations.values())).values()))
            # layout += f'\nAll runtimes: {durations}'
            median_duration = np.median(np.array(durations))
            printout.append((median_duration, layout))

    print("Top three layouts:")
    for median_duration, layout in sorted(printout)[:3]:
        print('Layouts:')
        print(layout)
        print('Runtime:', median_duration, 'ms')


if __name__ == '__main__':
    N = 200
    A = np.random.rand(N, N, N)
    B = np.random.rand(N, N)

    @dace.program
    def sample(A, B):
        B[:] = np.sum(A, axis=0)

    sdfg = sample.to_sdfg(A, B)
    auto_optimize(sdfg, dace.DeviceType.CPU)

    # If there is no valid instrumented data available yet, run program in data instrumentation mode
    instrument_data(sdfg, A, B)

    # Since we now have instrumented data, we can tune data layouts
    tune_local_data_layouts(sdfg)
