# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
A set of built-in hooks.
"""
from contextlib import contextmanager
import fnmatch
import os
from typing import Any, Callable, List, Optional, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dace.sdfg import SDFG
    from dace.dtypes import InstrumentationType, DataInstrumentationType


@contextmanager
def profile(repetitions: int = 100, warmup: int = 0):
    """
    Context manager that enables profiling of each called DaCe program. If repetitions is greater than 1, the
    program is run multiple times and the average execution time is reported.

    Example usage:

    .. code-block:: python

        with dace.profile(repetitions=100) as profiler:
            some_program(...)
            # ...
            other_program(...)

        # Print all execution times of the last called program (other_program)
        print(profiler.times[-1])


    :param repetitions: The number of times to run each DaCe program.
    :param warmup: Number of additional repetitions to run the program without measuring time.
    :note: Running functions multiple times may affect the results of the program.
    """
    from dace.frontend.operations import CompiledSDFGProfiler  # Avoid circular import
    from dace.hooks import on_compiled_sdfg_call, _COMPILED_SDFG_CALL_HOOKS

    # TODO: By default, do not profile every invocation

    # If already profiling, return existing profiler and change its properties
    for hook in _COMPILED_SDFG_CALL_HOOKS:
        if isinstance(hook, CompiledSDFGProfiler):
            hook.times.clear()
            hook.repetitions = repetitions
            hook.warmup = warmup
            yield hook
            return

    profiler = CompiledSDFGProfiler(repetitions, warmup)

    with on_compiled_sdfg_call(context_manager=profiler):
        yield profiler

    # After profiling is complete, store the file
    if len(profiler.times) == 0:
        return

    if len(profiler.times) > 1:
        # More than one profile saves locally to the cwd
        profiler.filename = f'report-{profiler.report.name}.json'
    else:
        profiler.filename = os.path.join(profiler.times[0][0].build_folder, 'perf',
                                         f'report-{profiler.report.name}.json')

    profiler.report.save(profiler.filename)


@contextmanager
def instrument(itype: 'InstrumentationType',
               filter: Optional[Union[str, Callable[[Any], bool]]],
               annotate_maps: bool = True,
               annotate_tasklets: bool = False,
               annotate_states: bool = False,
               annotate_sdfgs: bool = False):
    """
    Context manager that instruments every called DaCe program. Depending on the given instrumentation
    type and parameters, annotates the given elements on the SDFG. Filtering is possible with strings
    and wildcards, or a function (if given).

    Example usage:

    .. code-block:: python

        with dace.instrument(dace.InstrumentationType.GPU_Events, 
                             filter='*add??') as profiler:
            some_program(...)
            # ...
            other_program(...)

        # Print instrumentation report for last call
        print(profiler.reports[-1])


    :param itype: Instrumentation type to use.
    :param filter: An optional string with ``*`` and ``?`` wildcards, or function that receives
                   one parameter, determining whether to instrument the element or not.
    :param annotate_maps: If True, instruments scopes (e.g., map, consume) in the SDFGs.
    :param annotate_tasklets: If True, instruments tasklets in the SDFGs.
    :param annotate_states: If True, instruments states in the SDFGs.
    :param annotate_sdfgs: If True, instruments whole SDFGs and sub-SDFGs.
    """
    from dace.hooks import on_call
    from dace.codegen.instrumentation.report import InstrumentationReport
    from dace.sdfg import SDFGState
    from dace.sdfg.nodes import EntryNode, Tasklet

    # If a string was given, construct predicate based on wildcard name matching
    filter_func: Callable[[Any], bool] = lambda _: True
    if isinstance(filter, str):
        filter_func = lambda elem: fnmatch.fnmatch(elem.name, filter)
    elif callable(filter):
        filter_func = filter

    class Instrumenter:
        def __init__(self):
            self.reports: List[InstrumentationReport] = []

        @property
        def report(self):
            if not self.reports:
                return InstrumentationReport(None)
            return self.reports[-1]

        @contextmanager
        def __call__(self, sdfg: 'SDFG'):
            #######
            # Instrument SDFG
            if annotate_sdfgs:
                for sd in sdfg.all_sdfgs_recursive():
                    if filter_func(sd):
                        sd.instrument = itype

            for n, _ in sdfg.all_nodes_recursive():
                should_try = False
                should_try |= annotate_states and isinstance(n, SDFGState)
                should_try |= annotate_maps and isinstance(n, EntryNode)
                should_try |= annotate_tasklets and isinstance(n, Tasklet)

                if should_try and filter_func(n):
                    n.instrument = itype

            # Execute
            yield sdfg

            # After execution, do nothing


    profiler = Instrumenter()
    with on_call(context_manager=profiler):
        yield profiler


def cli_optimize_on_call(sdfg: 'SDFG'):
    """
    Calls a command-line interface for interactive SDFG transformations
    on every DaCe program call.

    :param sdfg: The current SDFG to optimize.
    """

    from dace.transformation.optimizer import SDFGOptimizer
    opt = SDFGOptimizer(sdfg)
    return opt.optimize()
