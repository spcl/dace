# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
A set of built-in hooks.
"""
from contextlib import contextmanager
import fnmatch
import os
from typing import Any, Callable, List, Optional, Tuple, TYPE_CHECKING, Union

if TYPE_CHECKING:
    from dace.dtypes import InstrumentationType, DataInstrumentationType
    from dace.codegen.compiled_sdfg import CompiledSDFG
    from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport
    from dace.sdfg import SDFG


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
        filename = f'report-{profiler.report.name}.json'
    else:
        filename = os.path.join(profiler.times[0][0].build_folder, 'perf', f'report-{profiler.report.name}.json')

    profiler.report.filepath = filename
    profiler.report.save(filename)


def _make_filter_function(filter: Optional[Union[str, Callable[[Any], bool]]],
                          with_attr: bool = True) -> Callable[[Any], bool]:
    """
    Internal helper that makes a filtering function. 
      
      * If nothing is given, the filter always returns True.
      * If a string is given, performs wildcard matching.
      * If a callable is given, use predicate directly.
    

    :param filter: The filter to use.
    :param with_attr: If True, uses the ``name`` attribute for testing strings.
    """
    filter_func: Callable[[Any], bool] = lambda _: True
    if isinstance(filter, str):
        # If a string was given, construct predicate based on wildcard name matching
        if with_attr:
            filter_func = lambda elem: fnmatch.fnmatch(elem.name, filter)
        else:
            filter_func = lambda elem: fnmatch.fnmatch(elem, filter)
    elif callable(filter):
        filter_func = filter

    return filter_func


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

    # Create filtering function based on input
    filter_func = _make_filter_function(filter)

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
            # Instrument SDFG
            if annotate_sdfgs:
                for sd in sdfg.all_sdfgs_recursive():
                    if filter_func(sd):
                        sd.instrument = itype

            # Instrument elements
            for n, _ in sdfg.all_nodes_recursive():
                should_try = False
                should_try |= annotate_states and isinstance(n, SDFGState)
                should_try |= annotate_maps and isinstance(n, EntryNode)
                should_try |= annotate_tasklets and isinstance(n, Tasklet)

                if should_try and filter_func(n):
                    n.instrument = itype

            lastpath = sdfg.get_latest_report_path()

            # Execute
            yield sdfg

            # After execution, save report if new one was generated
            if lastpath != sdfg.get_latest_report_path():
                self.reports.append(sdfg.get_latest_report())

    profiler = Instrumenter()
    with on_call(context_manager=profiler):
        yield profiler


@contextmanager
def instrument_data(ditype: 'DataInstrumentationType',
                    filter: Optional[Union[str, Callable[[Any], bool]]],
                    restore_from: Optional[Union[str, 'InstrumentedDataReport']] = None,
                    verbose: bool = False):
    """
    Context manager that instruments (serializes/deserializes) the data of every called DaCe program.
    This can be used for reproducible runs and debugging. Depending on the given data instrumentation
    type and parameters, annotates the access nodes on the SDFG. Filtering is possible with strings
    and wildcards, or a function (if given). An optional instrumented data report can be given to
    load a specific set of data.

    Example usage:

    .. code-block:: python

        @dace
        def sample(a: dace.float64, b: dace.float64):
            arr = a + b
            return arr + 1

        with dace.instrument_data(dace.DataInstrumentationType.Save, filter='a??'):
            result_ab = sample(a, b)
        
        # Optionally, get the serialized data containers
        dreport = sdfg.get_instrumented_data()
        assert dreport.keys() == {'arr'}  # dreport['arr'] is now the internal ``arr``

        # Reload latest instrumented data (can be customized if ``restore_from`` is given)
        with dace.instrument_data(dace.DataInstrumentationType.Restore, filter='a??'):
            result_cd = sample(c, d)  # where ``c, d`` are different from ``a, b``
        
        assert numpy.allclose(result_ab, result_cd)



    :param ditype: Data instrumentation type to use.
    :param filter: An optional string with ``*`` and ``?`` wildcards, or function that receives
                   one parameter, determining whether to instrument the access node or not.
    :param restore_from: An optional parameter that specifies which instrumented data report to load
                         data from. It could be a path to a folder, an ``InstrumentedDataReport`` object,
                         or None to load the latest generated report.
    :param verbose: If True, prints information about created and loaded instrumented data reports.
    """
    import ctypes
    from dace.codegen.instrumentation.data.data_report import InstrumentedDataReport
    from dace.dtypes import DataInstrumentationType
    from dace.hooks import on_call, on_compiled_sdfg_call
    from dace.sdfg.nodes import AccessNode

    # Create filtering function based on input
    filter_func = _make_filter_function(filter, with_attr=False)

    class DataInstrumenter:
        @contextmanager
        def __call__(self, sdfg: 'SDFG'):
            for n, _ in sdfg.all_nodes_recursive():
                if isinstance(n, AccessNode) and filter_func(n.data):
                    n.instrument = ditype

            dreports = sdfg.available_data_reports()

            # Execute
            yield sdfg

            # After execution, check if data report was created or warn otherwise
            if ditype == DataInstrumentationType.Save:
                reports_after_execution = sdfg.available_data_reports()
                if len(dreports) == len(reports_after_execution):
                    print('No data instrumentation reports created. All data containers may have been filtered out.')
                elif verbose:
                    last_report = sorted(reports_after_execution)[-1]
                    folder = os.path.join(sdfg.build_folder, 'data', str(last_report))
                    print('Instrumented data report created at', folder)

    instrumenter = DataInstrumenter()

    if ditype == DataInstrumentationType.Restore:
        # Restore data into compiled SDFG
        class DataRestoreHook:
            @contextmanager
            def __call__(self, csdfg: 'CompiledSDFG', args: Tuple[Any, ...]):
                # Restore data from requested data report
                set_report = csdfg.get_exported_function('__dace_set_instrumented_data_report')
                if set_report is None:
                    print('Data instrumentation restores not found. All data containers may have been filtered out.')
                    yield
                    return

                if isinstance(restore_from, str):
                    folder = restore_from
                elif isinstance(restore_from, InstrumentedDataReport):
                    folder = restore_from.folder
                else:  # Use latest
                    timestamp = sorted(csdfg.sdfg.available_data_reports())[-1]
                    folder = os.path.join(csdfg.sdfg.build_folder, 'data', str(timestamp))
                    if verbose:
                        print('Loading instrumented data report from', folder)

                set_report(csdfg._libhandle, ctypes.c_char_p(os.path.abspath(folder).encode('utf-8')))
                yield

        with on_compiled_sdfg_call(context_manager=DataRestoreHook()):
            with on_call(context_manager=instrumenter):
                yield instrumenter
    else:
        with on_call(context_manager=instrumenter):
            yield instrumenter


def cli_optimize_on_call(sdfg: 'SDFG'):
    """
    Calls a command-line interface for interactive SDFG transformations
    on every DaCe program call.

    :param sdfg: The current SDFG to optimize.
    """

    from dace.transformation.optimizer import SDFGOptimizer
    opt = SDFGOptimizer(sdfg)
    return opt.optimize()
