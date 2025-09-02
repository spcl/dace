# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
A command-line tool that provides performance measurements and analysis on
Python scripts, modules, or existing instrumentation report files.
"""
import argparse
from contextlib import contextmanager
import runpy
import sys
import os
import shutil
from typing import Callable, List, Optional, Tuple, Union
import warnings

import dace
from dace.codegen.instrumentation.report import InstrumentationReport
from dace import dtypes

ExitCode = Union[int, str]
DEFAULT_REPETITIONS = 100


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser('daceprof',
                                     usage='''daceprof [-h] [arguments] file ...
       daceprof [arguments] -m module ...
       daceprof [arguments] -i profile.json''',
                                     formatter_class=argparse.RawDescriptionHelpFormatter,
                                     description='''
daceprof - DaCe Profiler and Report Viewer

This tool provides performance results and modeling for DaCe programs.
Calling it will collect results for every called program (or SDFG), unless
filtering arguments are used.

daceprof can be used with a Python script:
  daceprof [ARGUMENTS] myscript.py [SCRIPT ARGUMENTS]

with a Python executable module:
  daceprof [ARGUMENTS] -m package.module [MODULE ARGUMENTS]

or to print an existing report:
  daceprof [ARGUMENTS] -i profile.json
''')

    parser.add_argument('file', help='Path to the script or module', nargs='?')

    # Execution arguments
    group = parser.add_argument_group('execution arguments')
    group.add_argument('--module', '-m', help='Run an installed Python module', action='store_true')
    group.add_argument('--input', '-i', help='Read a report file and print summary', type=str)

    # Profiling arguments
    group = parser.add_argument_group('profiling arguments')
    group.add_argument('--repetitions',
                       '-r',
                       help='Runs each profiled program for the specified number of repetitions',
                       type=int,
                       default=DEFAULT_REPETITIONS)
    group.add_argument('--warmup',
                       '-w',
                       help='Number of additional repetitions to run without measurement',
                       type=int,
                       default=0)
    group.add_argument(
        '--type',
        '-t',
        help='Instrumentation type to use. If not given, times the entire SDFG with a wall-clock timer',
        choices=[
            e.name for e in dtypes.InstrumentationType
            if e not in [dtypes.InstrumentationType.No_Instrumentation, dtypes.InstrumentationType.Undefined]
        ])
    group.add_argument('--instrument',
                       help='Which elements to instrument. Can be a comma-separated list of element '
                       'types from the following: map, tasklet, state, sdfg',
                       default='map')
    group.add_argument('--sequential', help='Disable CPU multi-threading in code generation', action='store_true')

    # Data instrumentation
    group = parser.add_argument_group('data instrumentation arguments')
    group.add_argument('--save-data',
                       '-ds',
                       help='Enable data instrumentation and store all (or filtered) arrays',
                       action='store_true')
    group.add_argument('--restore-data',
                       '-dr',
                       help='Reproducibly run code by restoring all (or filtered) arrays',
                       action='store_true')

    # Filtering arguments
    group = parser.add_argument_group('filtering arguments')
    group.add_argument('--filter',
                       '-f',
                       help='Filter profiled elements with wildcards (e.g., *_map, assign_??). '
                       'Multiple filters can be separated by commas',
                       type=str)
    group.add_argument('--filter-data',
                       '-df',
                       help='Filter arrays to save/load in data instrumentation (with wildcards, comma-separated)',
                       type=str)

    # Report printout arguments
    group = parser.add_argument_group('report arguments')
    group.add_argument('--sort',
                       help='Sort report by a specific criterion',
                       choices=('min', 'max', 'mean', 'median', 'counter', 'value'))
    group.add_argument('--ascending', '-a', help='Sort in ascending order', action='store_true')
    group.add_argument('--csv', help='Print report as CSV', action='store_true')
    group.add_argument('--output', '-o', help='Report output file path', type=str)

    # Remainder of arguments
    parser.add_argument('args', nargs=argparse.REMAINDER, help='Script arguments')

    args = parser.parse_args()
    args.instrument = args.instrument.split(',')

    return parser, args


def validate_arguments(args: argparse.Namespace) -> Optional[str]:
    if not args.input and not args.file:
        return 'A script or module is required'
    if args.module and args.input:
        return 'Choose either script, module, or input file.'
    if args.input and (args.file or args.args):
        return 'Input file mode cannot specify additional arguments.'
    if args.input and args.output and not args.csv:
        return 'Cannot load and save a report at the same time.'
    if args.save_data and args.restore_data:
        return 'Choose either saving data containers or restoring them.'
    if args.type and (args.warmup or args.repetitions != DEFAULT_REPETITIONS):
        warnings.warn('Instrumentation mode is enabled, repetitions and warmup will be ignored.')
    for inst in args.instrument:
        if inst not in ('map', 'tasklet', 'state', 'sdfg'):
            return (f'Instrumentation element "{inst}" is not valid, please use a comma-separated list composed of '
                    '{map, tasklet, state, sdfg}')

    return None


@contextmanager
def _nop(*args, **kwargs):
    yield


def run_script_or_module(args: argparse.Namespace) -> Tuple[Optional[InstrumentationReport], Optional[str], ExitCode]:
    """
    Runs the script or module and returns the report file.

    :param args: The arguments with which ``daceprof`` was called.
    :return: A tuple of (report file name if created, exit code of original program)
    """
    # Modify argument list
    file = args.file
    sys.argv = [file] + args.args

    # Enable relevant call hooks
    hooks = enable_hooks(args)

    # Run script or module
    retval = None
    errcode = 0
    # Use built-in context managers for two hooks: profiling and data instrumentation
    if args.type:
        profile_ctx = dace.instrument(filter=args.filter,
                                      itype=dtypes.InstrumentationType[args.type],
                                      annotate_maps='map' in args.instrument,
                                      annotate_tasklets='tasklet' in args.instrument,
                                      annotate_states='state' in args.instrument,
                                      annotate_sdfgs='sdfg' in args.instrument)
    elif args.save_data or args.restore_data:
        # Data instrumentation will run once
        profile_ctx = _nop()
    else:
        # Profile full application
        profile_ctx = dace.profile(repetitions=args.repetitions, warmup=args.warmup)

    # Data instrumentation
    if args.save_data or args.restore_data:
        ditype = (dtypes.DataInstrumentationType.Save if args.save_data else dtypes.DataInstrumentationType.Restore)
        data_instrumenter = dace.instrument_data(filter=args.filter_data, ditype=ditype, verbose=True)
    else:
        data_instrumenter = _nop()

    with data_instrumenter:
        with profile_ctx as profiler:
            try:
                if args.module:
                    runpy.run_module(file, run_name='__main__')
                else:
                    runpy.run_path(file, run_name='__main__')
            except SystemExit as ex:
                # Skip internal exits
                if ex.code is not None and ex.code != 0:
                    print('daceprof: Application returned error code', ex.code)
                    errcode = ex.code

    # Unregister hooks
    for hook in hooks:
        dace.hooks.unregister_sdfg_call_hook(hook)

    # Warn if multiple reports were created
    if profiler:
        if args.type and len(profiler.reports) > 1:
            print('daceprof: Multiple report files created, showing last')
        if not args.type and len(profiler.times) > 1:
            print('daceprof: Multiple report files created, showing combined report')

        # Get instrumentation report file, if filled
        if profiler.report.events:
            retval = profiler.report

    return retval, errcode


def enable_hooks(args: argparse.Namespace) -> List[int]:
    # profile_entire_sdfg = args.type is None
    registered = []

    if args.sequential:

        def make_sequential(sdfg: dace.SDFG):
            # Disable OpenMP sections
            for sd in sdfg.all_sdfgs_recursive():
                sd.openmp_sections = False
            # Disable OpenMP maps
            for n, _ in sdfg.all_nodes_recursive():
                if isinstance(n, dace.nodes.EntryNode):
                    sched = getattr(n, 'schedule', False)
                    if sched in (dace.ScheduleType.CPU_Multicore, dace.ScheduleType.CPU_Persistent,
                                 dace.ScheduleType.Default):
                        n.schedule = dace.ScheduleType.Sequential

        registered.append(dace.hooks.register_sdfg_call_hook(before_hook=make_sequential))

    return registered


def save_as_csv(args: argparse.Namespace, report: InstrumentationReport):
    durations, counters = report.as_csv()
    if args.output:  # Print to file
        durfile = args.output + '.durations.csv'
        ctrfile = args.output + '.counters.csv'
        if durations:
            with open(durfile, 'w') as fp:
                fp.write(durations)
            print(f'Durations saved to {durfile}')
        if counters:
            with open(ctrfile, 'w') as fp:
                fp.write(counters)
            print(f'Counters saved to {ctrfile}')
    else:  # Print to console
        if durations:
            print('Durations:')
            print(durations)
        if durations and counters:
            print('\n')
        if counters:
            print('Counters:')
            print(counters)


def print_report(args: argparse.Namespace, reportfile: Union[str, InstrumentationReport]):
    if isinstance(reportfile, str):
        path = os.path.abspath(reportfile)
        if not os.path.isfile(path):
            print(path, 'does not exist, aborting.')
            exit(1)

        report = InstrumentationReport(path)
    else:
        report = reportfile

    if args.sort:
        report.sortby(args.sort, args.ascending)

    if args.csv:
        save_as_csv(args, report)
    else:
        print(report)


def main():
    parser, args = parse_arguments()

    # Argument checks
    errorstr = validate_arguments(args)
    if errorstr:
        parser.print_usage()
        print('error:', errorstr)
        exit(2)

    # Execute program or module
    if not args.input:
        report, errcode = run_script_or_module(args)

        if report is None:
            if not args.save_data and not args.restore_data:
                print('daceprof: No DaCe program calls detected or no report file generated.')
        else:
            if args.output:  # Save report
                if args.csv:
                    save_as_csv(args, report)
                else:
                    shutil.copyfile(report.filepath, args.output)
            else:  # Print report
                if report:
                    print('daceprof: Report file saved at', os.path.abspath(report.filepath))
                print_report(args, report)

        # Forward error code from internal application
        if errcode:
            exit(errcode)

    else:  # Input file given, print report and exit
        print_report(args, args.input)


if __name__ == '__main__':
    main()
