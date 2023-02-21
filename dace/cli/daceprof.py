# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
A command-line tool that provides performance measurements and analysis on
Python scripts, modules, or existing instrumentation report files.
"""
import argparse
import runpy
import sys
import os
import shutil
from typing import Callable, List, Optional, Tuple, Union

import dace
from dace.codegen.instrumentation.report import InstrumentationReport
from dace import dtypes

ExitCode = Union[int, str]


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
                       default=100)
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
    group.add_argument('--sequential', help='Disable CPU multi-threading in code generation', action='store_true')

    # Data instrumentation
    group = parser.add_argument_group('data instrumentation arguments')
    group.add_argument('--save-data',
                       '-ds',
                       help='Enable data instrumentation and store all (or filtered) arrays',
                       action='store_true')
    group.add_argument('--restore-data',
                       '-dr',
                       help='Reproducibly run code by restoreing all (or filtered) arrays',
                       action='store_true')

    # Filtering arguments
    group = parser.add_argument_group('filtering arguments')
    group.add_argument('--interactive',
                       '-I',
                       help='Shows interactive prompts about which parts of the program to profile',
                       action='store_true')
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

    return parser, parser.parse_args()


def validate_arguments(args: argparse.Namespace) -> Optional[str]:
    if not args.input and not args.file:
        return 'A script or module is required'
    if args.module and args.input:
        return 'Choose either script, module, or input file.'
    if args.input and (args.file or args.args):
        return 'Input file mode cannot specify additional arguments.'
    if args.input and args.output:
        return 'Cannot load and save a report at the same time.'
    if args.save_data and args.restore_data:
        return 'Choose either saving data or restoring it.'

    return None


def run_script_or_module(args: argparse.Namespace) -> Tuple[Optional[str], ExitCode]:
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
    with dace.profile(repetitions=args.repetitions) as profiler:
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

    # TODO: Get instrumentation report file

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
                    if sched == dace.ScheduleType.CPU_Multicore or sched == dace.ScheduleType.Default:
                        n.schedule = dace.ScheduleType.Sequential

        registered.append(dace.hooks.register_sdfg_call_hook(before_hook=make_sequential))
    

    return registered


def print_report(args: argparse.Namespace, reportfile: str):
    path = os.path.abspath(reportfile)
    if not os.path.isfile(path):
        print(path, 'does not exist, aborting.')
        exit(1)

    report = InstrumentationReport(path)
    if args.sort:
        report.sortby(args.sort, args.ascending)

    if args.csv:
        print(report.as_csv())
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
        reportfile, errcode = run_script_or_module(args)

        if reportfile is None:
            print('daceprof: No DaCe program calls detected or no report file generated.')
        else:
            if args.output:  # Save report
                shutil.copyfile(reportfile, args.output)
            else:  # Print report
                print('daceprof: Report file saved at', os.path.abspath(reportfile))
                print_report(args, reportfile)

        # Forward error code from internal application
        if errcode:
            exit(errcode)

    else:  # Input file given, print report and exit
        print_report(args, args.input)


if __name__ == '__main__':
    main()
