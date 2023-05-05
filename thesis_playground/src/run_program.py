from argparse import ArgumentParser

import dace
import os

from utils.general import use_cache, enable_debug_flags
from utils.execute_dace import RunConfig, run_program, test_program
from data import ParametersProvider


def main():
    parser = ArgumentParser(description="Run a single program on the GPU")
    parser.add_argument("program", help="Name of the program to run")
    parser.add_argument('--normalize-memlets', action='store_true', default=False)
    parser.add_argument('--cache', action='store_true', default=False, help="Use the cached generated code")
    parser.add_argument('-r', '--repetitions', type=int, default=1, help="Number of repetitions to run")
    parser.add_argument('--only-test', action='store_true', default=False, help="Only test the program")
    parser.add_argument('--debug', action='store_true', default=False, help="Configure for debug build")
    parser.add_argument('--use-dace-auto-opt', default=False, action='store_true',
                        help='Use DaCes auto_opt instead of mine')
    parser.add_argument('--pattern', choices=['const', 'formula', 'worst'], type=str, default=None,
                        help='Pattern for in and output')
    parser.add_argument('--NBLOCKS', type=int)
    parser.add_argument('--KLEV', type=int)
    parser.add_argument('--KLON', type=int)

    args = parser.parse_args()
    run_config = RunConfig()
    run_config.set_from_args(args)

    if args.debug:
        enable_debug_flags()

    if args.cache:
        if not use_cache(args.program):
            return 1

    if args.only_test:
        test_program(args.program, run_config)
    else:
        params = ParametersProvider(args.program)
        params.update_from_args(args)
        print(params)
        run_program(args.program, run_config, params, repetitions=args.repetitions)


if __name__ == '__main__':
    main()
