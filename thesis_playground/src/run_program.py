from argparse import ArgumentParser

from utils.paths import get_default_sdfg_file
from utils.general import use_cache, enable_debug_flags
from utils.execute_dace import RunConfig, run_program, test_program
from execute.parameters import ParametersProvider


def main():
    parser = ArgumentParser(description="Run a single program on the GPU")
    parser.add_argument("program", help="Name of the program to run")
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
    parser.add_argument('--read-sdfg', action='store_true')
    parser.add_argument('--specialize-symbols', action='store_true', default=False)

    args = parser.parse_args()
    run_config = RunConfig()
    run_config.set_from_args(args)

    if args.debug:
        enable_debug_flags()

    if args.cache:
        if not use_cache(args.program):
            return 1

    if args.only_test:
        additional_args = {}
        if args.read_sdfg:
            additional_args['sdfg_file'] = get_default_sdfg_file(args.program)
        test_program(args.program, run_config, **additional_args)
    else:
        params = ParametersProvider(args.program)
        params.update_from_args(args)
        additional_args = {}
        if args.read_sdfg:
            additional_args['sdfg_file'] = get_default_sdfg_file(args.program)
        if args.specialize_symbols:
            additional_args['specialize_symbols'] = True

        run_program(args.program, run_config, params, repetitions=args.repetitions, **additional_args)


if __name__ == '__main__':
    main()
