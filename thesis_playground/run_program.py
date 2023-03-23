from argparse import ArgumentParser
import os
from subprocess import run

import dace

from utils import use_cache
from execute_utils import run_program


def main():
    parser = ArgumentParser(description="Run a single program on the GPU")
    parser.add_argument("program", help="Name of the program to run")
    parser.add_argument('--normalize-memlets', action='store_true', default=False)
    parser.add_argument('--cache', action='store_true', default=False, help="Use the cached generated code")
    parser.add_argument('-r', '--repetitions', type=int, default=1, help="Number of repetitions to run")

    args = parser.parse_args()

    if args.cache:
        if not use_cache(args.program):
            return 1

    run_program(args.program, repetitions=args.repetitions, device=dace.DeviceType.GPU,
                normalize_memlets=args.normalize_memlets)


if __name__ == '__main__':
    main()
