from argparse import ArgumentParser

import dace

from utils import get_programs_data
from test import compile_for_profile


def main():
    parser = ArgumentParser()
    parser.add_argument(
        '-p', '--programs',
        type=str,
        nargs='+',
        help='Names of the programs to use. Can be several separated by space')
    parser.add_argument(
            '-c', '--class',
            type=int,
            choices=[1, 2, 3],
            default=None,
            dest='kernel_class',
            help="Run all programs of a given class")

    args = parser.parse_args()

    programs = get_programs_data()['programs']
    selected_programs = [] if args.programs is None else args.programs
    if args.kernel_class is not None:
        selected_programs = [p for p in programs if p.startswith(f"cloudsc_class{args.kernel_class}")]

    if len(selected_programs) == 0:
        print("ERRROR: Need to specify programs either with --programs or --class")
        return 1

    for program in selected_programs:
        print(f"Compile {program} for GPU without normalising memlets")
        compile_for_profile(program, dace.DeviceType.GPU, False)


if __name__ == '__main__':
    main()
