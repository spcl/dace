from argparse import ArgumentParser

from execute.parameters import ParametersProvider
from utils.python_kernels_mapping import get_function_by_name
from utils.python import generate_sdfg


def action(args):
    symbols = ParametersProvider('cloudsc_vert_loop_7').get_dict()
    symbols.update({'NBLOCKS': 100000})
    sdfg = generate_sdfg(get_function_by_name(args.program), symbols, True, True, args.use_dace_auto_opt)
    if args.gen_code:
        sdfg.compile()
        print(f"Generated code is at: {sdfg.build_folder}")


def main():
    parser = ArgumentParser()
    parser.add_argument('program', type=str)
    parser.add_argument('--gen-code', action='store_true', help='Generate CUDA code')
    parser.add_argument('--use-dace-auto-opt', action='store_true', help='Use default DaCe auto opt')
    parser.set_defaults(func=action)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
