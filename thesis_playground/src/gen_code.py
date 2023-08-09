from argparse import ArgumentParser
import dace
import os

from utils.paths import get_sdfg_gen_code_folder


def main():
    parser = ArgumentParser(description="Generate the CUDA code from a given SDFG")
    parser.add_argument('sdfg_path', type=str, help='Path to SDFG')
    args = parser.parse_args()

    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_path)
    csdfg = sdfg.compile()
    print(csdfg.filename)
    for code_object in sdfg.generate_code():
        filename = os.path.join(get_sdfg_gen_code_folder(), args.sdfg_path.split('.')[0],
                                f"{code_object.name}.{code_object.language}")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as file:
            file.write(code_object.clean_code)
            print(f"Saved code into {filename}")

    print(f"Signature of SDFG: {sdfg.signature()}")


if __name__ == '__main__':
    main()
