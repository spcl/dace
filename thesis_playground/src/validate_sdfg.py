from argparse import ArgumentParser
import dace


def main():
    parser = ArgumentParser(description="Load a SDFG from a given path and validate it")
    parser.add_argument('sdfg_path', help="Path to the SDFG to load")
    args = parser.parse_args()

    sdfg = dace.sdfg.sdfg.SDFG.from_file(args.sdfg_path)
    print(sdfg.name)
    print(sdfg.validate())
    print(dace.sdfg.validation.validate_sdfg(sdfg))


if __name__ == '__main__':
    main()
