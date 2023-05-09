from argparse import ArgumentParser
from inspect import getmembers, isclass

import scripts


def main():
    parser = ArgumentParser(description="Collection of scripts to run/plot/print predefined stuff. Each script is a"
                            "subcommand")
    subparsers = parser.add_subparsers(
        title="Scripts",
        help="See the help of the respective script")

    for name, object in getmembers(scripts):
        if isclass(object) and name != 'Script':
            object(subparsers)

    args = parser.parse_args()

    if "func" in args:
        args.func(args)
    else:
        print("ERROR: A script must be specified by using the respective subcommand")
        parser.print_help()


if __name__ == '__main__':
    main()
