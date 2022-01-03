import argparse
from dace.codegen.instrumentation.report import InstrumentationReport
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path', help='Path to the file containing the report')
    parser.add_argument('--sort',
                        '-s',
                        help='Sort by a specific criterion',
                        choices=('min', 'max', 'mean', 'median', 'counter',
                                 'value'))
    parser.add_argument('--ascending',
                        '-a',
                        help='Sort in ascending order',
                        action='store_true')

    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        print(path, 'does not exist or isn\'t a regular file, aborting.')
        exit(1)

    report = InstrumentationReport(path)
    if args.sort:
        report.sortby(args.sort, args.ascending)
    print(report)
