import argparse
from dace.codegen.instrumentation.report import InstrumentationReport
import os


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('path',
                        help='Path to the file containing the report')

    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.isfile(path):
        print(path, 'does not exist or isn\'t a regular file, aborting.')
        exit(1)

    report = InstrumentationReport(path)
    print(report)