from argparse import ArgumentParser
from subprocess import run
import csv
from tabulate import tabulate
import copy


def main():
    parser = ArgumentParser(description='Prints all available metric names/keys of the given ncu report')
    parser.add_argument('--sections', nargs="+", default=None)
    parser.add_argument('--values', action='store_true', default=False, help="Also print values")
    parser.add_argument('file')

    args = parser.parse_args()

    sections_output = run(['ncu', '--list-sections'], capture_output=True)
    sections = []
    for row in sections_output.stdout.decode('UTF-8').split('\n')[3:-1]:
        sections.append(row.split(' ')[0])

    # Go over all sections separately
    data = {}
    for section in sections:
        csv_stdout = run(['ncu', '--import', args.file, '--csv', '--section', section, '--page',
                          'details', '--details-all'], capture_output=True)

        reader = csv.reader(csv_stdout.stdout.decode('UTF-8').split('\n')[:-1])
        # create dict where key is header name and value the index/column of it
        header = {}
        for index, key in enumerate(next(reader)):
            header[key] = index

        if list(header)[0].startswith('==WARNING=='):
            print(f"Skip section {section}")
            continue
        # extract the available keys
        keys = set()
        values = {}
        for row in reader:
            key = row[header['Metric Name']]
            keys.add(key)
            values[key] = (row[header['Metric Value']], row[header['Metric Unit']])
        data[section] = {}
        data[section]['keys'] = keys
        data[section]['values'] = values

    flat_data = []
    headers = []

    print_sections = []
    if args.sections is None:
        print_sections = list(data['keys'].keys())
    else:
        print_sections = args.sections

    for section in print_sections:
        headers.append(section)
        if args.values:
            headers.extend(['Value', 'Unit'])
    sections_left = copy.deepcopy(print_sections)
    while len(sections_left) > 0:
        row = []
        for section in print_sections:
            if len(data[section]['keys']) > 0:
                key = data[section]['keys'].pop()
                row.append(key)
                if args.values:
                    row.extend([*data[section]['values'][key]])
            elif section in sections_left:
                del sections_left[sections_left.index(section)]
        flat_data.append(row)

    print(tabulate(flat_data, headers=headers))


if __name__ == '__main__':
    main()
