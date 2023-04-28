from argparse import ArgumentParser
import os
import json
from subprocess import run


def main():
    parser = ArgumentParser()
    parser.add_argument('name')
    args = parser.parse_args()

    results_directory = 'complete_results'
    run_path = os.path.join(os.path.dirname(__file__), 'run.py')
    this_dir = os.path.join(results_directory, args.name)

    with open(os.path.join(this_dir, "info.json")) as info_file:
        info_data = json.load(info_file)
        raw_data_dir = os.path.join(this_dir, 'raw_data')
        if not os.path.exists(raw_data_dir):
            os.mkdir(raw_data_dir)

        for class_number in info_data['classes']:
            run(['python3', run_path, '--class', str(class_number), '--ncu-report', '--roofline',
                 '--ncu-report-folder', raw_data_dir, '--results-folder', raw_data_dir,
                 '--output', f"class_{class_number}.json",
                 *info_data['additional_run_flags']])


if __name__ == '__main__':
    main()
