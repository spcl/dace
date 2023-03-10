import csv
import sys
import numpy as np
from argparse import ArgumentParser
from typing import List, Dict, Callable, Any, Optional
import json

class Data:
    kernel_name: str = None
    process_id: int = None
    durations_unit: str = None
    durations: List[float] = []
    cycles_unit: str = None
    cycles: List[int] = []

    def __init__(self):
        self.kernel_name = None
        self.process_id = None
        self.durations_unit = None
        self.durations = []
        self.cycles_unit = None
        self.cycles = []

    def add_row(self, row: List[str], header: Dict[str, int]) -> bool:
        """
        Adds the given row to this data object. If the kernel name and process id does not match the existing ones, will
        reject adding the data and return false. If if is the first row added, will take process id and kernel name from
        it

        :param row: Row with data to add
        :type row: list[str]
        :param header: Dictionary with keys being header names and value the index in row
        :type header: dict[str, int]
        :return: False if adding data was rejected, True otherwise
        :rtype: bool
        """
        kernel_name = row[header['Kernel Name']]
        process_id = row[header['Process ID']]
        if self.kernel_name is None and self.process_id is None:
            self.kernel_name = kernel_name
            self.process_id = process_id

        if self.kernel_name != kernel_name or self.process_id != process_id:
            return False
        else:
            if row[header['Metric Name']] == 'Elapsed Cycles':
                self._add_value(row, header, 'cycles', lambda x: int(x.replace(',', '')))
            if row[header['Metric Name']] == 'Duration':
                self._add_value(row, header, 'durations', lambda x: float(x.replace(',','')))
        return True

    def _add_value(self, row: List[str], header: Dict[str, int], value_name: str, parse: Callable[[str], Any]):
        """
        Private function to add one value/metric to the data

        :param row: Row with the data
        :type row: list[str]
        :param header: Dictionary with keys being header names and value the index in row
        :type header: dict[str, int]
        :param value_name: The name of the value to add. There must be an attribute with this name and one with
        <value_name>_unit in this class
        :type value_name: str
        :param parse: Function to call to parse the string into another type to be stored into the list
        :type parse: Callable[[str], Any]
        """
        unit = row[header['Metric Unit']]
        if getattr(self, f"{value_name}_unit", None) is None:
            setattr(self, f"{value_name}_unit", unit)
        elif unit != getattr(self, f"{value_name}_unit", None):
            return False
        getattr(self, value_name).append(parse(row[header['Metric Value']]))

    
    def get_stat(self) -> str:
        return f"duration [{self.durations_unit}](#={len(self.durations)}): min: {min(self.durations)}, max: {max(self.durations)}, " \
               f"avg: {np.average(self.durations)}, median: {np.median(self.durations)}\n"\
               f"cycles   [{self.cycles_unit}](#={len(self.cycles)}): min: {min(self.cycles)}, max: {max(self.cycles)}, " \
               f"avg: {np.average(self.cycles)}, median: {np.median(self.cycles)}"


def read_csv(in_stream, out_file: Optional[str]=None):
    reader = csv.reader(in_stream)
    data = [Data()]
    # create dict where key is header name and value the index/column of it
    header = {}
    for index, key in enumerate(next(reader)):
        header[key] = index

    for row in reader:
        if row[header['Metric Name']] in ['Duration', 'Elapsed Cycles']:
            # if row was rejected it was from a different kernel, add thus a new data object to the ilst
            if not data[-1].add_row(row, header):
                data.append(Data())
            assert data[-1].add_row(row, header)

    if out_file is not None:
        print(f"Write output into {out_file}")
        all_data = {}
        for d in data:
            all_data[d.kernel_name] = d.__dict__
        with open(out_file, mode='w') as file:
            json.dump(all_data, file)
    else:
        for d in data:
            print(d.kernel_name)
            print(d.get_stat())
    



def main():
    parser = ArgumentParser()
    parser.add_argument(
            '-f', '--file',
            type=str,
            help='The csv file to read')
    parser.add_argument(
            '-o', '--output',
            type=str,
            help='The output file, format is json')

    args = parser.parse_args()

    
    if args.file is None:
        read_csv(sys.stdin, args.output)
    else:
        # TODO: Read from stdin via sys.stdin
        with open(args.file) as csvfile:
            read_csv(csvfile, args.output)


if __name__ == '__main__':
    main()
