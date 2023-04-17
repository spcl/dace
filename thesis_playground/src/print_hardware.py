from argparse import ArgumentParser
import json
from tabulate import tabulate


def main():
    parser = ArgumentParser(description="Print Hardware stats of a given gpu")
    parser.add_argument('GPU', type=str, help="Name of the GPU")
    args = parser.parse_args()

    hardware_filename = 'nodes.json'
    flat_data = []
    with open(hardware_filename) as hardware_file:
        nodes_data = json.load(hardware_file)
        hardware_data = nodes_data['GPUs'][args.GPU]
        max_p_s = hardware_data['flop_per_second']
        global_memory = hardware_data['bytes_per_second']['global']
        shared_memory = hardware_data['bytes_per_second']['shared']
        l1_cache = hardware_data['bytes_per_cycle']['l1']
        l2_cache = hardware_data['bytes_per_second']['l2']
        gpu_clock = hardware_data['graphics_clock']
        flat_data = [
             ['Max bandwidth global', 'Theoretical', global_memory['theoretical'], 'bytes / second'],
             ['Max bandwidth global', 'Theoretical', global_memory['theoretical'] / gpu_clock, 'bytes / cycle'],
             ['Max bandwidth global', 'Measured', global_memory['measured'], 'bytes / second'],
             ['Max bandwidth global', 'Measured', global_memory['measured'] / gpu_clock, 'bytes / cycle'],
             ['Max bandwidth shared', 'Theoretical', shared_memory['theoretical'], 'bytes / second'],
             ['Max bandwidth shared', 'Theoretical', shared_memory['theoretical'] / gpu_clock, 'bytes / cycle'],
             ['Max bandwidth shared', 'Measured', shared_memory['measured'], 'bytes / second'],
             ['Max bandwidth shared', 'Measured', shared_memory['measured'] / gpu_clock, 'bytes / cycle'],
             ['Max bandwidth L1', 'Theoretical', l1_cache['theoretical'] * gpu_clock, 'bytes / second'],
             ['Max bandwidth L1', 'Theoretical', l1_cache['theoretical'], 'bytes / cycle'],
             ['Max bandwidth L1', 'Measured', l1_cache['measured'] * gpu_clock, 'bytes / second'],
             ['Max bandwidth L1', 'Measured', l1_cache['measured'], 'bytes / cycle'],
             ['Max bandwidth L2', 'Measured', l2_cache['measured'], 'bytes / second'],
             ['Max bandwidth L2', 'Measured', l2_cache['measured'] / gpu_clock, 'bytes / cycle'],
             ['Max double performance', 'Theoretical', max_p_s['theoretical'], 'bytes / second'],
             ['Max double performance', 'Theoretical', max_p_s['theoretical'] / gpu_clock, 'bytes / cycle'],
             ['Peak intensity', 'Theo. & global', max_p_s['theoretical'] / global_memory['theoretical'], 'flop / byte'],
             # TODO: What is about the intensity using the different clock rates?
                ]

    headers = ['Name', 'Obtained by', 'Value', 'Unit']
    print(tabulate(flat_data, headers=headers))


if __name__ == '__main__':
    main()
