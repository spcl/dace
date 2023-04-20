import sys
sys.path.insert(0, '/apps/ault/spack/opt/spack/linux-centos8-zen/gcc-8.4.1/cuda-11.8.0-fjdnxm6yggxxp75sb62xrxxmeg4s24ml/nsight-compute-2022.3.0/extras/python/')
import ncu_report


def get_action(filename: str):
    """
    Gets the action inside the given ncu-rep file. Assumes there is only one range and action

    :param filename: The path/filename of the ncu-report
    :type filename: str
    """
    my_context = ncu_report.load_report(filename)
    num_ranges = my_context.num_ranges()
    if num_ranges > 1:
        print(f"WARNING: More than one range found in {filename} only taking the first")
    my_range = my_context.range_by_idx(0)
    num_actions = my_range.num_actions()
    if num_actions > 1:
        print(f"WARNING: More than one action found in {filename} only taking the first")
    my_action = my_range.action_by_idx(0)
    return my_action
