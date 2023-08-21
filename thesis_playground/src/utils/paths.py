from typing import List, Optional
from os.path import join, dirname, split, abspath, exists
from os import getcwd, listdir, makedirs


def create_if_not_exist(path: str) -> str:
    """
    Creates the folders in the given path and returns the path itself.

    :param path: The path
    :type path: str
    :return: The given path
    :rtype: str
    """
    if not exists(path):
        makedirs(path)
    return path


def get_results_dir(folder_name: str = 'results') -> str:
    """
    Returns path to the directory where the results are stored

    :param folder_name: Name of the folder
    :type folder_name: str
    :return: Path to the directory
    :rtype: str
    """
    return join(get_thesis_playground_root_dir(), folder_name)


def get_dacecache() -> str:
    """
    Returns path to the currently used dacecache folder

    :return: The path to the .dacecache folder
    :rtype: str
    """
    return join(getcwd(), '.dacecache')


def get_default_sdfg_file(program: str) -> str:
    """
    Returns the path to the (default) sdfg file of the given program stored in the current .dacecache folder.
    Assumes that it has been generated before.

    :param program: The name of the program
    :type program: str
    :return: The path to the sdfg file
    :rtype: str
    """
    from utils.general import get_programs_data
    programs_data = get_programs_data()
    return join(get_dacecache(), f"{programs_data['programs'][program]}_routine", "program.sdfg")


def get_thesis_playground_root_dir() -> str:
    """
    Returns path to the thesis_playground folder in the dace repo. Should serve as root folder for most other paths

    :return: absoulte path to folder
    :rtype: str
    """
    return split(split(dirname(abspath(__file__)))[0])[0]


def get_complete_results_dir() -> str:
    """
    Get Path to the directory of the complete results

    :return: Path to dir
    :rtype: str
    """
    return join(get_thesis_playground_root_dir(), 'complete_results')


def get_list_of_complete_results_name() -> List[str]:
    """
    Lists all folder with data (which contain an info.json) in the complete results folder.

    :return: List of names/folders inside the complete results folder (no paths given)
    :rtype: List[str]
    """
    names = []
    for name in listdir(get_complete_results_dir()):
        if exists(join(get_complete_results_dir(), name, 'info.json')):
            names.append(name)
    return names


def get_verbose_graphs_dir() -> str:
    """
    Gets path to the directory where all the SDFGs are stored when SDGFs are generated verbosly (e.g. not for compiling)

    :return: Path to the dir
    :rtype: str
    """
    return join(get_thesis_playground_root_dir(), 'sdfg_graphs')


def get_vert_loops_dir() -> str:
    """
    Returns path to the vert_loop_results directory

    :return: Path
    :rtype: str
    """
    return join(get_thesis_playground_root_dir(), 'vert_loop_results')


def get_playground_results_dir() -> str:
    """
    Get path to directory where results from playground are saved

    :return: Path
    :rtype: str
    """
    return join(get_thesis_playground_root_dir(), 'playground_results')


def get_results_2_folder() -> str:
    return create_if_not_exist(join(get_thesis_playground_root_dir(), 'results_v2'))


def get_experiments_2_file() -> str:
    return join(get_results_2_folder(), 'experiments.csv')


def get_plots_2_folder() -> str:
    return create_if_not_exist(join(get_thesis_playground_root_dir(), 'plots_v2'))


def get_sdfg_gen_code_folder() -> str:
    return create_if_not_exist(join(get_thesis_playground_root_dir(), 'sdfg_gen_code'))


def get_results_2_logdir(node: Optional[str] = None, profile_name: Optional[str] = None) -> str:
    """
    Get path to logfiles given node and profile name. If node or profile name is not given returns path to logfiles of
    all nodes or all profiles of node

    :param node: Name of the node, optional
    :type node: Optional[str]
    :param profile_name: Name of the profile, optional
    :type profile_name: Optional[str]
    :return: Path
    :rtype: str
    """
    logdir = join(get_results_2_folder(), "logs")
    if node is not None:
        if profile_name is not None:
            return create_if_not_exist(join(logdir, node, profile_name))
        return create_if_not_exist(join(logdir, node))
    return create_if_not_exist(logdir)
