from subprocess import run
from math import sqrt, ceil
import os
import logging
from typing import Optional, List, Dict, Union, Tuple
import pandas as pd
from numbers import Number
import seaborn as sns
import dace
from dace.sdfg import SDFG
from dace import nodes

from utils.paths import get_dacecache, get_full_cloudsc_log_dir, get_full_cloudsc_results_dir, get_full_cloudsc_plot_dir
from utils.general import enable_debug_flags, remove_build_folder
from utils.run_config import RunConfig
from utils.data_analysis import compute_speedups
from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines_dict, \
                       replace_legend_names, legend_on_lines_dict, get_node_gpu_map, get_arrowprops, rotate_xlabels


logger = logging.getLogger(__name__)


opt_levels = {
    "baseline": {
        "run_config": RunConfig(k_caching=False, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "baseline"
        },
    "k-caching": {
        "run_config": RunConfig(k_caching=True, change_stride=False, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "k_caching"
        },
    "change-strides": {
        "run_config": RunConfig(k_caching=False, change_stride=True, outside_loop_first=False,
                                move_assignment_outside=False, full_cloudsc_fixes=True),
        "name": "change_strides"
        },
    "all": {
        "run_config": RunConfig(k_caching=True, change_stride=True, outside_loop_first=True, full_cloudsc_fixes=True),
        "name": "all_opt"
        },
    "all-custom": {
        "run_config": RunConfig(k_caching=True, change_stride=True, outside_loop_first=True, full_cloudsc_fixes=True),
        "name": "all_opt_custom"
        }
}


def get_program_name(version: int) -> str:
    if version == 3:
        program = 'cloudscexp3'
    elif version == 4:
        program = 'cloudscexp4'
    else:
        program = 'cloudscexp2'
    return program


def get_build_folder_name(version: int) -> str:
    if version == 3:
        program = 'CLOUDSCOUTER3'
    elif version == 4:
        program = 'CLOUDSCOUTER4'
    else:
        program = 'CLOUDSCOUTER2'
    return program


def add_synchronize(dacecache_folder: str):
    """
    Adds a cudaDeviceSynchronize and cudaStreamSynchronize before each time measurement start or end. A time measurement
    is a std::chrono::high_resoltion_clock::now() call. Only does this inside the cpu file.

    :param dacecache_folder: The folde in .dacecache where the 'src' folder with the source files lies.
    :type dacecache_folder: str
    """
    src_dir = os.path.join(get_dacecache(), dacecache_folder, 'src', 'cpu')
    src_file = os.path.join(src_dir, os.listdir(src_dir)[0])
    if len(os.listdir(src_dir)) > 1:
        logger.warning(f"More than one files in {src_dir}")
    lines = run(['grep', '-rn', 'std::chrono::high_resolution_clock::now', src_file], capture_output=True).stdout.decode('UTF-8')
    logger.debug("Add synchronizes to %s", src_file)
    line_numbers_to_insert = []

    for line in lines.split('\n'):
        if len(line) > 0:
            if line.split()[2].startswith('__dace_t'):
                line_numbers_to_insert.append(int(line.split(':')[0])-1)

    logger.debug("Insert synchronize into line numbers: %s", line_numbers_to_insert)
    with open(src_file, 'r') as f:
        contents = f.readlines()
        for offset, line_number in enumerate(line_numbers_to_insert):
            contents.insert(line_number+offset,
                    "DACE_GPU_CHECK(cudaDeviceSynchronize());DACE_GPU_CHECK(cudaStreamSynchronize(__state->gpu_context->streams[0]));\n")

    with open(src_file, 'w') as f:
        contents = "".join(contents)
        f.write(contents)


def instrument_sdfg(sdfg: SDFG, opt_level: str, device: str):
    """
    Instruments the given SDFG. Uses the opt level to decide on what to instrument.

    :param sdfg: The SDFG to instrument. Does it in-place
    :type sdfg: SDFG
    :param opt_level: The opt-level, used to decide what to instrument
    :type opt_level: str
    :param device: The device for which the SDFG was generated.
    :type device: str
    """
    if opt_level in ['k-caching', 'baseline']:
        if device == 'CPU':
            cloudsc_state = sdfg.find_state('stateCLOUDSC')
        elif device == 'GPU':
            cloudsc_state = sdfg.find_state('CLOUDSCOUTER4_copyin')
        nblocks_maps = [n for n in cloudsc_state.nodes() if isinstance(n, nodes.MapEntry) and n.label ==
                        'stateCLOUDSC_map']
    if opt_level in ['all', 'change-strides']:
        changed_strides_state = sdfg.find_state('with_changed_strides')
        nsdfg_changed_strides = [n for n in changed_strides_state.nodes() if isinstance(n, nodes.NestedSDFG)][0]
        nblocks_maps = [n for n in nsdfg_changed_strides.sdfg.find_state('CLOUDSCOUTER4_copyin').nodes()
                        if isinstance(n, nodes.MapEntry) and n.label == 'stateCLOUDSC_map']
        sdfg.find_state('transform_data').instrument = dace.InstrumentationType.Timer
        sdfg.find_state('transform_data_back').instrument = dace.InstrumentationType.Timer

    if len(nblocks_maps) > 1:
        logger.warning("There is more than one map to instrument: %s", nblocks_maps)
    nblocks_maps[0].instrument = dace.InstrumentationType.Timer


def get_sdfg(opt_level: str, device: str, version: int) -> SDFG:
    """
    Loads the generated SDFG from memory

    :param opt_level: The optimisation level
    :type opt_level: str
    :param device: The device for which the SDFG was generated
    :type device: str
    :param version: The version used
    :type version: int
    :return: The loaded SDFG
    :rtype: SDFG
    """
    program = get_program_name(version)
    verbose_name = f"{program}_{opt_levels[opt_level]['name']}"
    sdfg_file = os.path.join(get_full_cloudsc_log_dir(), f"{verbose_name}_{device.lower()}.sdfg")
    return dace.sdfg.sdfg.SDFG.from_file(sdfg_file)


def compile_sdfg(sdfg: SDFG,
                 nblocks: int,
                 version: int,
                 opt_level: str,
                 device: str,
                 instrument: bool = True,
                 debug: bool = False,
                 build_dir: Optional[str] = None):
    """
    Compile the given SDFG

    :param sdfg: The SDFG to compile
    :type sdfg: SDFG
    :param nblocks: The number of blocks to compile it for
    :type nblocks: int
    :param version: The version used
    :type version: int
    :param opt_level: The optimisation level used
    :type opt_level: str
    :param device: The device to SDFG is generated for
    :type device: str
    :param instrument: True if to instrument the SDFG, defaults to True
    :type instrument: bool, optional
    :param debug: True if to build with debug flags, defaults to False
    :type debug: bool, optional
    :param build_dir: The directory to generate the build folder in. If none will take the default in .dacecache,
                      defaults to None
    :type build_dir: Optional[str], optional
    """
    remove_build_folder(dacecache_folder=get_build_folder_name(version))
    program = get_program_name(version)
    for nsdfg in sdfg.sdfg_list:
        nsdfg.add_constant('NBLOCKS', nblocks)

    if debug:
        logger.info("Enable Debug Flags")
        enable_debug_flags()
    if build_dir is not None:
        sdfg.build_folder = build_dir
    else:
        sdfg.build_folder = os.path.abspath(sdfg.build_folder)
    logger.info("Build into %s", sdfg.build_folder)

    if instrument:
        instrument_sdfg(sdfg, opt_level, device)
    sdfg.validate()
    sdfg.compile()
    if instrument and device == 'GPU':
        add_synchronize(get_build_folder_name(version))
    signature_file = os.path.join(get_full_cloudsc_log_dir(), f"signature_dace_{program}.txt")
    logger.info("Write signature file into %s", signature_file)
    with open(signature_file, 'w') as file:
        file.write(sdfg.signature())


def get_experiment_list_df() -> pd.DataFrame:
    experiments_file = os.path.join(get_full_cloudsc_results_dir(), 'experiments.csv')
    if not os.path.exists(experiments_file):
        return pd.DataFrame()
    else:
        return pd.read_csv(experiments_file, index_col=['experiment id'])


def save_experiment_list_df(df: pd.DataFrame):
    experiments_file = os.path.join(get_full_cloudsc_results_dir(), 'experiments.csv')
    df.to_csv(experiments_file)


def read_reports(sdfg: SDFG) -> List[Dict[str, Number]]:
    reports = sdfg.get_instrumentation_reports()
    data = []
    for index, report in enumerate(reports):
        for entry in report.durations.values():
            for key in entry:
                data.append({'scope': key, 'runtime': list(entry[key].values())[0][0], 'run': index})
    return data


def run_cloudsc_cuda(
        executable_name: str,
        data_name: str,
        size: int,
        repetitions: int) -> List[Dict[str, Union[str, Number]]]:
    """
    Runs the cloudsc CUDA version and returns runtime without data movement

    :param executable_name: Name of the executable
    :type executable_name: str
    :param data_name: Name to give in the 'opt level' entry in the returned data
    :type data_name: str
    :param size: Size to pass to the executable
    :type size: int
    :param repetitions: Number of repetitions to run
    :type repetitions: int
    :return: List with one dictionary per run listing runtime and other metadata
    :rtype: List[Dict[str, Union[str, Number]]]
    """
    data = []
    logger.debug('Run %s using %s for size %s repeating it %i time', data_name, executable_name, size, repetitions)
    for i in range(repetitions):
        cloudsc_output = run([f"./bin/{executable_name}", '1', str(size), '128'],
                             cwd='/users/msamuel/dwarf-p-cloudsc-original/build_cuda',
                             capture_output=True)
        if cloudsc_output.returncode == 0:
            for line in cloudsc_output.stdout.decode('UTF-8').split('\n'):
                if 'core' in line:
                    data.append({
                        'scope': 'Map stateCLOUDSC_map',
                        'opt level': data_name,
                        'nblocks': size,
                        'runtime': line.split()[7],
                        'run': i
                        })
                    logger.debug('Results line: %s', line)
        else:
            logger.warning('Running cuda cloudsc failed')
            logger.warning('stderr: %s', cloudsc_output.stderr.decode('UTF-8'))
    return data


def get_data(experiment_id: int) -> Dict[str, Union[str, pd.DataFrame]]:
    """
    Returns the data for the given experiment id. Returns a dictionary with the following entries:
        - gpu: The GPU used. Is a string
        - node: The node used. Is a string
        - data: The measurement data. Is a pandas DataFrame
        - avg_data: The measurement data averaged over all runs. Is a pandas DataFrame
        - run_count: The number of runs is an int

    :param experiment_id: The experiment id
    :type experiment_id: int
    :return: Dictionary with all data
    :rtype: Dict[str, Union[str, pd.DataFrame]]
    """
    index_cols = ['run', 'scope', 'opt level', 'nblocks']
    experiment_list_df = get_experiment_list_df()
    node = experiment_list_df.loc[[int(experiment_id)]]['node'].values[0]
    gpu = get_node_gpu_map()[node]
    data = pd.read_csv(os.path.join(get_full_cloudsc_results_dir(node, experiment_id),
                                    'results.csv')).set_index(index_cols)
    if 'Unnamed: 0' in data.columns:
        data.drop('Unnamed: 0', axis='columns', inplace=True)
    dace_opt_levels = ['all', 'k-caching', 'change-strides', 'baseline']
    # data = data.reset_index().assign(version=lambda x: 'DaCe' if x['opt level'] in dace_opt_levels else 'Cloudsc')
    data.reset_index(inplace=True)
    data['version'] = data.apply(lambda x: 'DaCe' if x['opt level'] in dace_opt_levels else 'Cloudsc', axis=1)
    index_cols.append('version')
    data['scope'] = data['scope'].map({
        'State transform_data': 'transform',
        'State transform_data_back': 'transform back',
        'Map stateCLOUDSC_map': 'work'
        })
    data.set_index(index_cols, inplace=True)

    index_cols.remove('run')
    run_counts = data.groupby(index_cols).count()['runtime']
    run_count = run_counts.min()
    if run_counts.max() != run_count:
        logger.warning("The number of run counts is not equal, ranges from %s to %s", run_count, run_count.max())
        logger.warning("%s", run_counts)

    avg_data = data.groupby(index_cols).mean()

    return {'gpu': gpu, 'node': node, 'data': data, 'avg_data': avg_data, 'run_count': run_count}


def plot_bars(experiment_id: int):
    data_dict = get_data(experiment_id)
    avg_data = data_dict['avg_data']
    data = data_dict['data']
    figure = get_new_figure()
    figure.suptitle(f"Runtimes on the full cloudsc code on {data_dict['node']} using a NVIDIA {data_dict['node']} GPU "
                    f"averaging over {data_dict['run_count']} runs")

    sizes = avg_data.reset_index()['nblocks'].unique()
    nrows = 2
    ncols = ceil((len(sizes) / nrows))
    axes = figure.subplots(nrows, ncols, sharex=True)
    axes_1d = [a for axs in axes for a in axs]
    for idx, (ax, size) in enumerate(zip(axes_1d, sizes)):
        ax.set_title(size)
        sns.barplot(data.xs((size, 'work', 'DaCe'), level=('nblocks', 'scope', 'version')).reset_index(),
                    x='opt level', y='runtime', ax=ax, order=['baseline', 'k-caching', 'change-strides', 'all'])
        rotate_xlabels(ax)
        ax.set_xlabel('')
        if idx % ncols == 0:
            ax.set_ylabel('Runtime [ms]')
        else:
            ax.set_ylabel('')
        rotate_xlabels(ax, replace_dict={'baseline': 'No optimisations', 'k-caching': 'K-caching',
                                         'change-strides': 'Changed array order', 'all': 'Both'})

    figure.tight_layout()
    save_plot(os.path.join(get_full_cloudsc_plot_dir(data_dict['node']), 'runtime_bar.pdf'))

    this_data = pd.concat([
        avg_data.xs('Cloudsc CUDA', level='opt level', drop_level=False),
        avg_data.xs('Cloudsc CUDA K-caching', level='opt level', drop_level=False),
        avg_data.xs('all', level='opt level', drop_level=False),
        avg_data.xs('change-strides', level='opt level', drop_level=False),
        ])
    figure = get_new_figure()
    figure.suptitle(f"Runtimes on the full cloudsc code on {data_dict['node']} using a NVIDIA {data_dict['node']} GPU "
                    f"averaging over {data_dict['run_count']} runs")
    axes = figure.subplots(nrows, ncols, sharex=True)
    axes_1d = [a for axs in axes for a in axs]
    for idx, (ax, size) in enumerate(zip(axes_1d, sizes)):
        ax.set_title(size)
        sns.barplot(this_data.xs((size, 'work'), level=('nblocks', 'scope')).reset_index(),
                    x='opt level', y='runtime', ax=ax,
                    order=['Cloudsc CUDA', 'change-strides', 'Cloudsc CUDA K-caching', 'all'],
                    hue='version', dodge=False)
        ax.get_legend().remove()
        rotate_xlabels(ax)
        ax.set_xlabel('')
        if idx % ncols == 0:
            ax.set_ylabel('Runtime [ms]')
        else:
            ax.set_ylabel('')
        replace_legend_names(ax.get_legend(), {'baseline': 'No optimisations', 'k-caching': 'K-caching',
                                               'change-strides': 'Changed array order', 'all': 'Both'})

    figure.tight_layout()
    save_plot(os.path.join(get_full_cloudsc_plot_dir(data_dict['node']), 'runtime_bar_cuda.pdf'))


def plot_lines(experiment_id: int):
    data_dict = get_data(experiment_id)
    desc = f"Run on {data_dict['node']} using a NVIDIA {data_dict['gpu']} GPU averaging over {data_dict['run_count']} "\
           "runs"
    data = data_dict['data']
    speedups = compute_speedups(data_dict['avg_data'], ('baseline'), ('opt level'))

    opt_order = ['Cloudsc CUDA K-caching', 'Cloudsc CUDA', 'all', 'k-caching', 'change-strides', 'baseline']
    figure = get_new_figure()
    figure.suptitle("Runtimes on the full cloudsc code")
    ax = figure.add_subplot(1, 1, 1)
    size_vs_y_plot(ax, 'Runtime [ms]', desc, data, size_var_name='nblocks')
    sns.lineplot(data.xs('work', level='scope'), x='nblocks', y='runtime', hue='opt level',
                 marker='o', hue_order=opt_order)
    legend_on_lines_dict(ax, {
        'Cloudsc CUDA K-caching': {'position': (9e4, 12), 'color_index': 0, 'rotation': 5},
        'DaCe with K-caching & changed array order': {'position': (9e4, 21), 'color_index': 2, 'rotation': 5},
        'Cloudsc CUDA': {'position': (1e5, 42), 'color_index': 1, 'rotation': 8},
        'DaCe with changed array order': {'position': (1e5, 33), 'color_index': 4, 'rotation': 8},
        'DaCe with K-caching': {'position': (1e5, 97), 'color_index': 3, 'rotation': 26},
        'DaCe without any special optimisation': {'position': (1e5+100, 78), 'color_index': 5, 'rotation': 18},
        })
    save_plot(os.path.join(get_full_cloudsc_plot_dir(data_dict['node']), 'runtime.pdf'))

    figure = get_new_figure()
    figure.suptitle("Speedups on the full cloudsc code")
    ax = figure.add_subplot(1, 1, 1)
    size_vs_y_plot(ax, 'Speedup', desc, speedups, size_var_name='nblocks')
    sns.lineplot(speedups.xs(('work', 'DaCe'), level=('scope', 'version')).drop('baseline', level='opt level'),
                 x='nblocks', y='runtime', hue='opt level',
                 marker='o', hue_order=opt_order[2:])
    legend_on_lines_dict(ax, {
        'DaCe with K-caching & changed array order': {'position': (9.7e4, 3.8), 'color_index': 0, 'rotation': -3},
        'DaCe with changed array order': {'position': (1e5, 2.1), 'color_index': 2, 'rotation': 2},
        'DaCe with K-caching': {'position': (6e4, 0.8), 'color_index': 1, 'rotation': -4},
        })
    save_plot(os.path.join(get_full_cloudsc_plot_dir(data_dict['node']), 'speedup.pdf'))
