from argparse import ArgumentParser
import logging
from datetime import datetime
import os
from subprocess import run, check_output
import pandas as pd
import seaborn as sns
from distutils.dir_util import copy_tree

from utils.log import setup_logging
from utils.full_cloudsc import get_sdfg, compile_sdfg, get_experiment_list_df, save_experiment_list_df, read_reports, \
                               run_cloudsc_cuda
from utils.paths import get_full_cloudsc_results_dir, get_thesis_playground_root_dir, get_full_cloudsc_plot_dir

from utils.plot import save_plot, get_new_figure, size_vs_y_plot, get_bytes_formatter, legend_on_lines, \
                       replace_legend_names, legend_on_lines_dict

logger = logging.getLogger(__name__)


def action_profile(args):
    version = 4
    opt_levels = ['all', 'k-caching', 'change-strides', 'baseline']
    sizes = [2**13, 2**14, 2**15, 2**16]
    experiment_list_df = get_experiment_list_df()
    if 'experiment id' in experiment_list_df.reset_index().columns and len(experiment_list_df.index) > 0:
        new_experiment_id = experiment_list_df.reset_index()['experiment id'].max() + 1
    else:
        new_experiment_id = 0
    node = check_output(['uname', '-a']).decode('UTF-8').split(' ')[1].split('.')[0]
    setup_logging(level='INFO', full_logfile=os.path.join(get_full_cloudsc_results_dir(node, new_experiment_id),
        'log.log'))

    git_hash = check_output(['git', 'rev-parse', '--short', 'HEAD'], cwd=get_thesis_playground_root_dir())\
        .decode('UTF-8').replace('\n', '')
    this_experiment_data = {
            'experiment id': new_experiment_id,
            'git hash': git_hash,
            'node': node,
            'datetime': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'device': args.device,
            'version': version
            }
    experiment_list_df = pd.concat([experiment_list_df, pd.DataFrame([this_experiment_data])
                                    .set_index(['experiment id'])])

    data = []
    # Run cuda
    for size in sizes:
        data.extend(run_cloudsc_cuda('dwarf-cloudsc-cuda-k-caching', 'Cloudsc CUDA K-caching', size, args.repetitions))
        data.extend(run_cloudsc_cuda('dwarf-cloudsc-cuda', 'Cloudsc CUDA', size, args.repetitions))
    for opt_level in opt_levels:
        logger.info("Get SDFG for %s on %s using version %i", opt_level, args.device, version)
        sdfg = get_sdfg(opt_level, args.device, version)
        for size in sizes:
            logger.info("Compile instrument SDFG for nblocks=%i", size)
            compile_sdfg(sdfg, size, version, opt_level, args.device, instrument=True, debug=False)
            result_file = os.path.join(get_full_cloudsc_results_dir(node, new_experiment_id),
                                       f"out_{opt_level}_{size}.out")
            output = run(['sh', 'run_cloudsc.sh', result_file, str(args.repetitions)], cwd=get_thesis_playground_root_dir(),
                         capture_output=True)
            logger.debug('stdout: %s', output.stdout.decode('UTF-8'))
            if output.returncode != 0:
                logger.warning('Running cloudsc failed')
                logger.warning('stderr: %s', output.stderr.decode('UTF-8'))
            this_data = read_reports(sdfg)
            for d in this_data:
                d.update({'nblocks': size, 'opt level': opt_level})
                data.append(d)
            sdfg.clear_instrumentation_reports()
    df = pd.DataFrame(data)
    df.to_csv(os.path.join(get_full_cloudsc_results_dir(node, new_experiment_id), 'results.csv'))
    save_experiment_list_df(experiment_list_df)


def action_print(args):
    experiment_list_df = get_experiment_list_df()
    node = experiment_list_df.loc[[int(args.experiment_id)]]['node'].values[0]
    results_df = pd.read_csv(os.path.join(get_full_cloudsc_results_dir(node, args.experiment_id), 'results.csv'))
    print(results_df)


def action_list(args):
    experiment_list_df = get_experiment_list_df()
    print(experiment_list_df)


def action_plot(args):
    experiment_list_df = get_experiment_list_df()
    node = experiment_list_df.loc[[int(args.experiment_id)]]['node'].values[0]
    results_df = pd.read_csv(os.path.join(get_full_cloudsc_results_dir(node, args.experiment_id), 'results.csv'))
    figure = get_new_figure()
    ax = figure.add_subplot(1, 1, 1)
    size_vs_y_plot(ax, 'Runtime [ms]', 'Runtimes of full cloudsc', results_df, size_var_name='nblocks')
    sns.lineplot(results_df, x='nblocks', y='runtime', hue='opt level')
    save_plot(os.path.join(get_full_cloudsc_plot_dir(node), 'runtime.pdf'))


def main():
    parser = ArgumentParser(description="Generate SDFG or code of the full cloudsc code")
    parser.add_argument('--repetitions', default=3, type=int)
    subparsers = parser.add_subparsers(
            title="Commands",
            help="See the help of the respective command")

    profile_parser = subparsers.add_parser('profile', description='Do profile runs')
    profile_parser.add_argument('--device', choices=['CPU', 'GPU'], default='GPU')
    profile_parser.set_defaults(func=action_profile)

    print_parser = subparsers.add_parser('print', description='Print stored results')
    print_parser.add_argument('experiment_id')
    print_parser.set_defaults(func=action_print)

    plot_parser = subparsers.add_parser('plot', description='plot stored results')
    plot_parser.add_argument('experiment_id')
    plot_parser.set_defaults(func=action_plot)

    list_parser = subparsers.add_parser('list', description='List stored experiments')
    list_parser.set_defaults(func=action_list)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
