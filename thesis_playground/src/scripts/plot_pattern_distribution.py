from argparse import ArgumentParser
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import dace
import os

from execute.parameters import ParametersProvider
from execute.data import set_input_pattern
from utils.print import print_with_time
from utils.paths import get_thesis_playground_root_dir
from utils.general import get_inputs, get_outputs, copy_to_device, get_programs_data, read_source, get_sdfg, \
                          optimize_sdfg
from utils.execute_dace import RNG_SEED
from utils.plot import set_general_plot_style, save_plot, rotate_xlabels
from scripts import Script


class PlotPatternDistribution(Script):
    name = 'plot-pattern-distribution'
    description = 'Plot the distribution of branches taken for class 2 and 3'

    def add_args(self, parser: ArgumentParser):
        parser.add_argument('--only-plot', action='store_true', default=False)

    @staticmethod
    def action(args):
        nnz_data = []
        program_list = ['cloudsc_class2_781', 'cloudsc_class2_1516', 'cloudsc_class2_1762',
                        'cloudsc_class3_691', 'cloudsc_class3_965', 'cloudsc_class3_1985', 'cloudsc_class3_2120']
        folder = os.path.join(get_thesis_playground_root_dir(), 'data_analysis')
        data_file = os.path.join(folder, 'pattern_distribution_data.csv')
        if args.only_plot:
            df = pd.read_csv(data_file)
        else:
            for program in program_list:
                programs = get_programs_data()['programs']
                program_name = programs[program]
                fsource = read_source(program)
                params = ParametersProvider(program)

                print_with_time(f"Run {program}")
                for pattern in ['formula', 'const', None]:
                    rng = np.random.default_rng(RNG_SEED)
                    inputs = get_inputs(program, rng, params)
                    outputs = get_outputs(program, rng, params)
                    if pattern is not None:
                        set_input_pattern(inputs, outputs, params, program, pattern)
                    inputs = copy_to_device(inputs)
                    outputs = copy_to_device(outputs)
                    sdfg = get_sdfg(fsource, program_name)
                    optimize_sdfg(sdfg, dace.DeviceType.GPU, use_my_auto_opt=True, symbols=params.get_dict())
                    sdfg(**inputs, **outputs)
                    # this_nnz_data = {'program': program, 'pattern': pattern}
                    for var_name in outputs:
                        pattern = 'No pattern' if pattern is None else pattern
                        nnz_data.append({'program': program, 'pattern': pattern, 'var_name': var_name,
                                         'nnz': np.count_nonzero(outputs[var_name])})

            df = pd.DataFrame.from_dict(nnz_data)
            df.to_csv(data_file)

        set_general_plot_style()
        figure = plt.figure()
        figure.suptitle('Number of non zero entries per output')
        axes = figure.subplots(2, 4, gridspec_kw={'hspace': 0.5})
        for ax_row, program_row in zip(axes, [program_list[0:3], program_list[3:]]):
            for ax, program in zip(ax_row, program_row):
                plot_df = df[df['program'] == program]
                sns.barplot(data=plot_df,
                            x='var_name', y='nnz', hue='pattern', ax=ax)
                ax.set_title(program)
                ax.legend([], [], frameon=False)
                rotate_xlabels(ax)
                ax.set_xlabel('')
                ax.set_ylabel('')

        figure.delaxes(axes[0][3])
        plt.legend(loc='center', bbox_to_anchor=(0.5, 1.75))
        save_plot(os.path.join(folder, 'pattern_distribution.png'))
