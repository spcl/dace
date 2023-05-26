import os
import seaborn as sns
import matplotlib.pyplot as plt

from utils.paths import get_complete_results_dir
from utils.plot import save_plot, set_general_plot_style, rotate_xlabels
from utils.complete_results import get_roofline_dataframe
from scripts import Script


class PlotMueClasses(Script):
    name = "plot-mue-classes"
    description = "Plot MUE, BW and IO efficiency of the classes"

    @staticmethod
    def action(args):
        df = get_roofline_dataframe(['all_first_opt'])
        df['io efficiency'] = df['theoretical bytes'] / df['measured bytes']
        df['bw efficiency'] = df['achieved bandwidth'] / df['peak bandwidth']
        df['mue'] = df['io efficiency'] * df['bw efficiency']

        set_general_plot_style()
        plot_data = df[['program', 'experiment name', 'io efficiency', 'bw efficiency', 'mue']]\
            .melt(id_vars=['program', 'experiment name'], value_vars=['io efficiency', 'bw efficiency', 'mue'])
        ax = sns.barplot(data=plot_data, x='program', y='value', hue='variable')
        ax.set_xlabel('')
        ax.set_ylabel('')
        rotate_xlabels(ax)
        sns.move_legend(ax, 'center', title=None, frameon=False, ncols=3, bbox_to_anchor=(0.5, 0.97))
        plt.title('MUE and IO/Bandwidth efficiency for the classes using formula pattern and my improvements')
        plt.tight_layout()
        save_plot(os.path.join(get_complete_results_dir(), 'plots', 'mue.pdf'))
