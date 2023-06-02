import os
import seaborn as sns

from utils.plot import save_plot, get_new_figure
from utils.paths import get_complete_results_dir, get_list_of_complete_results_name
from utils.complete_results import get_dataframe
from scripts import Script


class PlotMeasurementDistribution(Script):
    name = "plot-measurement-distribution"
    description = "Plot distribution of various metrics/values measured"

    @staticmethod
    def action(args):
        for name in get_list_of_complete_results_name():
            data, classes, parameters = get_dataframe([name])
            data = data.join(classes, on='program')

            fig = get_new_figure()
            ax = fig.add_subplots(1, 1, )
            data.reset_index(inplace=True)
            sns.displot(data[(data['measurement name'] == 'Kernel Time') | (data['measurement name'] == 'Total time')],
                        x='value', col='program', col_wrap=4, hue='measurement name', bins=50, ax=ax)
            save_plot(os.path.join(get_complete_results_dir(), name, 'plots', 'measurement_distribution_time.png'))
