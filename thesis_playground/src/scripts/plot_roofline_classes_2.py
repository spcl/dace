import matplotlib
import seaborn as sns

from scripts import Script
from utils.complete_results import get_dataframe


class PlotRooflineClasses2(Script):
    name = 'plot-roofline-classes-2'
    description = "Plot the roofline of all the classes in one plot"

    @staticmethod
    def action(args):
        data = get_dataframe('all_first_opt')
        data.info()
        print(data)
