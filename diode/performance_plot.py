import re
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_gtk3cairo import FigureCanvasGTK3Cairo \
        as FigureCanvas


class PerformancePlot:
    def __init__(self, builder):
        self.builder = builder
        self.data = []
        self.xlabs = []

    def parse_result_log(self, resfile):
        with open(resfile) as f:
            data = f.read()
        p = re.compile('\s(\d+\.\d+)$', re.MULTILINE)
        times = p.findall(data)
        return times

    def add_run(self, name, times):
        self.xlabs.append(name)
        times_np = np.asfarray(times, float)
        self.data.append(np.median(times_np))

    def render(self):
        sw = self.builder.get_object('performance_window')
        old = sw.get_children()
        for w in old:
            w.destroy()

        fig = Figure(figsize=(15, 15), dpi=70)
        ax = fig.add_subplot(111)
        ax.set_ylabel("Runtime [s]")
        ax.set_xlabel("Variant [order of exec.]")
        ax.bar(x=np.arange(len(self.data)), height=(self.data))
        for idx, data in enumerate(self.data):
            ax.annotate(
                str(data)[:6],
                horizontalalignment='center',
                xy=(idx, data),
                xytext=(idx, data))
        ax.plot()
        canvas = FigureCanvas(fig)
        sw.add_with_viewport(canvas)
        sw.show_all()
