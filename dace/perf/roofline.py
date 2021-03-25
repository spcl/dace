import dace
from dace.sdfg.graph import SubgraphView, Graph
from dace.sdfg.nodes import CodeNode, LibraryNode
from dace.properties import Property
import dace.symbolic as sym
import dace.dtypes as types

from typing import Any, Dict, List, Union, Type

from dace.perf.arith_counter import count_arithmetic_ops, \
                          count_arithmetic_ops_state, \
                          count_arithmetic_ops_subgraph

from dace.perf.movement_counter import count_moved_data, \
                             count_moved_data_state, \
                             count_moved_data_subgraph

import os, sys, platform
import subprocess

import matplotlib.pyplot as plot
from matplotlib.cm import get_cmap

import math
import sympy
import numpy as np

class PerformanceSpec:
    ''' PerformanceSpec Struct
        contains hardware information for Roofline model
    '''
    def __init__(self,
                 peak_bandwidth,
                 peak_performance,
                 data_type: Type,
                 debug = True
                 ):
        self.peak_bandwidth = peak_bandwidth
        self.peak_performance = peak_performance
        self._data_type = data_type
        self.debug = debug

        # infer no of bytes used per unit
        self._infer_bytes(self._data_type)

    def _infer_bytes(self, dtype):
        # infer no of bytes used per unit
        successful = False
        try:
            # dace dtype
            self.bytes_per_datapoint = types._BYTES[self.data_type.type]
            successful = True
        except Exception:
            pass

        try:
            # np dtype
            self.bytes_per_datapoint = types._BYTES[dtype]
            successful = True
        except Exception:
            pass

        if not successful:
            print("WARNING: Could not infer data size from data_type")
            print("Assuming 32 bit precision for data")
            self.bytes_per_datapoint = 4
        else:
            if self.debug:
                print(f"Parsed data size of {self.bytes_per_datapoint} from input")

    @property
    def data_type(self):
        return self._data_type

    @data_type.setter
    def data_type(self, dtype):
        self._data_type = dtype
        self._infer_bytes(dtype)

    @staticmethod
    def from_json(filename):
        with open(filename) as json_file:
            data = json.load(json_file)
            spec = PerformanceSpec(data["peak_bandwidth"],
                                   data["peak_performance,"],
                                   data["data_type"])
        return spec




class Roofline:
    ''' class Roofline
        for OI calculation and Roofline plots
    '''
    def __init__(self,
                 specs: PerformanceSpec,
                 symbols: Union[Dict[str, int], Dict[dace.symbol, int]],
                 debug: bool = True,
                 name: str = "roofline"):

        self.specs = specs
        self.data = {}
        self.data_symbolic = {}
        self.gflops_measured = {}
        self.gflops_roof = {}
        self.debug = debug
        self.name = name
        self.symbols = symbols


    def evaluate(self, name: str,
                   graph: Graph,
                   symbols_replacement: Dict[str, Any] = None,
                   runtimes = None):

        if name in self.data:
            index = 1
            while(True):
                if not name+str(index) in self.data:
                    name = name+str(index)
                    break
                index += 1

        print(f"Performance counter on {name}")

        # data format:
        # [operational_intensity, runtime_sample]
        self.data[name] = None
        # data format:
        # [operational_intensity_symbolic]
        self.data_symbolic[name] = None
        memory_count = 0
        flop_count = 0

        symbols_replacement = symbols_replacement or {}

        if isinstance(graph, SubgraphView):
            memory_count = count_moved_data_subgraph(graph._graph, graph, symbols_replacement)
            flop_count = count_arithmetic_ops_subgraph(graph._graph, graph, symbols_replacement)
        if isinstance(graph, dace.sdfg.SDFG):
            memory_count = count_moved_data(graph, symbols_replacement)
            flop_count = count_arithmetic_ops(graph, symbols_replacement)
        if isinstance(graph, dace.sdfg.SDFGState):
            memory_count = count_moved_data_state(graph, symbols_replacement)
            flop_count = count_arithmetic_ops_state(graph, symbols_replacement)

        print("memory_count", memory_count)
        print("flop_count", flop_count)

        operational_intensity = flop_count / (memory_count * self.specs.bytes_per_datapoint)

        # evaluate remaining sym functions
        
        x, y = sympy.symbols('x y')
        sym_locals = {sympy.Function('int_floor') : sympy.Lambda((x,y), sympy.functions.elementary.integers.floor  (x/y)),
                      sympy.Function('int_ceil')  : sympy.Lambda((x,y), sympy.functions.elementary.integers.ceiling(x/y)),
                      sympy.Function('floor')     : sympy.Lambda((x),   sympy.functions.elementary.integers.floor(x)),
                      sympy.Function('ceiling')   : sympy.Lambda((x),   sympy.functions.elementary.integers.ceiling(x)),
                      sympy.Function('bigo')      : sympy.Lambda((x),   x)
                      }
        for fun, lam in sym_locals.items():
            operational_intensity.replace(fun, lam)
        

        self.data_symbolic[name] = operational_intensity
        try:
            print("before evaluation", operational_intensity)
            self.data[name] = sym.evaluate(operational_intensity, self.symbols)
            print("after evaluation", self.data[name])
            self.data[name] = float(self.data[name])
            print("after cast", self.data[name])
            self.gflops_roof[name] = min(self.data[name] * self.specs.peak_bandwidth, self.specs.peak_performance)
        except TypeError:
            print("Not all the variables are defined in Symbols")
            print("Data after evaluation attempt:")
            print(self.data[name])

        if runtimes:
            # TODO: convert runtime into GFLOPS
            gflop = float(sym.evaluate(flop_count, self.symbols) * 10**(-9))
            self.gflops_measured[name] = list(map(lambda rt: gflop / rt, runtimes))

        if self.debug:
            print(f"Determined OI {operational_intensity}={self.data[name]} on {graph}")

        return name

    def plot(self, save_path = None, save_name = None, groups = None, show = False):

        base_x = 2
        base_y = 10

        x_0 = 1e-9
        x_ridge = self.specs.peak_performance / self.specs.peak_bandwidth
        x_2 = max([val for val in self.data.values()] + [20]) *base_x**(+2.0)
        y_0 = x_0*self.specs.peak_bandwidth
        y_ridge = self.specs.peak_performance
        y_2 = self.specs.peak_performance

        plot.loglog([x_0,x_ridge,x_2],[y_0,y_ridge,y_2],
                     basex = base_x, basey = base_y, linewidth = 3.0, color = "grey")

        # define plotting borders, adaptive to data
        # hacky but does the job
        x_min = min([val for val in self.data.values()] + [0.5])*base_x**(-1.0)
        x_max = max([val for val in self.data.values()] + [20]) *base_x**(+2.0)
        y_min = min( min([val for val in self.data.values()])*y_ridge / x_ridge, \
                     min([1]+[minrt for minrt in [min(rt) for rt in self.gflops_measured.values() if rt]]) \
                     ) * base_y**(-0.5)
        y_max = y_ridge * (base_y**1.5)

        # define a color scheme and cycle through it
        # see https://stackoverflow.com/questions/4971269/how-to-pick-a-new-color-for-each-plotted-line-within-a-figure-in-matplotlib
        colors = get_cmap("tab10").colors
        for i, (key,oi) in enumerate(self.data.items()):
            plot.loglog([oi, oi], [y_0, y_max],label=key, basex = base_x, basey = base_y, color=colors[i%10], linewidth = 1.5)
            if key in self.gflops_measured:
                if len(self.gflops_measured[key]) < 10:
                    # just plot everything
                    for rt in self.gflops_measured[key]:
                        plot.plot([oi],[rt], marker='o',markersize=10,color=colors[i%10], mew=1.5, mfc = 'none')
                else:
                    # boxplot
                    if len(sorted(list(dict.fromkeys(self.data.values())))) >= 2:
                        boxplot_width = min(40, max(20, np.log2(min(np.divide(sorted(list(dict.fromkeys(self.data.values())))[1:],
                                                                              sorted(list(dict.fromkeys(self.data.values())))[:-1] )))*20))
                    else:
                        boxplot_width = 40

                    perc_100 = np.quantile(self.gflops_measured[key],1)
                    perc_75 =  np.quantile(self.gflops_measured[key],0.75)
                    perc_50 =  np.quantile(self.gflops_measured[key],0.5)
                    perc_25 =  np.quantile(self.gflops_measured[key],0.25)
                    perc_0  =  np.quantile(self.gflops_measured[key],0)
                    plot.plot([oi],[perc_100], '.', color = colors[i%10], mew=2, markersize = 5)
                    plot.plot([oi, oi],[perc_25 , perc_75], solid_capstyle = 'butt', color = colors[i%10], linewidth = boxplot_width, alpha = 0.5)
                    plot.plot([oi],[perc_50], '_', color = colors[i%10], mew=0.85, markersize = boxplot_width)
                    plot.plot([oi],[perc_0], '.', color = colors[i%10], mew=2, markersize = 5)


        plot.title(f"{self.name}[{self.symbols}]")
        plot.xlabel(f"Operational Intensity (FLOP/byte)")
        plot.ylabel(f"GFLOP/s")
        plot.legend()

        # set axis

        plot.xlim(x_min, x_max)
        plot.ylim(y_min, y_max)

        plot.grid()

        if save_path is not None and save_path != '' and save_path[-1] != '/':
            save_path += '/'

        if save_name is None:
            save_name = self.name

        if save_path is not None:
            plot.savefig(fname = save_path+save_name)

        if show is True:
            plot.show()
            plot.close()

        plot.clf()


    def sdfv(self):
        pass

    @staticmethod
    def gather_system_specs():
        # TODO: Auto system specs inferrence
        self.gather_cpu_specs()
        self.gather_gpu_specs()

    @staticmethod
    def gather_cpu_specs():
        # TODO
        pass

    @staticmethod
    def gather_gpu_specs():
        # TODO
        pass



if __name__ == '__main__':
    # some tests

    # My Crapbook:
    # ark.intel.com
    # Bandwidth =  1867 Mhz   *   64 bits    *       2       /  8
    #            memory speed    channel size   dual channel   byte/bit
    # Peak Perf = 2.7 GhZ     *     4        *      2        *  4
    #            Compute speed    VEX/SIMD         FMA         core (incl. hyperthreading?)

    peak_bandwidth = 1.867 * 64 * 2 / 8
    peak_performance = 2.7 * 4 * 2 * 4

    spec = PerformanceSpec(peak_bandwidth, peak_performance, dace.float64)
    roofline = Roofline(spec, {'N':30}, name = "test")
    roofline.data["test1"] = 1
    roofline.data["test2"] = 0.5
    roofline.data["test3"] = 0.25
    roofline.data["test4"] = 0.125
    roofline.data["test5"] = 2.5
    roofline.data["test6"] = 30
    roofline.runtimes["test1"] = [1,2,1.5,1.25,1.75,1.6,1.4,1.4,1.3,1.2,1.9,1.8,1.1,0.9,2,1.5]
    roofline.runtimes["test2"] = [0.5]
    roofline.plot(show = True, save_path = '')
