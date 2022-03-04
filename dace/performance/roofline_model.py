import dace
import abc

from typing import Dict, Generator

from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter
from dace.performance import performance_model

from dace.performance.analysis import kerncraft_wrapper

class RooflineModel(performance_model.PerformanceModel):
    """Wrapper class for roofline analysis provided by backend."""

    def __init__(self, machine_file_path):
        super(RooflineModel, self).__init__(machine_file_path=machine_file_path)

        self._backend = kerncraft_wrapper.KerncraftWrapper(self._machine_file_path, cache_predictor="SIM")

    def analyze(self, kernel: dace.SDFG, symbol_values: Dict[str, int]) -> Dict:
        return self._backend.roofline(kernel, symbol_values)
    
    @staticmethod
    def kernels(sdfg: dace.SDFG) -> Generator[dace.SDFG, None, None]:
        for node, state in sdfg.all_nodes_recursive():
            if isinstance(node, dace.nodes.MapEntry):
                if xfh.get_parent_map(state, node) is not None:
                    continue

                subgraph_nodes = state.scope_subgraph(node).nodes()
                cutout = cutter.cutout_state(state, *subgraph_nodes, make_copy=False)
                yield cutout
