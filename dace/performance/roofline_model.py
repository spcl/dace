import dace

from typing import Dict, Generator, Tuple

from dace.transformation import helpers as xfh
from dace.performance import performance_model

from dace.performance.analysis import kerncraft_wrapper

class RooflineModel(performance_model.PerformanceModel):
    """Wrapper class for roofline analysis provided by backend."""

    def __init__(self, machine_file_path):
        super(RooflineModel, self).__init__(machine_file_path=machine_file_path)

        self._backend = kerncraft_wrapper.KerncraftWrapper(self._machine_file_path, cache_predictor="SIM")

    def analyze(self, state: dace.SDFGState, subgraph: dace.sdfg.ScopeSubgraphView, symbol_values: Dict[str, int]) -> Dict:
        return self._backend.roofline(state, subgraph, symbol_values)
    
    @staticmethod
    def kernels(sdfg: dace.SDFG) -> Generator[Tuple[dace.SDFGState, dace.sdfg.ScopeSubgraphView], None, None]:
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if xfh.get_parent_map(state, node) is not None:
                        continue

                    yield state, state.scope_subgraph(node)
