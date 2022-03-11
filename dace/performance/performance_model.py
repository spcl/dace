import abc
import dace

from typing import Dict, Generator

from dace.transformation import helpers as xfh
from dace.sdfg.analysis import cutout as cutter


class PerformanceModel(abc.ABC):
    def __init__(self, machine_file_path):
        self._machine_file_path = machine_file_path

    @abc.abstractmethod
    def analyze(self, kernel: dace.SDFG, symbols: Dict[str, int], **kwargs) -> Dict:
        pass

    @staticmethod
    def kernels(sdfg: dace.SDFG) -> Generator[dace.SDFG, None, None]:
        for state in sdfg.nodes():
            for node in state.nodes():
                if isinstance(node, dace.nodes.MapEntry):
                    if xfh.get_parent_map(state, node) is not None:
                        continue

                    subgraph = state.scope_subgraph(node)
                    cutout = cutter.cutout_state(state, *(subgraph.nodes()), make_copy=False)
                    yield cutout
