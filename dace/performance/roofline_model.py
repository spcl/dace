import dace

from typing import Dict

from dace.performance import performance_model
from dace.performance.backends import kerncraft_wrapper

class RooflineModel(performance_model.PerformanceModel):
    """Wrapper class for roofline analysis provided by backend."""

    def __init__(self, machine_file_path):
        super(RooflineModel, self).__init__(machine_file_path=machine_file_path)

        self._backend = kerncraft_wrapper.KerncraftWrapper(self._machine_file_path, cache_predictor="SIM")

    def analyze(self, kernel: dace.SDFG, symbols: Dict[str, int]) -> Dict:
        return self._backend.roofline(kernel, symbols)
