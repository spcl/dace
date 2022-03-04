import abc

from typing import Dict


class PerformanceModel(abc.ABC):
    def __init__(self, machine_file_path):
        self._machine_file_path = machine_file_path

    @abc.abstractmethod
    def analyze(self, **kwargs) -> Dict:
        pass
