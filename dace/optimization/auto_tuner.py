# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace

from abc import ABC, abstractmethod

class AutoTuner(ABC):
    """
    General API for automatic SDFG tuners.
    Contains a single method: ``tune``, which initiates the tuning process.
    """
    def __init__(self, sdfg: dace.SDFG) -> None:
        self._sdfg = sdfg

    @abstractmethod
    def tune(self) -> None:
        """
        Tunes an SDFG.
        """
        pass
