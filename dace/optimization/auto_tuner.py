# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import abc
import dace
from typing import Dict


class AutoTuner(abc.ABC):

    def __init__(self, sdfg: dace.SDFG) -> None:
        self._sdfg = sdfg

    @abc.abstractmethod
    def optimize(self, apply: bool = True, measurements: int = 30) -> Dict:
        pass
