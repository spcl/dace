# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Dict


class AutoTuner:

    def __init__(self, sdfg: dace.SDFG) -> None:
        self._sdfg = sdfg

    def optimize(self, apply: bool = True, measurements: int = 30) -> Dict:
        return {}
