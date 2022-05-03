# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from typing import Any, Dict


class AutoTuner:
    """
    General API for automatic SDFG tuners.
    Contains a single method: ``optimize``, which initiates the tuning process.
    """
    def __init__(self, sdfg: dace.SDFG) -> None:
        self._sdfg = sdfg

    def optimize(self, apply: bool = True, measurements: int = 30) -> Dict[Any, Any]:
        """
        Tunes an SDFG.
        :param apply: Applies the best-found configuration on the original SDFG.
        :param measurements: The number of times to run the SDFG for performance analysis.
        :return: A dictionary mapping measured configurations to results (usually string to numeric runtimes).
        """
        return {}
