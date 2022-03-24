# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path


if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)

    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                node.instrument = dace.DataInstrumentationType.No_Instrumentation

    sdfg.save(Path(os.environ["HOME"]) / "projects/tuning-dace/hG-prepared.sdfg")

