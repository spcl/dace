# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import utils as optim_utils

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient-tuned.sdfg"
    original_sdfg = dace.SDFG.from_file(sdfg_path)
    original_sdfg.instrument = dace.InstrumentationType.GPU_Events

    dreport = original_sdfg.get_instrumented_data()

    runtime = optim_utils.measure(original_sdfg, dreport=dreport, repetitions=30, print_report=True)
    print(runtime)
