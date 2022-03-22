# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import cutout_tuner as ct
from dace import optimization as optim
from dace.optimization import utils as optim_utils

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient.sdfg"
    original_sdfg = dace.SDFG.from_file(sdfg_path)

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient-subgraph-transfer.sdfg"
    transfer_sdfg = dace.SDFG.from_file(sdfg_path)

    dreport = original_sdfg.get_instrumented_data()

    runtime = optim_utils.measure(original_sdfg, dreport=dreport, repetitions=30, print_report=True)
    transfer_runtime = optim_utils.measure(transfer_sdfg, dreport=dreport, repetitions=30, print_report=True)

    print(runtime, transfer_runtime)
