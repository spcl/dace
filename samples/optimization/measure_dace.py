# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path

from dace.optimization import utils as optim_utils

if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-otf-transfer.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)

    dreport = sdfg.get_instrumented_data()

    arguments = {}
    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events

        for dnode in state.data_nodes():
            array = sdfg.arrays[dnode.data]
            if array.transient:
                continue

            try:
                data = dreport.get_first_version(dnode.data)
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array, data)
            except KeyError:
                print("Missing data in dreport, random array")
                arguments[dnode.data] = dace.data.make_array_from_descriptor(array)

    for name, array in list(sdfg.arrays.items()):
        if array.transient:
            continue

        if not name in arguments:
            print("Deleted: ", name)
            del sdfg.arrays[name]

    print("Compiling")
    with dace.config.set_temporary('debugprint', value=False):
        with dace.config.set_temporary('instrumentation', 'report_each_invocation', value=False):
            with dace.config.set_temporary('compiler', 'allow_view_arguments', value=True):
                csdfg = sdfg.compile()
                
                print("Measuring")
                for _ in range(30):
                    csdfg(**arguments)

                csdfg.finalize()

    report = sdfg.get_latest_report()
    print(report)

