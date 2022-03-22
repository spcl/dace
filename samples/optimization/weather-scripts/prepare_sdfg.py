# Copyright 2019-2022 ETH Zurich and the DaCe authors. All rights reserved.
import os
import dace
import numpy as np

from pathlib import Path


if __name__ == '__main__':

    sdfg_path = Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient.sdfg"
    sdfg = dace.SDFG.from_file(sdfg_path)

    w = ['__g_self__ext5', '__g_self__ext5_0', '__g_self__ext5_0_0', '__g_self__ext5_1', '__g_self__ext5_10',
          '__g_self__ext5_2',  '__g_self__ext5_3',  '__g_self__ext5_4',  '__g_self__ext5_5',
          '__g_self__ext5_6',  '__g_self__ext5_7',  '__g_self__ext5_8',  '__g_self__ext5_9',
    ]

    for sd in sdfg.all_sdfgs_recursive():
        for ww in w:
            if ww in sd.arrays:
                print("Removed")
                del sd.arrays[ww]

    transients = ["_q_advected_y", "_q_advected_x", "_q_x_advected_mean", "_q_x_advected_mean", "_q_y_advected_mean", "_q_advected_x_y_advected_mean", "__tmp613"]
    for nsdfg in sdfg.all_sdfgs_recursive():
        for name, array in nsdfg.arrays.items():
            for tran in transients:
                if "advected_x_y_advected_mean" in name:
                    array.transient = True
                    print(name)
                    break

    

    #sdfg.apply_gpu_transformations()
    sdfg.simplify()

    for state in sdfg.nodes():
        state.instrument = dace.InstrumentationType.GPU_Events
        for node in state.nodes():
            if isinstance(node, dace.nodes.AccessNode):
                node.instrument = dace.DataInstrumentationType.No_Instrumentation

    sdfg.save(Path(os.environ["HOME"]) / "projects/tuning-dace/hG-transient-plus.sdfg")

