# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
from simple_systolic_array import P, make_sdfg
from dace.config import Config

KERNEL_NAME = ("_this_is_a_very_long_kernel_name_that_does_not_fit_"
               "in_the_61_character_limit")

if __name__ == "__main__":

    Config.set("compiler", "fpga_vendor", value="intel_fpga")

    sdfg = make_sdfg("name_too_long")
    for node, _ in sdfg.all_nodes_recursive():
        if isinstance(node, dace.sdfg.nodes.CodeNode):
            node.label += KERNEL_NAME
    sdfg.specialize({"P": 4})
    try:
        code = sdfg.generate_code()
    except dace.codegen.targets.intel_fpga.NameTooLongError:
        pass
    else:
        raise RuntimeError("No exception thrown.")
