# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import re
import fpga_transpose

if __name__ == "__main__":

    sdfg = fpga_transpose.make_sdfg("remove_degenerate_loop_test")

    size = 8192

    sdfg.specialize({"N": size, "M": 1})  # Degenerate dimension

    codes = sdfg.generate_code()
    tasklet_name = sdfg.name + "_tasklet"
    for code in codes:
        if code.target_type == "device":
            break  # code now points to the appropriate code object
    else:  # Sanity check
        raise ValueError("Didn't find tasklet in degenerate map.")

    if re.search(r"for \(.+\bj\b < \bM\b", code.code) is not None:
        raise ValueError("Single iteration loop was not removed.")

    first_assignment = re.search(r"\bj\b\s*=\s*0\s*;", code.code)
    if first_assignment is None:
        raise ValueError("Assignment to constant variable not found.")

    a_input = np.arange(size, dtype=np.float64).reshape((size, 1))
    a_output = np.empty((1, size), dtype=np.float64)

    sdfg(a_input=a_input, a_output=a_output)

    if any(a_input.ravel() != a_output.ravel()):
        raise ValueError("Unexpected output.")
