# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The ``<array>_idx`` helpers of the experimental readable CPU generator compute in the narrowest valid integer."""

import dace
from dace.config import set_temporary


def helpers(shape) -> list[str]:
    sdfg = dace.SDFG("index_ctype_" + "_".join(str(s) for s in shape).replace(" ", ""))
    sdfg.add_array("A", shape, dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        "write",
        {f"i{d}": f"0:{s}" for d, s in enumerate(shape)},
        {},
        "o = 1",
        {"o": dace.Memlet("A[" + ", ".join(f"i{d}" for d in range(len(shape))) + "]")},
        external_edges=True,
    )
    with set_temporary("compiler", "cpu", "implementation", value="experimental_readable"):
        code = sdfg.generate_code()[0].clean_code
    return [line.strip() for line in code.splitlines() if "A_idx(" in line and "return" in line]


def test_a_constant_size_that_fits_computes_in_32_bits():
    assert all("int32_t A_idx(" in line for line in helpers((1000, 1000))) and helpers((1000, 1000))


def test_a_constant_size_beyond_32_bits_computes_in_64_bits():
    assert all("int64_t A_idx(" in line for line in helpers((100000, 100000))) and helpers((100000, 100000))


def test_a_symbolic_size_computes_in_64_bits():
    assert all("int64_t A_idx(" in line for line in helpers(("N", 10))) and helpers(("N", 10))


if __name__ == "__main__":
    test_a_constant_size_that_fits_computes_in_32_bits()
    test_a_constant_size_beyond_32_bits_computes_in_64_bits()
    test_a_symbolic_size_computes_in_64_bits()
