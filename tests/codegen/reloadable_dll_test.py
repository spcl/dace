# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Loading a library whose path is already loaded must load that library, never a stale copy left next to it."""

import os
import pathlib
import shutil

import numpy as np

import dace
from dace.codegen import compiler


def constant_sdfg(name: str, value: int) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array("out", [1], dace.int64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet("constant", {}, {"o"}, f"o = {value}")
    state.add_edge(tasklet, "o", state.add_write("out"), None, dace.Memlet("out[0]"))
    return sdfg


def run(csdfg) -> int:
    out = np.zeros(1, np.int64)
    csdfg(out=out)
    return int(out[0])


def test_a_leftover_copy_is_not_loaded_in_place_of_the_new_library():
    first = constant_sdfg("reloadable_dll_stale_copy", 1).compile()
    library = pathlib.Path(first._lib._library_filename)
    # A copy an earlier reload left behind: on disk, no longer loaded.
    shutil.copyfile(library, f"{library}_0")
    # The library is rebuilt in place while the first build is still loaded.
    rebuilt = constant_sdfg("reloadable_dll_stale_copy", 2)
    rebuilt.build_folder = os.path.join(dace.Config.get("default_build_folder"), "reloadable_dll_stale_copy_rebuilt")
    rebuilt_library = pathlib.Path(rebuilt.compile()._lib._library_filename)
    os.unlink(library)
    shutil.copyfile(rebuilt_library, library)

    second = compiler.get_program_handle(library, rebuilt)

    assert run(second) == 2
    assert run(first) == 1


if __name__ == "__main__":
    test_a_leftover_copy_is_not_loaded_in_place_of_the_new_library()
