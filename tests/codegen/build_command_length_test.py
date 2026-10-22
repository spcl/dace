# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A build command runs however long it is: a configure line naming hundreds of kernel libraries exceeded the 128 KiB
the kernel allows one ``sh -c`` argument (``OSError: Argument list too long``)."""

from dace.codegen import compiler


def test_a_build_command_longer_than_one_shell_argument_still_runs(tmp_path):
    marker = tmp_path / "ran"
    padding = "x" * 200_000

    compiler._run_liveoutput(f"true {padding} && touch {marker}", shell=True, cwd=str(tmp_path))

    assert marker.exists()
