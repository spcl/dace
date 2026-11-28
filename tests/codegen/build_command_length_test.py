# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A build command runs however long it is: a configure line naming hundreds of kernel libraries exceeded the 128 KiB
the kernel allows one ``sh -c`` argument (``OSError: Argument list too long``)."""

from dace.codegen import compiler


def test_a_build_command_longer_than_one_shell_argument_still_runs(tmp_path):
    marker = tmp_path / "ran"
    padding = "x" * 200_000

    compiler._run_liveoutput(f"true {padding} && touch {marker}", shell=True, cwd=str(tmp_path))

    assert marker.exists()


def test_a_cmake_definition_longer_than_one_exec_argument_still_configures(tmp_path):
    source, build = tmp_path / "source", tmp_path / "build"
    source.mkdir()
    build.mkdir()
    (source / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\nproject(long_definition NONE)\n"
        'string(LENGTH "${DACE_LIBS}" length)\nfile(WRITE "${CMAKE_BINARY_DIR}/length" "${length}")\n'
    )
    libraries = " ".join(f"/opt/kernels/lib{i:05d}/libkernel_{i:05d}.so" for i in range(5000))
    command = ["cmake", f'"{source}"', f'-DDACE_LIBS="{libraries}"']

    spilled = compiler.spill_long_definitions(command, str(tmp_path))
    compiler._run_liveoutput(" ".join(spilled), shell=True, cwd=str(build))

    assert (build / "length").read_text() == str(len(libraries))
