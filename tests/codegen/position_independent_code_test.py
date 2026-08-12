# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Position-independent code is a property of the build, not a flag someone remembered to list.

Everything DaCe emits ends up in a ``dlopen``-ed shared library, so an object built without PIC
either fails to link or leaves text relocations in a library loaded into a long-lived Python host.
``CMAKE_POSITION_INDEPENDENT_CODE`` is what makes that independent of the configurable compiler
arguments.
"""
import json
import os

import numpy as np

import dace
from dace.config import set_temporary


def compile_commands_with_empty_cpu_args(tmp_path) -> list:
    """Compile a trivial SDFG with ``compiler.cpu.args`` cleared, and return the commands used."""

    @dace.program
    def addone(a: dace.float64[8]):
        a += 1.0

    sdfg = addone.to_sdfg()
    sdfg.build_folder = str(tmp_path / 'build_folder')
    with set_temporary('compiler', 'cpu', 'args', value=''):
        csdfg = sdfg.compile()

    a = np.zeros(8, np.float64)
    csdfg(a=a)
    assert np.allclose(a, 1.0), 'the library built without any configured compiler arguments does not run'

    database = os.path.join(sdfg.build_folder, 'build', 'compile_commands.json')
    assert os.path.exists(database), f'no compile database at {database}, so there is nothing to assert on'
    with open(database) as fp:
        return json.load(fp)


def test_pic_survives_clearing_the_configured_compiler_arguments(tmp_path):
    """The one setting is global, so an empty ``compiler.cpu.args`` cannot take PIC with it."""
    entries = compile_commands_with_empty_cpu_args(tmp_path)
    generated = [e for e in entries if e['file'].endswith(('.cpp', '.cu'))]
    assert generated, f'the compile database names no generated source: {[e["file"] for e in entries]}'
    missing = [e['file'] for e in generated if '-fPIC' not in e['command']]
    assert not missing, f'these translation units are compiled without -fPIC: {missing}'


if __name__ == '__main__':
    import tempfile
    import pathlib

    with tempfile.TemporaryDirectory() as folder:
        test_pic_survives_clearing_the_configured_compiler_arguments(pathlib.Path(folder))
