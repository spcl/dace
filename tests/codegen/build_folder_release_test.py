# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A process-scoped build folder must not outlive the process that owns it.

Under ``cache: unique`` the folder is named after the PID, so nothing outside the process can ever
address it -- yet it survived the run, and a test sweep left one behind per (SDFG name, worker);
this repo's tree had 540 of them at 238 MB. These pin that such a folder is dropped when its
process ends, and that every other policy keeps its folder, since there the folder IS the build
cache and deleting it would turn the next run into a full recompile.

Dropping is deliberately NOT tied to ``CompiledSDFG`` lifetime: CPython frees that object through
the garbage collector, so there is no moment a test -- or a user -- could observe.
"""
import os
import pathlib
import subprocess
import sys
import textwrap

import pytest

import dace
from dace.codegen import compiler


@dace.program
def addone(A: dace.float64[16]):
    for i in dace.map[0:16]:
        A[i] += 1.0


def run_child(body: str, tmp_path, cache: str) -> str:
    """Run ``body`` in a fresh interpreter under cache policy ``cache``, and return the folder it reports.

    The LAST line, not all of stdout: ``DACE_testing_serialization=1`` prints ahead of the child's own
    ``print(build_folder)``.

    Written to a file rather than passed to ``python -c``: the dace frontend reads a program's
    source back off disk, and there is none for ``-c``.
    """
    script = tmp_path / 'child.py'
    script.write_text(
        textwrap.dedent(f"""
        import os
        import numpy as np
        import dace
        from {pathlib.Path(__file__).stem} import addone

        with dace.config.set_temporary('cache', value={cache!r}), \\
             dace.config.set_temporary('default_build_folder', value={str(tmp_path)!r}):
        {textwrap.indent(textwrap.dedent(body), ' ' * 12)}
        """))
    roots = [str(pathlib.Path(dace.__file__).parent.parent), str(pathlib.Path(__file__).parent)]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(roots))
    # Config.get() lets a DACE_cache env var win over set_temporary() (dace/config.py), so a
    # surrounding DACE_cache=... (CI pins one for the whole matrix leg) would silently override
    # the ``cache`` argument above in the child. Same fix as custom_build_folder_test.py's
    # ``unlaunched`` fixture.
    env.pop('DACE_cache', None)
    out = subprocess.run([sys.executable, str(script)], check=True, capture_output=True, text=True, env=env)
    lines = out.stdout.strip().splitlines()
    return lines[-1] if lines else ''


#: Compile, run, and report the folder the run used, so the parent can check it after the exit.
COMPILE_AND_REPORT = """
    sdfg = addone.to_sdfg(simplify=True)
    sdfg.name = 'release_probe'
    csdfg = sdfg.compile()
    data = np.zeros(16)
    csdfg(A=data)
    assert np.allclose(data, 1.0)
    assert os.path.isdir(sdfg.build_folder)
    print(sdfg.build_folder)
"""


def test_unique_folder_does_not_outlive_its_process(tmp_path):
    folder = run_child(COMPILE_AND_REPORT, tmp_path, cache='unique')

    assert folder, 'child did not report its build folder'
    assert not os.path.isdir(folder), 'process-scoped build folder outlived the process that owned it'


def test_named_folder_is_kept(tmp_path):
    folder = run_child(COMPILE_AND_REPORT, tmp_path, cache='name')

    assert os.path.isdir(folder), 'dropped a folder a later run addresses by name'


def test_assigned_folder_is_kept(tmp_path):
    """An explicitly assigned folder belongs to the caller, whatever the cache policy says."""
    folder = run_child(f"""
        sdfg = addone.to_sdfg(simplify=True)
        sdfg.name = 'release_assigned'
        sdfg.build_folder = {str(tmp_path / 'mine')!r}
        sdfg.compile()
        print(sdfg.build_folder)
        """,
                       tmp_path,
                       cache='unique')

    assert os.path.isdir(folder), 'dropped a folder the caller assigned'


def test_disposable_predicate_follows_the_cache_policy(tmp_path, monkeypatch):
    # A surrounding DACE_cache=... wins over set_temporary() below (dace/config.py Config.get()),
    # so it must be cleared here or every policy would read back as whatever the env var says.
    monkeypatch.delenv('DACE_cache', raising=False)
    sdfg = addone.to_sdfg(simplify=True)
    for policy, expected in (('unique', True), ('name', False), ('hash', False), ('single', False)):
        with dace.config.set_temporary('cache', value=policy):
            assert compiler.build_folder_is_disposable(sdfg) is expected, policy


if __name__ == '__main__':
    pytest.main([__file__])
