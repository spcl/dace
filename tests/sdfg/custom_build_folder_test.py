# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
import dace
import os
import tempfile


@dace.program
def customprog(A: dace.float64[20]):
    return A + 1


def test_default_build_folder(monkeypatch):
    """Tests if the `default_build_folder` configuration key is respected.
    """
    monkeypatch.delenv("DACE_cache", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir:
        with dace.config.set_temporary('default_build_folder', value=tmpdir), dace.config.set_temporary('cache',
                                                                                                        value='single'):
            # Ensure build folder matches
            sdfg = customprog.to_sdfg()
            assert str(sdfg.build_folder).startswith(tmpdir)
            assert str(sdfg.build_folder).endswith("/single_cache")

            # Ensure that `build_folder` is not serialized if it was not specified.
            json_dump = sdfg.to_json()
            assert 'build_folder' not in json_dump['attributes']

            sdfg_restore = dace.SDFG.from_json(json_dump)
            assert sdfg_restore._build_folder is None

            csdfg = sdfg.compile()
            assert sdfg._build_folder is None

            # Ensure files were generated in the right folder
            sdfg_dump_path = os.path.join(sdfg.build_folder, 'program.sdfgz')
            assert os.path.isfile(sdfg_dump_path)

            # Because the build folder was explicitly set during compilation, it should be dumped.
            assert csdfg._sdfg._build_folder is not None
            assert csdfg._sdfg._build_folder == sdfg.build_folder

            # Also test if it is was stored in the dump.
            prog_sdfg = dace.SDFG.from_file(sdfg_dump_path)
            assert prog_sdfg._build_folder == sdfg.build_folder

            # Ensure file is closed so it can be deleted
            del csdfg


def test_explicitly_set_build_folder(monkeypatch):
    monkeypatch.delenv("DACE_cache", raising=False)
    with tempfile.TemporaryDirectory() as tmpdir_def, tempfile.TemporaryDirectory() as tmpdir_used:
        with dace.config.set_temporary('default_build_folder',
                                       value=tmpdir_def), dace.config.set_temporary('cache', value='single'):

            # Ensure build folder matches
            sdfg = customprog.to_sdfg()
            sdfg.build_folder = tmpdir_used

            # Because the build folder was set explicitly it is used exactly, i.e. the `cache` mode is ignored.
            assert str(sdfg.build_folder) == tmpdir_used
            assert str(sdfg._build_folder) == tmpdir_used

            # Ensure that `build_folder` is serialized because it was set explicitly.
            json_dump = sdfg.to_json()
            assert json_dump['attributes']['build_folder'] == tmpdir_used

            sdfg_restore = dace.SDFG.from_json(json_dump)
            assert sdfg_restore._build_folder == tmpdir_used


if __name__ == '__main__':
    print("Must be called using `pytest`.")
