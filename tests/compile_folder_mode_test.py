# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import uuid
import pathlib
import copy
import re
import tempfile

import dace
from dace.codegen import compiler as sdfg_compiler


@pytest.fixture
def tmp_path() -> pathlib.Path:
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield pathlib.Path(tmp_dir)


def _make_test_sdfg() -> dace.SDFG:
    sdfg = dace.SDFG("test_sdfg_" + str(uuid.uuid1()).replace("-", "_"))
    state = sdfg.add_state()
    for name in "abc":
        sdfg.add_array(
            name,
            shape=(10, ),
            dtype=dace.float64,
            transient=False,
        )

    state.add_mapped_tasklet(
        "comp",
        map_ranges={"__i": "0:10"},
        inputs={
            "__in1": dace.Memlet("a[__i]"),
            "__in2": dace.Memlet("b[__i]"),
        },
        outputs={
            "__out": dace.Memlet("c[__i]"),
        },
        code="__out = __in1 + __in2",
        external_edges=True,
    )
    sdfg.validate()

    return sdfg


def _run_sdfg(csdfg):
    res = {name: np.array(np.random.rand(10), copy=True, dtype=np.float64) for name in "abc"}
    ref = copy.deepcopy(res)
    ref["c"] = ref["a"] + ref["b"]

    csdfg(**res)

    assert all(np.allclose(ref[n], res[n]) for n in res.keys())


def _load_and_run_sdfg(build_folder, sdfg):
    csdfg = sdfg_compiler.load_precompiled_sdfg(build_folder, sdfg)
    _run_sdfg(csdfg)


def test_development_folder_mode():
    with dace.config.temporary_config() as Config:
        Config.set('compiler', 'build_folder_mode', value="development")
        sdfg = _make_test_sdfg()
        build_folder = pathlib.Path(sdfg.build_folder)
        assert not build_folder.exists()

        not_csdfg = sdfg.compile(return_program_handle=False)
        assert not_csdfg is None

    assert build_folder.exists()
    assert sdfg_compiler.get_folder_mode(build_folder) == "development"

    expected_files = {
        "build": pathlib.Path.is_dir,
        "include": pathlib.Path.is_dir,
        "map": pathlib.Path.is_dir,
        "perf": pathlib.Path.is_dir,
        "sample": pathlib.Path.is_dir,
        "src": pathlib.Path.is_dir,
        "CACHEDIR.TAG": pathlib.Path.is_file,
        "FOLDER_MODE": pathlib.Path.is_file,
        "dace.conf": pathlib.Path.is_file,
        "dace_files.csv": pathlib.Path.is_file,
        "dace_environments.csv": pathlib.Path.is_file,
        "program.sdfgz": pathlib.Path.is_file,
    }

    for found_path in build_folder.iterdir():
        found_file = found_path.name
        assert found_file in expected_files
        assert expected_files[found_file](found_path)

    # Now run it.
    _load_and_run_sdfg(build_folder, sdfg)

    # Special case for development is that if there is no `FOLDER_MODE` it is still development.
    version_file = build_folder / "FOLDER_MODE"
    assert version_file.exists()
    version_file.unlink()
    assert not version_file.exists()
    assert sdfg_compiler.get_folder_mode(build_folder) == "development"


def test_production_folder_mode():
    with dace.config.temporary_config() as Config:
        Config.set('compiler', 'build_folder_mode', value="production")
        sdfg = _make_test_sdfg()
        build_folder = pathlib.Path(sdfg.build_folder)
        assert not build_folder.exists()

        not_csdfg = sdfg.compile(return_program_handle=False)
        assert not_csdfg is None

    assert build_folder.exists()
    assert sdfg_compiler.get_folder_mode(build_folder) == "production"

    lib_path = sdfg_compiler.get_binary_name(build_folder, sdfg_name=sdfg.name, folder_mode="production")
    libstub_path = sdfg_compiler._get_stub_library_path(lib_path.name)

    expected_files = {
        "program.sdfgz": pathlib.Path.is_file,
        "CACHEDIR.TAG": pathlib.Path.is_file,
        "FOLDER_MODE": pathlib.Path.is_file,
        lib_path.name: pathlib.Path.is_file,
        libstub_path.name: pathlib.Path.is_file,
    }

    for found_path in build_folder.iterdir():
        found_file = found_path.name
        assert found_file in expected_files
        assert expected_files[found_file](found_path)

    # Now run it.
    _load_and_run_sdfg(build_folder, sdfg)

    # If we delete the FOLDER_MODE file we will get an error.
    version_file = build_folder / "FOLDER_MODE"
    assert version_file.exists()
    version_file.unlink()
    assert not version_file.exists()

    with pytest.raises(NotADirectoryError,
                       match=re.escape(f'``{build_folder}`` does not appear to be a valid old-style build folder.')):
        sdfg_compiler.get_folder_mode(build_folder)

    assert sdfg_compiler.get_folder_mode(build_folder, probe=True) is None


def _test_build_with_scheme_one_and_then_switch_impl(
    version1: str,
    version2: str,
) -> None:
    with dace.config.temporary_config() as conf:
        conf.set('compiler', 'build_folder_mode', value=version1)

        sdfg = _make_test_sdfg()
        build_folder = pathlib.Path(sdfg.build_folder).resolve()
        assert not build_folder.exists()

        csdfg1 = sdfg.compile()

    lib1_path = pathlib.Path(csdfg1.filename)
    assert sdfg_compiler.get_folder_mode(build_folder) == version1
    assert lib1_path.exists()

    with dace.config.temporary_config() as conf:
        conf.set('compiler', 'build_folder_mode', value=version1)

        # This is for ensuring that the code is actually regenerated.
        conf.set('compiler', 'use_cache', value=False)
        sdfg._recompile = True
        sdfg.regenerate_code = True

        csdfg2 = sdfg.compile()

    lib2_path = pathlib.Path(csdfg2.filename)

    assert sdfg_compiler.get_folder_mode(build_folder) == version1
    assert csdfg1.sdfg.name != csdfg2.sdfg.name
    assert lib1_path != lib2_path
    assert lib1_path.exists()  # Still existing.
    assert lib2_path.exists()
    assert str(lib1_path).startswith(str(build_folder))
    assert str(lib2_path).startswith(str(build_folder))

    if version1 == "production":
        assert lib1_path.parent == build_folder
        assert lib2_path.parent == build_folder
        libstub1_path = sdfg_compiler._get_stub_library_path(lib1_path.name)
        libstub2_path = sdfg_compiler._get_stub_library_path(lib2_path.name)

        expected_files = {
            "CACHEDIR.TAG": pathlib.Path.is_file,
            "FOLDER_MODE": pathlib.Path.is_file,
            "program.sdfgz": pathlib.Path.is_file,
            lib1_path.name: pathlib.Path.is_file,
            libstub1_path.name: pathlib.Path.is_file,
            lib2_path.name: pathlib.Path.is_file,
            libstub2_path.name: pathlib.Path.is_file,
        }
        for found_path in build_folder.iterdir():
            found_file = found_path.name
            assert found_file in expected_files
            assert expected_files[found_file](found_path)

    else:
        assert str(lib1_path).startswith(str(build_folder / "build"))
        assert str(lib2_path).startswith(str(build_folder / "build"))
        assert (build_folder / "build").is_dir()
        assert (build_folder / "src").is_dir()
        assert (build_folder / "include").is_dir()

    _run_sdfg(csdfg1)
    _run_sdfg(csdfg2)


def test_build_with_scheme_one_and_then_switch():
    _test_build_with_scheme_one_and_then_switch_impl(
        version1="development",
        version2="production",
    )
    _test_build_with_scheme_one_and_then_switch_impl(
        version1="production",
        version2="development",
    )


def _expected_binary_path(
    build_folder: pathlib.Path,
    sdfg_name: str,
    folder_mode: str,
    lib_extension: str,
) -> pathlib.Path:
    if folder_mode == "development":
        return build_folder / "build" / f"lib{sdfg_name}.{lib_extension}"
    assert folder_mode == "production"
    return build_folder / f"lib{sdfg_name}.{lib_extension}"


def _test_get_binary_name_detects_folder_mode_switch_impl(
    build_folder: pathlib.Path,
    version1: str,
    version2: str,
) -> None:
    sdfg_name = "some_sdfg"
    lib_extension = "so"
    build_folder.mkdir()

    with dace.config.temporary_config() as conf:
        # The configuration keeps naming `version1`; after the switch only the
        #  `FOLDER_MODE` file knows about `version2`, so `get_binary_name()` must
        #  take the mode from the folder and not from the configuration.
        conf.set('compiler', 'build_folder_mode', value=version1)

        sdfg_compiler.generate_program_folder(None, [], str(build_folder), folder_mode=version1)
        assert sdfg_compiler.get_folder_mode(build_folder) == version1

        lib1_path = sdfg_compiler.get_binary_name(build_folder, sdfg_name=sdfg_name, lib_extension=lib_extension)
        assert lib1_path == _expected_binary_path(build_folder, sdfg_name, version1, lib_extension)

        # Now switch the folder to the second mode; the configuration still says `version1`.
        sdfg_compiler.generate_program_folder(None, [], str(build_folder), folder_mode=version2)
        assert sdfg_compiler.get_folder_mode(build_folder) == version2

        lib2_path = sdfg_compiler.get_binary_name(build_folder, sdfg_name=sdfg_name, lib_extension=lib_extension)
        assert lib2_path == _expected_binary_path(build_folder, sdfg_name, version2, lib_extension)
        assert lib1_path != lib2_path


def test_get_binary_name_detects_folder_mode_switch(tmp_path):
    _test_get_binary_name_detects_folder_mode_switch_impl(
        build_folder=tmp_path / "dev_to_prod",
        version1="development",
        version2="production",
    )
    _test_get_binary_name_detects_folder_mode_switch_impl(
        build_folder=tmp_path / "prod_to_dev",
        version1="production",
        version2="development",
    )


def test_get_folder_mode_probes_inconsistent_old_style_folder(tmp_path):
    # An old-style development folder that was generated but never compiled, i.e.
    #  there is no `FOLDER_MODE` file and no `build` folder. Such a folder is
    #  inconsistent, thus probing must return `None` and `get_binary_name()` must
    #  fall back to the configuration instead of raising.
    build_folder = tmp_path / "old_style_never_compiled"
    build_folder.mkdir()
    for sub_folder in ["map", "src", "include", "sample"]:
        (build_folder / sub_folder).mkdir()

    assert sdfg_compiler.get_folder_mode(build_folder, probe=True) is None
    with pytest.raises(NotADirectoryError,
                       match=re.escape(f'The old-style folder ``{build_folder}`` is inconsistent.')):
        sdfg_compiler.get_folder_mode(build_folder)

    for folder_mode in ["development", "production"]:
        with dace.config.temporary_config() as conf:
            conf.set('compiler', 'build_folder_mode', value=folder_mode)
            lib_path = sdfg_compiler.get_binary_name(build_folder, sdfg_name="some_sdfg", lib_extension="so")
        assert lib_path == _expected_binary_path(build_folder, "some_sdfg", folder_mode, "so")


def test_already_loaded_and_comple_again():
    _test_build_with_scheme_one_and_then_switch_impl(
        version1="development",
        version2="development",
    )
    _test_build_with_scheme_one_and_then_switch_impl(
        version1="production",
        version2="production",
    )


if __name__ == '__main__':
    test_development_folder_mode()
    test_production_folder_mode()
    test_already_loaded_and_comple_again()
    test_build_with_scheme_one_and_then_switch()
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_get_binary_name_detects_folder_mode_switch(pathlib.Path(tmp_dir))
    with tempfile.TemporaryDirectory() as tmp_dir:
        test_get_folder_mode_probes_inconsistent_old_style_folder(pathlib.Path(tmp_dir))


def test_a_missing_library_is_reported_as_a_missing_file(tmp_path: pathlib.Path):
    """``get_program_handle`` names the library it could not find.

    The check built its message by concatenating the path onto a string, and ``library_path`` is a
    ``pathlib.Path`` two lines above -- so the missing-library branch raised ``TypeError: can only
    concatenate str (not "PosixPath") to str`` instead of the ``FileNotFoundError`` it means. A
    caller that legitimately probes for a library saw the wrong exception type, and the message
    that says WHICH library never reached anyone.
    """
    missing = tmp_path / "never_built" / "libprogram.so"
    with pytest.raises(FileNotFoundError, match=re.escape(str(missing))):
        sdfg_compiler.get_program_handle(missing, _make_test_sdfg())
