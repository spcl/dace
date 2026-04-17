# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import numpy as np
import pytest
import uuid
import pathlib
import copy
import re

from typing import Tuple

import dace
from dace.codegen import compiler as sdfg_compiler


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


def test_development_folder_version():
    with dace.config.temporary_config() as Config:
        Config.set('compiler', 'build_folder_version', value="development")
        sdfg = _make_test_sdfg()
        build_folder = pathlib.Path(sdfg.build_folder)
        assert not build_folder.exists()

        not_csdfg = sdfg.compile(return_program_handle=False)
        assert not_csdfg is None

    assert build_folder.exists()
    assert sdfg_compiler.get_folder_version(build_folder) == "development"

    expected_files = {
        "build": pathlib.Path.is_dir,
        "include": pathlib.Path.is_dir,
        "map": pathlib.Path.is_dir,
        "perf": pathlib.Path.is_dir,
        "sample": pathlib.Path.is_dir,
        "src": pathlib.Path.is_dir,
        "CACHEDIR.TAG": pathlib.Path.is_file,
        "dace.conf": pathlib.Path.is_file,
        "dace_files.csv": pathlib.Path.is_file,
        "dace_environments.csv": pathlib.Path.is_file,
        "program.sdfgz": pathlib.Path.is_file,
        "VERSION": pathlib.Path.is_file,
    }

    for found_path in build_folder.iterdir():
        found_file = found_path.name
        assert found_file in expected_files
        assert expected_files[found_file](found_path)

    # Now run it.
    _load_and_run_sdfg(build_folder, sdfg)

    # Special case for development is that if there is no `VERSION` it is still development.
    version_file = build_folder / "VERSION"
    assert version_file.exists()
    version_file.unlink()
    assert not version_file.exists()
    assert sdfg_compiler.get_folder_version(build_folder) == "development"


def test_production_folder_version():
    with dace.config.temporary_config() as Config:
        Config.set('compiler', 'build_folder_version', value="production")
        sdfg = _make_test_sdfg()
        build_folder = pathlib.Path(sdfg.build_folder)
        assert not build_folder.exists()

        not_csdfg = sdfg.compile(return_program_handle=False)
        assert not_csdfg is None

    assert build_folder.exists()
    assert sdfg_compiler.get_folder_version(build_folder) == "production"

    lib_path = sdfg_compiler.get_binary_name(build_folder, sdfg_name=sdfg.name, folder_version="production")
    libstub_path = sdfg_compiler._get_stub_library_path(lib_path.name)

    expected_files = {
        "VERSION": pathlib.Path.is_file,
        lib_path.name: pathlib.Path.is_file,
        libstub_path.name: pathlib.Path.is_file,
    }

    for found_path in build_folder.iterdir():
        found_file = found_path.name
        assert found_file in expected_files
        assert expected_files[found_file](found_path)

    # Now run it.
    _load_and_run_sdfg(build_folder, sdfg)

    # If we delete the VERSION file we will get an error.
    version_file = build_folder / "VERSION"
    assert version_file.exists()
    version_file.unlink()
    assert not version_file.exists()

    with pytest.raises(NotADirectoryError,
                       match=re.escape(f'``{build_folder}`` does not appear to be a valid build folder.')):
        sdfg_compiler.get_folder_version(build_folder)

    assert sdfg_compiler.get_folder_version(build_folder, probe=True) is None


def _test_build_with_scheme_one_and_then_switch_impl(
    version1: str,
    version2: str,
) -> None:
    with dace.config.temporary_config() as conf:
        conf.set('compiler', 'build_folder_version', value=version1)

        sdfg = _make_test_sdfg()
        build_folder = pathlib.Path(sdfg.build_folder).resolve()
        assert not build_folder.exists()

        csdfg1 = sdfg.compile()

    lib1_path = pathlib.Path(csdfg1.filename)
    assert sdfg_compiler.get_folder_version(build_folder) == version1
    assert lib1_path.exists()

    with dace.config.temporary_config() as conf:
        conf.set('compiler', 'build_folder_version', value=version1)

        # This is for ensuring that the code is actually regenerated.
        conf.set('compiler', 'use_cache', value=False)
        sdfg._recompile = True
        sdfg.regenerate_code = True

        csdfg2 = sdfg.compile()

    lib2_path = pathlib.Path(csdfg2.filename)

    assert sdfg_compiler.get_folder_version(build_folder) == version1
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
            "VERSION": pathlib.Path.is_file,
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
    test_development_folder_version()
    test_production_folder_version()
    test_already_loaded_and_comple_again()
    test_build_with_scheme_one_and_then_switch()
