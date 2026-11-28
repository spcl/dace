# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The HIP BLAS expansions link hipBLAS from ROCm 7 or newer: the version is read from the ROCm root (or
``hipconfig``), an older ROCm counts as not installed, and building a hipBLAS call there names the version."""

import pathlib

import pytest

from dace.libraries import hip_dialect
from dace.libraries.blas.environments.hipblas import hipBLAS


@pytest.fixture
def rocm(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch):
    """A ROCm root at ``tmp_path`` whose ``.info/version`` the test writes; the version cache is cleared around it."""

    def install(version: str) -> pathlib.Path:
        (tmp_path / ".info").mkdir(exist_ok=True)
        (tmp_path / ".info" / "version").write_text(version)
        return tmp_path

    monkeypatch.setenv("ROCM_PATH", str(tmp_path))
    monkeypatch.delenv("HIP_PATH", raising=False)
    hip_dialect.rocm_version.cache_clear()
    yield install
    hip_dialect.rocm_version.cache_clear()


@pytest.mark.parametrize(
    ("text", "version"), [("7.2.4-43\n", (7, 2)), ("7.1.52801-9999", (7, 1)), ("6.4.0", (6, 4)), ("unknown", None)]
)
def test_a_rocm_version_string_reads_as_major_and_minor(text, version):
    assert hip_dialect.parsed_version(text) == version


def test_the_version_comes_from_the_rocm_roots_info_file(rocm):
    rocm("7.2.4-43")

    assert hip_dialect.rocm_version() == (7, 2)


@pytest.mark.parametrize(("version", "installed"), [("7.0.0", True), ("7.2.4", True), ("6.4.1", False)])
def test_hipblas_counts_as_installed_only_on_rocm_7_or_newer(rocm, monkeypatch, version, installed):
    rocm(version)
    monkeypatch.setattr(hip_dialect, "find_rocm_library", lambda name: f"/rocm/lib/lib{name}.so")

    assert hipBLAS.is_installed() is installed


def test_building_a_hipblas_call_on_an_older_rocm_names_the_version(rocm):
    rocm("6.4.1")

    with pytest.raises(RuntimeError, match="hipBLAS needs ROCm 7.0 or newer; this ROCm is 6.4"):
        hipBLAS.handle_setup_code(None)


def test_rocm_7_passes_the_check(rocm):
    rocm("7.2.4")

    hip_dialect.require_rocm("hipBLAS")
