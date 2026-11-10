# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the host CPU and GPU queries in the code generator."""

import hashlib
import sys

import pytest

from dace.codegen import common, compiler


@pytest.fixture
def uncached():
    """Clears the caches of the queries around a test."""
    compiler.host_cpu.cache_clear()
    common.local_gpus.cache_clear()
    yield
    compiler.host_cpu.cache_clear()
    common.local_gpus.cache_clear()


@pytest.mark.skipif(not compiler.cpuinfo_fields(), reason="needs /proc/cpuinfo")
def test_host_cpu_without_py_cpuinfo(uncached, monkeypatch):
    monkeypatch.setitem(sys.modules, "cpuinfo", None)
    model, flags = compiler.host_cpu()
    flags_line = next(line for line in compiler.cpuinfo_fields() if not line.startswith("model name"))
    assert flags == flags_line.partition(":")[2].split()


def test_host_isa_id_hashes_the_cpuinfo_fields():
    fields = compiler.cpuinfo_fields()
    if fields:
        assert compiler.host_isa_id() == hashlib.sha256("".join(fields).encode()).hexdigest()[:12]


def test_nvidia_gpus_without_pynvml(uncached, monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(common, "get_gpu_backend", lambda: "cuda")
    monkeypatch.setattr(
        common,
        "_run_tool",
        lambda args: "NVIDIA GeForce RTX 3070, 8.6\nNVIDIA H100, 9.0\n" if args[0] == "nvidia-smi" else "",
    )
    assert common.local_gpus() == (
        common.GPUDevice("NVIDIA GeForce RTX 3070", "sm_86"),
        common.GPUDevice("NVIDIA H100", "sm_90"),
    )


def test_amd_gpus_without_amdsmi_query_no_nvidia_tool(uncached, monkeypatch):
    monkeypatch.setitem(sys.modules, "amdsmi", None)
    monkeypatch.setitem(sys.modules, "pynvml", None)
    monkeypatch.setattr(common, "get_gpu_backend", lambda: "hip")
    queried = []

    def run(args):
        queried.append(args[0])
        return "gfx000\ngfx942\ngfx942\n" if args[0] == "rocm_agent_enumerator" else ""

    monkeypatch.setattr(common, "_run_tool", run)
    assert common.local_gpus() == (common.GPUDevice("", "gfx942"), common.GPUDevice("", "gfx942"))
    assert queried == ["rocm_agent_enumerator"]


def test_no_gpu_backend_means_no_gpus(uncached, monkeypatch):
    def no_backend():
        raise RuntimeError("no GPU")

    monkeypatch.setattr(common, "get_gpu_backend", no_backend)
    assert common.local_gpus() == ()


if __name__ == "__main__":
    test_host_isa_id_hashes_the_cpuinfo_fields()
