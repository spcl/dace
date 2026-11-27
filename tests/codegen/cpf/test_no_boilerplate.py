# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A device rendering without boilerplate (``render(..., no_boilerplate=True)``): one unit, the default stream, every
device transient carved from the caller's workspace, and the passes that get the SDFG there."""

import ctypes
import pathlib
import tempfile

import numpy as np
import pytest

import dace
from dace.codegen.cpf import EntrySignature, render
from dace.transformation.passes.gpu_specialization.bind_default_stream import DEFAULT_STREAM, DropExitStreamSync
from dace.transformation.passes.gpu_specialization.gpu_stream_scheduling import stream_sync_code
from dace.transformation.passes.workspace_carving import CarveTransientsFromWorkspace, DemotePersistentTransients
from tests.codegen.cpf.conftest import require_gpu_backend
from tests.codegen.cpf.test_emission_cuda import nvcc_library

N, M, K = (dace.symbol(name) for name in "NMK")
PAD = 3
PADDED_SIGNATURE = EntrySignature(
    ("x", "y", "N", "M", "K", "workspace", "workspace_size"), workspace="workspace", workspace_size="workspace_size"
)


def padded_transient_sdfg(name: str) -> dace.SDFG:
    """``y[i, j, k] = 2 * x[i, j, K - 1 - k] + 1`` through a 3-D device transient whose rows are padded by
    :data:`PAD`, written by one kernel and read reversed by the next."""
    sdfg = dace.SDFG(name)
    device = dace.StorageType.GPU_Global
    sdfg.add_array("x", [N, M, K], dace.float64, storage=device)
    sdfg.add_array("y", [N, M, K], dace.float64, storage=device)
    sdfg.add_array("t", [N, M, K], dace.float64, storage=device, transient=True, strides=[M * (K + PAD), K + PAD, 1])
    ranges = {"i": "0:N", "j": "0:M", "k": "0:K"}
    fill = sdfg.add_state("fill")
    fill.add_mapped_tasklet(
        "fill",
        ranges,
        {"a": dace.Memlet("x[i, j, k]")},
        "b = 2 * a",
        {"b": dace.Memlet("t[i, j, k]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    use = sdfg.add_state_after(fill, "use")
    use.add_mapped_tasklet(
        "use",
        ranges,
        {"a": dace.Memlet("t[i, j, K - 1 - k]")},
        "b = a + 1",
        {"b": dace.Memlet("y[i, j, k]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    return sdfg


def test_the_carve_sizes_a_padded_transient_by_its_strides():
    sdfg = padded_transient_sdfg("cpf_nb_padded_layout")
    sdfg.add_symbol("workspace_size", dace.int64)
    sdfg.add_array("workspace", ["workspace_size"], dace.uint8, storage=dace.StorageType.GPU_Global)
    layout = CarveTransientsFromWorkspace("workspace", "workspace_size").apply_pass(sdfg, {})
    sizes = {N: 3, M: 5, K: 7}
    assert layout.offsets == {"t": 0}
    # The last row's padding is never addressed, so the slot ends at the last element.
    elements = (sizes[N] - 1) * sizes[M] * (sizes[K] + PAD) + (sizes[M] - 1) * (sizes[K] + PAD) + sizes[K]
    assert int(dace.symbolic.evaluate(layout.size_bytes, sizes)) == 8 * elements + 64
    assert sdfg.arrays["t"].lifetime is dace.AllocationLifetime.SDFG


@pytest.mark.parametrize("language", ["cuda", "hip"])
def test_the_entry_carves_the_padded_transient_and_waits_once(language: str):
    rendering = render(
        padded_transient_sdfg(f"cpf_nb_padded_{language}"),
        language=language,
        signature=PADDED_SIGNATURE,
        no_boilerplate=True,
        check_compiles=False,
    )
    entry = rendering.code[rendering.code.index('extern "C"') :]
    assert rendering.device_code == ""
    assert "AUTO-GENERATED" not in rendering.code, "the one unit is the agent's to edit"
    assert "t = reinterpret_cast<double *>(workspace + 0);" in entry, entry
    assert "Malloc" not in entry and "Free" not in entry, entry
    # The closing device synchronization is the one wait: no stream wait before it.
    assert "StreamSynchronize" not in rendering.code, entry
    assert entry.rstrip().splitlines()[-2].strip() == f"cpf_gpu_check({language}DeviceSynchronize());", entry


@pytest.mark.gpu
def test_the_padded_transient_runs_in_a_workspace_of_exactly_the_bound():
    require_gpu_backend("cuda")
    import cupy  # Deferred: the optional GPU dependency, needed only where a GPU runs this test.

    name = "cpf_nb_padded_run"
    rendering = render(padded_transient_sdfg(name), language="cuda", signature=PADDED_SIGNATURE, no_boilerplate=True)
    sizes = {"N": 3, "M": 33, "K": 129}
    x = np.random.default_rng(7).random((sizes["N"], sizes["M"], sizes["K"]))
    nbytes = int(dace.symbolic.evaluate(dace.symbolic.pystr_to_symbolic(rendering.workspace_bytes), sizes))
    workspace = cupy.full(nbytes, 0xA5, dtype=cupy.uint8)
    device = {"x": cupy.asarray(x), "y": cupy.zeros_like(cupy.asarray(x)), "workspace": workspace}
    scalars = {**sizes, "workspace_size": nbytes}
    with tempfile.TemporaryDirectory() as directory:
        entry = ctypes.CDLL(str(nvcc_library(rendering, pathlib.Path(directory), name)))[name]
        entry.argtypes = [ctypes.c_void_p] * 2 + [ctypes.c_int] * 3 + [ctypes.c_void_p, ctypes.c_int64]
        entry(*[ctypes.c_void_p(device[a].data.ptr) if a in device else scalars[a] for a in PADDED_SIGNATURE.order])
    np.testing.assert_array_equal(cupy.asnumpy(device["y"]), 2 * x[:, :, ::-1] + 1)


def gpu_then_sync_sdfg() -> tuple[dace.SDFG, dace.SDFGState, dace.SDFGState]:
    """A kernel state, a default-stream wait spliced after it, a host state, and the wait that ends the program."""
    sdfg = dace.SDFG("cpf_nb_exit_sync")
    sdfg.add_array("a", [N], dace.float64, storage=dace.StorageType.GPU_Global)
    kernel = sdfg.add_state("kernel")
    kernel.add_mapped_tasklet(
        "twice",
        {"i": "0:N"},
        {"x": dace.Memlet("a[i]")},
        "y = 2 * x",
        {"y": dace.Memlet("a[i]")},
        schedule=dace.ScheduleType.GPU_Device,
        external_edges=True,
    )
    states = [kernel]
    for label in ("spliced", "host", "exit"):
        state = sdfg.add_state_after(states[-1], label)
        if label != "host":
            state.add_tasklet("wait", {}, {}, stream_sync_code([DEFAULT_STREAM]), dace.Language.CPP, side_effects=True)
        states.append(state)
    return sdfg, states[1], states[3]


def test_only_the_wait_that_ends_the_program_is_dropped():
    sdfg, spliced, exit_state = gpu_then_sync_sdfg()
    assert DropExitStreamSync().apply_pass(sdfg, {}) == 1
    assert exit_state not in sdfg.nodes()
    assert spliced in sdfg.nodes() and len(spliced.nodes()) == 1
    assert DropExitStreamSync().apply_pass(sdfg, {}) is None


def test_persistent_transients_of_the_top_level_sdfg_are_demoted():
    sdfg = dace.SDFG("cpf_nb_demote")
    persistent = dace.AllocationLifetime.Persistent
    sdfg.add_array("kept", [N], dace.float64, transient=True, lifetime=persistent)
    sdfg.add_array("argument", [N], dace.float64, lifetime=persistent)
    nested = dace.SDFG("cpf_nb_demote_inner")
    nested.add_array("inner", [N], dace.float64, transient=True, lifetime=persistent)
    sdfg.add_state().add_nested_sdfg(nested, {}, {})
    assert DemotePersistentTransients().apply_pass(sdfg, {}) == 1
    assert sdfg.arrays["kept"].lifetime is dace.AllocationLifetime.SDFG
    assert sdfg.arrays["argument"].lifetime is persistent
    assert nested.arrays["inner"].lifetime is persistent


def test_a_constant_the_kernel_and_the_entry_share_is_defined_once():
    """The device unit and the frame each wrote the SDFG's constants and size helpers; one unit holds them once
    (ext_break_capture, tsvc_2_s3112: hipcc refused the redefinition)."""
    sdfg = padded_transient_sdfg("cpf_nb_constant")
    sdfg.add_constant("SCALE", 2.0)
    for state in sdfg.states():
        for node in state.nodes():
            if isinstance(node, dace.sdfg.nodes.Tasklet) and node.label == "fill":
                node.code = dace.properties.CodeBlock("b = SCALE * a")
    rendering = render(sdfg, language="hip", signature=PADDED_SIGNATURE, no_boilerplate=True, check_compiles=False)
    definitions = [
        line for line in rendering.code.splitlines() if line.startswith(("constexpr ", "[[maybe_unused]] static"))
    ]
    assert definitions and len(definitions) == len(set(definitions)), rendering.code


if __name__ == "__main__":
    test_the_carve_sizes_a_padded_transient_by_its_strides()
    test_the_entry_carves_the_padded_transient_and_waits_once("cuda")
    test_the_entry_carves_the_padded_transient_and_waits_once("hip")
    test_the_padded_transient_runs_in_a_workspace_of_exactly_the_bound()
    test_only_the_wait_that_ends_the_program_is_dropped()
    test_persistent_transients_of_the_top_level_sdfg_are_demoted()
    test_a_constant_the_kernel_and_the_entry_share_is_defined_once()
