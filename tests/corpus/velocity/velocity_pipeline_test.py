# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""ICON velocity tendencies through the CPU pipeline: dace-fortran parses the Fortran, ``ConfigPropagation`` bakes
the namelist configuration into one external SDFG per combination, and each variant is parallelized and checked bit
for bit against the gfortran build of the same source.

Needs a dace-fortran checkout in ``DACE_FORTRAN_ROOT`` (its velocity translation unit and its e2e harness), with
dace-fortran installed over this DaCe. The ``dace-fortran`` workflow provides both.
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

import dace
from dace.sdfg.state import ConditionalBlock
from dace.transformation.passes.config_propagation import ConfigPropagation

pytestmark = pytest.mark.long

ENTRY = "mo_velocity_advection::velocity_tendencies"
DOMAINS = {"lvn_only": [0, 1], "istep": [1, 2], "ldeepatmo": [0], "lextra_diffu": [1]}
CONFIG_NAMES = tuple(DOMAINS)
#: Maps after parallelization per (lvn_only, istep); the unspecialized kernel has 61. Fewer maps is what
#: pruning the configuration branches buys, so a change here is a change in what the pipeline removes.
MAPS = {(0, 1): 51, (0, 2): 38, (1, 1): 38, (1, 2): 26}


def dace_fortran_root() -> Path:
    root = Path(os.environ.get("DACE_FORTRAN_ROOT", ""))
    assert (root / "dace_fortran").is_dir(), "DACE_FORTRAN_ROOT must name a dace-fortran checkout"
    return root


def velocity_source() -> Path:
    return dace_fortran_root() / "tests" / "icon" / "atmosphere" / "velocity_advection_inlined_single_tu.f90"


def load_dace_fortran_module(name: str, relative: str):
    """A dace-fortran test helper by path: this checkout's own ``tests`` package shadows dace-fortran's."""
    spec = importlib.util.spec_from_file_location(name, dace_fortran_root() / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def branch_conditions(sdfg: dace.SDFG) -> list[str]:
    return [
        cond.as_string
        for block in sdfg.all_control_flow_blocks(recursive=True)
        if isinstance(block, ConditionalBlock)
        for cond, _ in block.branches
        if cond is not None
    ]


def test_config_propagation_prunes_every_config_branch(tmp_path):
    from dace_fortran.build import build_sdfg

    sdfg = build_sdfg(velocity_source().read_text(), entry=ENTRY, name="velocity_tendencies", out_dir=str(tmp_path))
    before = len(branch_conditions(sdfg))
    out = tmp_path / "variants"
    root = ConfigPropagation(DOMAINS, str(out)).apply_pass(sdfg, {})

    dispatch = [b for b in root.nodes() if isinstance(b, ConditionalBlock)]
    assert len(dispatch) == 1 and len(dispatch[0].branches) == len(MAPS)
    files = sorted(p.name for p in out.iterdir())
    assert files == sorted(f"velocity_tendencies_lvn_only_{l}_istep_{s}.sdfgz" for l, s in MAPS)
    for name in files:
        variant = dace.SDFG.from_file(str(out / name))
        conditions = branch_conditions(variant)
        assert not [c for c in conditions if any(n in c for n in CONFIG_NAMES)], name
        assert len(conditions) < before, name


@pytest.mark.parametrize("lvn_only, istep", sorted(MAPS))
def test_cpu_pipeline_matches_gfortran(lvn_only, istep, monkeypatch):
    root = dace_fortran_root()
    harness = load_dace_fortran_module("dace_fortran_e2e", "tests/icon/ocean/_ocean_e2e.py")
    mesh = load_dace_fortran_module("dace_fortran_icon_mesh", "tests/icon/atmosphere/_icon_mesh.py")

    # The harness child imports dace-fortran's ``tests`` package and this hook by name
    paths = [str(root), str(Path(__file__).resolve().parent), os.environ.get("PYTHONPATH", "")]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(p for p in paths if p))
    monkeypatch.setenv("OCEAN_E2E_SDFG_HOOK", "velocity_hook:run")
    monkeypatch.setenv(
        "VT_CONFIG", json.dumps({"lvn_only": lvn_only, "istep": istep, "ldeepatmo": 0, "lextra_diffu": 1})
    )
    # A random 8 x 8 block patch with locality-biased connectivity, 7 full levels
    patch = mesh.velocity_patch(refin_extent=harness._REFIN_CTRL_EXTENT)
    # nflatlev is a DO lower bound, so its BSS 0 would start jk at 0, out of bounds
    patch["module_seeds"] |= {"nflatlev": 1, "nrdmax": 1, "lextra_diffu": 1}
    result = harness.run_kernel_e2e(
        velocity_source(),
        ENTRY,
        seed=1001,
        scalar_overrides={"lvn_only": lvn_only, "istep": istep, "ldeepatmo": 0},
        **patch,
    )
    assert result["passed"], result["output"][-3500:]
    assert f"VT_PIPELINE maps={MAPS[lvn_only, istep]} loops=0" in result["output"]
    assert result["n_changed"] > 0, "no output changed: the kernel did no work"
    assert result["max_diff"] == 0.0, f"differs from gfortran, max |d| = {result['max_diff']:.3e}"
