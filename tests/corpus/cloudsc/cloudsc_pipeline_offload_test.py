# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the opt-in GPU-offload gate in :mod:`tests.corpus.cloudsc.pipelines`.

Fast and in-process: the phase plans are built (no CloudSC parse, no pipeline run) and the offload
stage is applied to a tiny hand-built blocked SDFG, which is then CUDA-code-generated. No GPU needed --
code generation is host-side, and a graph that is not device-scheduled emits no ``.cu`` at all, so the
codegen assertion is not vacuous (``test_cuda_codegen_check_rejects_a_host_graph`` pins that).

    pytest tests/corpus/cloudsc/cloudsc_pipeline_offload_test.py -v
"""
import contextlib
import importlib
import io
import sys
import tempfile
import types
from pathlib import Path

import pytest

from tests.corpus.cloudsc import pipelines
from tests.corpus.cloudsc.cloudsc_offload_to_gpu_test import blocked_sdfg
from tests.corpus.cloudsc.pipelines import (OFFLOAD_PHASE, OFFLOAD_VARIANTS, VARIANTS, generate_cuda_code,
                                            load_offload_pass, run_pipeline, variant_phases)


def phase_names(variant: str, offload: bool):
    return [name for name, _ in variant_phases(variant, offload=offload)]


@pytest.mark.parametrize('variant', VARIANTS)
def test_gate_defaults_off(variant):
    """No offload phase unless asked for, and the default matches an explicit ``offload=False``."""
    assert OFFLOAD_PHASE not in phase_names(variant, offload=False)
    assert [name for name, _ in variant_phases(variant)] == phase_names(variant, offload=False)


@pytest.mark.parametrize('variant', OFFLOAD_VARIANTS)
def test_gate_on_appends_offload_phase(variant):
    """The offload phase is terminal, holds exactly the one stage, and leaves the plan before it
    untouched -- so every earlier phase is still numeric-checked exactly as before."""
    off = variant_phases(variant, offload=False)
    on = variant_phases(variant, offload=True)
    assert len(on) == len(off) + 1
    assert [(name, len(stages)) for name, stages in on[:-1]] == [(name, len(stages)) for name, stages in off]
    name, stages = on[-1]
    assert name == OFFLOAD_PHASE
    assert [label for label, _ in stages] == ['offload_to_gpu']


def test_canon_cpu_ignores_the_gate():
    """A CPU recipe must not silently turn into a GPU graph."""
    assert 'canon_cpu' not in OFFLOAD_VARIANTS
    assert phase_names('canon_cpu', offload=True) == phase_names('canon_cpu', offload=False)


@pytest.mark.parametrize('variant', OFFLOAD_VARIANTS)
def test_offload_stage_offloads_and_generates_cuda(variant):
    """Applying the appended stage to a CloudSC-shaped blocked SDFG leaves a validating,
    device-scheduled graph that code-generates a CUDA kernel -- the check the offload phase runs in
    place of ``numeric_check``."""
    sdfg = blocked_sdfg()
    _label, apply_fn = variant_phases(variant, offload=True)[-1][1][0]
    apply_fn(sdfg)
    sdfg.validate()
    assert 'gpu_pin' in sdfg.arrays and 'gpu_pout' in sdfg.arrays
    assert generate_cuda_code(sdfg) == 1


def test_run_pipeline_reports_how_the_offload_phase_was_checked():
    """The full ``run_pipeline`` wiring, on a tiny blocked SDFG (sub-second -- not CloudSC): the
    offload phase runs last, a wired ``numeric_check`` is called for every earlier phase, and the
    offload line says outright whether the graph was RUN on the device or only code-generated. Both
    outcomes are legal -- which one happens is a property of the host, not of the pipeline -- but they
    must never be confusable, and a device run must have called ``numeric_check`` for the phase."""
    checked = []
    sdfg = blocked_sdfg()
    sdfg.name = 'pipeline_offload_probe'
    with tempfile.TemporaryDirectory() as dump:
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            run_pipeline(sdfg,
                         'parallelize',
                         Path(dump),
                         tag='probe',
                         numeric_check=lambda _sdfg, phase: checked.append(phase),
                         resume=False,
                         offload=True)
    printed = buffer.getvalue()
    assert checked, 'phases before offload must still be numeric-checked'
    assert '__global__ kernel(s)' in printed, printed
    on_device = f'numeric[{OFFLOAD_PHASE}]: VERIFIED ON DEVICE' in printed
    structural = f'numeric[{OFFLOAD_PHASE}]: NOT RUN ON DEVICE' in printed
    assert on_device != structural, f'the offload line must claim exactly one of the two:\n{printed}'
    assert on_device == (OFFLOAD_PHASE in checked), (f'claimed device run={on_device} but numeric_check '
                                                     f'called for the phase={OFFLOAD_PHASE in checked}')


def test_structural_fallback_is_what_a_gpu_less_host_gets(monkeypatch):
    """With no usable GPU the phase must still be checked, structurally, and say so -- CloudSC CPU
    runs cannot be made to depend on CUDA. Forced rather than assumed, so this pins the fallback on a
    GPU box too."""
    monkeypatch.setattr(pipelines, 'gpu_is_runnable', lambda: False)
    sdfg = blocked_sdfg()
    load_offload_pass()(sdfg)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        numeric = pipelines.check_offload_phase(sdfg, lambda _sdfg, _phase: None)
    assert numeric is False
    assert f'numeric[{OFFLOAD_PHASE}]: NOT RUN ON DEVICE (no usable GPU on this host)' in buffer.getvalue()


def test_structural_fallback_when_no_numeric_check_is_wired():
    """``numeric_check=None`` is structural-only everywhere else in the pipeline; the offload phase
    must not silently invent a check it was not given."""
    sdfg = blocked_sdfg()
    load_offload_pass()(sdfg)
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        numeric = pipelines.check_offload_phase(sdfg, None)
    assert numeric is False
    assert f'numeric[{OFFLOAD_PHASE}]: NOT RUN ON DEVICE (no numeric_check wired)' in buffer.getvalue()


def test_cuda_codegen_check_rejects_a_host_graph():
    """The codegen check has teeth: a CPU-scheduled graph emits no ``.cu``, so it fails."""
    with pytest.raises(AssertionError, match='not device-scheduled'):
        generate_cuda_code(blocked_sdfg())


def test_offload_pass_resolves_when_pipelines_is_loaded_by_path(monkeypatch):
    """The dace-fortran variant matrix exec's pipelines.py BY PATH, where its own top-level ``tests``
    package shadows this one and the dotted import fails. The by-path fallback must still find the
    pass, or the 3 Fortran SDFGs could never be offloaded."""
    shadow = types.ModuleType('tests')
    shadow.__path__ = []
    for name in [n for n in sys.modules if n.startswith('tests.')]:
        monkeypatch.delitem(sys.modules, name)
    monkeypatch.setitem(sys.modules, 'tests', shadow)
    with pytest.raises(ImportError):
        importlib.import_module('tests.corpus.cloudsc.offload_cloudsc_to_gpu')
    assert load_offload_pass().__name__ == 'offload_cloudsc_to_gpu'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
