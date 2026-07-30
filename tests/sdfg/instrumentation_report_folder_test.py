# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``SDFG.is_instrumented()`` correctness and creation of the ``perf/``
report folder.

The runtime's ``report.save()`` opens ``<build_folder>/perf/report-*.json``
with a bare ``std::ofstream``, which neither creates directories nor reports a
failed open - if ``perf/`` is missing, every instrumentation report is dropped
silently. It therefore must be created for every instrumented SDFG, in BOTH
folder modes ('production' used to skip it entirely), and only for
instrumented SDFGs (uninstrumented folders stay lean).
"""

import os

import dace
from dace import dtypes


def _make_sdfg(name: str) -> dace.SDFG:
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [16], dace.float64)
    state = sdfg.add_state()
    state.add_mapped_tasklet(
        'work',
        map_ranges={'i': '0:16'},
        inputs={'__in': dace.Memlet('A[i]')},
        outputs={'__out': dace.Memlet('A[i]')},
        code='__out = __in + 1.0',
        external_edges=True,
    )
    return sdfg


def test_is_instrumented_distinguishes_enum_kinds():
    # AccessNodes carry a DataInstrumentationType whose No_Instrumentation is
    # a DIFFERENT enum member than InstrumentationType's: a cross-enum
    # comparison made every SDFG with an access node count as instrumented.
    sdfg = _make_sdfg('instr_probe')
    assert not sdfg.is_instrumented()

    sdfg.instrument = dtypes.InstrumentationType.Timer
    assert sdfg.is_instrumented()
    sdfg.instrument = dtypes.InstrumentationType.No_Instrumentation
    assert not sdfg.is_instrumented()

    state = next(iter(sdfg.states()))
    state.instrument = dtypes.InstrumentationType.Timer
    assert sdfg.is_instrumented()
    state.instrument = dtypes.InstrumentationType.No_Instrumentation
    assert not sdfg.is_instrumented()

    # Data instrumentation counts as well - against its OWN enum.
    an = next(n for n in state.nodes() if isinstance(n, dace.nodes.AccessNode))
    an.instrument = dtypes.DataInstrumentationType.Save
    assert sdfg.is_instrumented()
    an.instrument = dtypes.DataInstrumentationType.No_Instrumentation
    assert not sdfg.is_instrumented()


def test_perf_folder_created_exactly_when_instrumented(tmp_path):
    # Codegen-only (no compiler involved): generate_program_folder is the
    # function that owns the decision.
    from dace.codegen import codegen, compiler

    for folder_mode in ('development', 'production'):
        for instrumented in (False, True):
            sdfg = _make_sdfg(f'perfdir_{folder_mode}_{int(instrumented)}')
            if instrumented:
                sdfg.instrument = dtypes.InstrumentationType.Timer
            objects = codegen.generate_code(sdfg)
            out = str(tmp_path / sdfg.name)
            compiler.generate_program_folder(sdfg, objects, out, folder_mode=folder_mode)
            assert os.path.isdir(os.path.join(out, 'perf')) == instrumented, \
                f'perf/ existence mismatch for folder_mode={folder_mode}, instrumented={instrumented}'


if __name__ == '__main__':
    test_is_instrumented_distinguishes_enum_kinds()
    import tempfile, pathlib
    test_perf_folder_created_exactly_when_instrumented(pathlib.Path(tempfile.mkdtemp()))
