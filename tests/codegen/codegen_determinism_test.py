# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""generate_code must emit the same source for the same SDFG, run to run.

Six independently-named struct dtypes with no dependency on each other expose the bug: framecode's
custom-type emission iterated a plain ``set()`` of typeclasses, and ``typeclass.__hash__`` folds in
``hash(self.type)`` -- for a struct that is ``hash(ctypes.Structure)``, the default id()-based object
hash. ``id()`` of a module-level singleton class moves with ASLR from one process to the next, so the
struct definitions came out in a different order across runs even with PYTHONHASHSEED pinned
(PYTHONHASHSEED only seeds str/bytes hashing, not the default object hash). Two structs already flip
order some of the time; six make a same-order coincidence between independent processes vanishingly
unlikely, so the test is not a coin flip.
"""
import os
import subprocess
import sys
import textwrap

import dace
from dace.codegen import codegen

_STRUCT_NAMES = ['S_ALPHA', 'S_BRAVO', 'S_CHARLIE', 'S_DELTA', 'S_ECHO', 'S_FOXTROT']


def _build_sdfg() -> dace.SDFG:
    structs = [dace.struct(name, x=dace.float64, y=dace.int32) for name in _STRUCT_NAMES]

    sdfg = dace.SDFG('codegen_determinism_repro')
    state = sdfg.add_state()
    for i, s in enumerate(structs):
        sdfg.add_array(f'in_{i}', [2], dtype=s)
        sdfg.add_array(f'out_{i}', [2], dtype=s)
        t = state.add_tasklet(f'copy_{i}', {'ia'}, {'oa': dace.pointer(s)},
                              'oa->x = ia.x; oa->y = ia.y;',
                              language=dace.Language.CPP)
        r = state.add_read(f'in_{i}')
        w = state.add_write(f'out_{i}')
        state.add_edge(r, None, t, 'ia', dace.Memlet.simple(f'in_{i}', '0'))
        state.add_edge(t, 'oa', w, None, dace.Memlet.simple(f'out_{i}', '0'))

    sdfg.validate()
    return sdfg


def _generate_source(sdfg: dace.SDFG) -> str:
    code_objects = codegen.generate_code(sdfg)
    return '\n'.join(co.clean_code for co in code_objects)


# Self-contained: run in a fresh interpreter via `-c`, so it does not depend on this test module
# being importable under whatever sys.path pytest collected it with.
_SUBPROCESS_SCRIPT = textwrap.dedent('''
    import dace
    from dace.codegen import codegen

    names = ['S_ALPHA', 'S_BRAVO', 'S_CHARLIE', 'S_DELTA', 'S_ECHO', 'S_FOXTROT']
    structs = [dace.struct(name, x=dace.float64, y=dace.int32) for name in names]

    sdfg = dace.SDFG('codegen_determinism_repro')
    state = sdfg.add_state()
    for i, s in enumerate(structs):
        sdfg.add_array(f'in_{i}', [2], dtype=s)
        sdfg.add_array(f'out_{i}', [2], dtype=s)
        t = state.add_tasklet(f'copy_{i}', {'ia'}, {'oa': dace.pointer(s)}, 'oa->x = ia.x; oa->y = ia.y;',
                               language=dace.Language.CPP)
        r = state.add_read(f'in_{i}')
        w = state.add_write(f'out_{i}')
        state.add_edge(r, None, t, 'ia', dace.Memlet.simple(f'in_{i}', '0'))
        state.add_edge(t, 'oa', w, None, dace.Memlet.simple(f'out_{i}', '0'))

    sdfg.validate()
    code_objects = codegen.generate_code(sdfg)
    import sys
    sys.stdout.write('\\n'.join(co.clean_code for co in code_objects))
    ''')

_SUBPROCESS_SAMPLES = 5


def _generate_in_subprocess(hashseed) -> str:
    env = dict(os.environ)
    env['CUDA_VISIBLE_DEVICES'] = ''
    if hashseed is None:
        env.pop('PYTHONHASHSEED', None)
    else:
        env['PYTHONHASHSEED'] = hashseed
    result = subprocess.run([sys.executable, '-c', _SUBPROCESS_SCRIPT],
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=120)
    assert result.returncode == 0, result.stderr
    return result.stdout


def _assert_stable_across_processes(hashseed):
    samples = [_generate_in_subprocess(hashseed) for _ in range(_SUBPROCESS_SAMPLES)]
    first = samples[0]
    for i, sample in enumerate(samples[1:], start=1):
        assert sample == first, (f'generate_code produced different source in fresh-process sample {i} than sample 0 '
                                 f'for the same SDFG (PYTHONHASHSEED={hashseed!r})')


def test_generate_code_same_process_is_deterministic():
    sdfg = _build_sdfg()
    first = _generate_source(sdfg)
    second = _generate_source(sdfg)
    assert first == second


def test_generate_code_deterministic_across_processes_pythonhashseed_0():
    _assert_stable_across_processes('0')


def test_generate_code_deterministic_across_processes_pythonhashseed_unset():
    _assert_stable_across_processes(None)


if __name__ == '__main__':
    test_generate_code_same_process_is_deterministic()
    test_generate_code_deterministic_across_processes_pythonhashseed_0()
    test_generate_code_deterministic_across_processes_pythonhashseed_unset()
