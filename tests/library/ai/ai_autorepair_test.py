# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests for feeding a failed build back to the tasklets that caused it.

The probe compiles each generated tasklet alone, so it cannot see what only breaks once several of
them share a translation unit. What makes that usable is attribution: with more than one generated
tasklet in the SDFG, a diagnostic has to land on the one responsible, and a diagnostic that belongs
to no generated tasklet must not land on any of them.
"""

import contextlib
import os
import shutil
import sys

import numpy as np
import pytest

import dace
import dace.libraries.ai as ai
from dace.codegen import exceptions as cgx
from dace.libraries.ai import diagnose
from dace.libraries.ai.backend import TaskletSpec
from dace.libraries.ai.exceptions import AIExpansionError
from dace.libraries.ai.nodes import AINode

sys.path.insert(0, os.path.dirname(__file__))
from ai_test_utils import stub_provider  # noqa: E402

GOOD = '_out = 2.0 * _in;'
#: Passes a standalone probe and fails the real build: the name is simply not declared anywhere.
BAD = '_out = _in; undeclared_thing_here();'
#: Two tasklets emitting this collide, because the helper has external linkage.
COLLIDING_GLOBAL = 'double ai_shared_helper(double x) { return x + x; }'

pytestmark = pytest.mark.skipif(shutil.which('c++') is None and shutil.which('g++') is None,
                                reason='needs a host C++ compiler')


@contextlib.contextmanager
def sessions_in(tmp_path):
    """
    Runs with sessions on and verification off, in a temporary directory.

    :param tmp_path: The pytest temporary directory.
    """
    with dace.config.set_temporary('ai', 'sessions', value=True):
        with dace.config.set_temporary('ai', 'session_dir', value=str(tmp_path)):
            with dace.config.set_temporary('ai', 'verify', value=False):
                # With debugprint on, DaCe streams the compiler output live and leaves it out of
                # the exception, so a test that reads diagnostics from the exception has to turn it
                # off. ai.build() does the same thing internally, for the same reason.
                with dace.config.set_temporary('debugprint', value=False):
                    yield


def _two_slots(name: str, specs):
    """
    Builds an SDFG with two generated tasklets.

    :param name: Name of the SDFG.
    :param specs: One :class:`TaskletSpec` per slot, keyed by slot name.
    :return: A tuple of (SDFG, state).
    """
    sdfg = dace.SDFG(name)
    for array in 'ABCD':
        sdfg.add_array(array, [4], dace.float64)
    state = sdfg.add_state()
    for src, dst in (('A', 'B'), ('C', 'D')):
        node = AINode(f'n_{src}', 'Copy.', inputs={'_in'}, outputs={'_out'})
        state.add_node(node)
        me, mx = state.add_map(f'm_{src}', {f'i_{src}': '0:4'})
        state.add_memlet_path(state.add_read(src), me, node, dst_conn='_in', memlet=dace.Memlet(f'{src}[i_{src}]'))
        state.add_memlet_path(node, mx, state.add_write(dst), src_conn='_out', memlet=dace.Memlet(f'{dst}[i_{src}]'))
        with stub_provider(specs[node.name]):
            node.expand(state, 'ai')
    return sdfg, state


@pytest.mark.parametrize('culprit', ['n_A', 'n_C'])
def test_a_diagnostic_lands_on_the_tasklet_that_caused_it(tmp_path, culprit):
    with sessions_in(tmp_path):
        sdfg, _ = _two_slots(f'attrib_{culprit}',
                             {n: TaskletSpec(code=BAD if n == culprit else GOOD)
                              for n in ('n_A', 'n_C')})

        with pytest.raises((cgx.CompilationError, cgx.CompilerConfigurationError)) as info:
            sdfg.compile()

        found = diagnose.attribute(sdfg, diagnose.parse_diagnostics(str(info.value)))

    assert sorted(found.by_slot) == [culprit], 'the wrong tasklet was blamed'
    assert not found.unattributed


def test_a_collision_implicates_both_tasklets(tmp_path):
    """ The case that motivates all of this: neither tasklet is wrong on its own. """
    with sessions_in(tmp_path):
        sdfg, _ = _two_slots('collide',
                             {n: TaskletSpec(code=GOOD, code_global=COLLIDING_GLOBAL)
                              for n in ('n_A', 'n_C')})

        with pytest.raises((cgx.CompilationError, cgx.CompilerConfigurationError)) as info:
            sdfg.compile()

        found = diagnose.attribute(sdfg, diagnose.parse_diagnostics(str(info.value)))

    assert sorted(found.by_slot) == ['n_A', 'n_C'], 'a redefinition must implicate both definitions'

    # Each is told the other is involved, so it renames rather than assuming the other will
    message = diagnose._message_for('n_A', found.by_slot['n_A'], ['n_C'])
    assert 'ai_shared_helper' in message
    assert '"n_C"' in message and 'name collision' in message


def test_an_error_outside_generated_code_blames_nobody(tmp_path):
    with sessions_in(tmp_path):
        sdfg, state = _two_slots('innocent', {n: TaskletSpec(code=GOOD) for n in ('n_A', 'n_C')})

        # A hand-written tasklet, which no model produced and none should be asked to fix
        sdfg.add_array('E', [4], dace.float64)
        sdfg.add_array('F', [4], dace.float64)
        hand = state.add_tasklet('hand_written', {'_i'}, {'_o'}, '_o = not_declared_either(_i);',
                                 dace.dtypes.Language.CPP)
        me, mx = state.add_map('m_hand', {'k': '0:4'})
        state.add_memlet_path(state.add_read('E'), me, hand, dst_conn='_i', memlet=dace.Memlet('E[k]'))
        state.add_memlet_path(hand, mx, state.add_write('F'), src_conn='_o', memlet=dace.Memlet('F[k]'))

        with pytest.raises((cgx.CompilationError, cgx.CompilerConfigurationError)) as info:
            sdfg.compile()

        found = diagnose.attribute(sdfg, diagnose.parse_diagnostics(str(info.value)))
        assert not found.by_slot, 'an error in hand-written code was blamed on a generated tasklet'
        assert found.unattributed

        with pytest.raises(AIExpansionError, match='none of the errors is in code generated'):
            diagnose.repair(sdfg, info.value)

        # build() must surface the real compilation failure, not the reason it could not help
        with pytest.raises((cgx.CompilationError, cgx.CompilerConfigurationError)):
            ai.build(sdfg, rounds=1)


def test_build_repairs_and_succeeds(tmp_path):
    with sessions_in(tmp_path):
        sdfg, _ = _two_slots('repairable', {'n_A': TaskletSpec(code=BAD), 'n_C': TaskletSpec(code=GOOD)})

        with pytest.raises((cgx.CompilationError, cgx.CompilerConfigurationError)):
            sdfg.compile()

        with stub_provider(TaskletSpec(code=GOOD)) as provider:
            csdfg = ai.build(sdfg, rounds=1)

    assert csdfg is not None
    # Only the blamed tasklet was regenerated
    assert len(provider.calls) == 1
    assert 'did not compile' in provider.calls[0][-1]['content']

    slots = {s.name: s for s in ai.sessions(sdfg)}
    assert slots['n_A'].round == 2 and slots['n_C'].round == 1

    A, C = np.random.rand(4), np.random.rand(4)
    B, D = np.zeros(4), np.zeros(4)
    sdfg(A=A, B=B, C=C, D=D)
    assert np.allclose(B, 2 * A) and np.allclose(D, 2 * C)


def test_a_pinned_tasklet_is_left_alone(tmp_path):
    with sessions_in(tmp_path):
        sdfg, _ = _two_slots('pinned', {'n_A': TaskletSpec(code=BAD), 'n_C': TaskletSpec(code=GOOD)})
        ai.pin(sdfg, 'n_A')

        with pytest.raises((cgx.CompilationError, cgx.CompilerConfigurationError)) as info:
            sdfg.compile()

        with stub_provider(TaskletSpec(code=GOOD)) as provider:
            refined = diagnose.repair(sdfg, info.value)

    assert refined == [], 'a pinned tasklet was regenerated'
    assert provider.calls == []


def test_diagnostics_are_parsed_from_both_compiler_dialects():
    text = ('/tmp/build/prog.cpp:41:9: error: use of undeclared identifier \'foo\'\n'
            '/tmp/build/prog.cpp:12:5: note: previous definition is here\n'
            'C:\\build\\prog.cpp(41,9): error C2065: undeclared identifier\n'
            'make: *** [all] Error 1\n'
            'cc1plus: some unpositioned warning\n')

    found = diagnose.parse_diagnostics(text)
    assert [(d.line, d.severity) for d in found] == [(41, 'error'), (12, 'note'), (41, 'error')]
    assert found[0].column == 9
    assert 'undeclared identifier' in found[0].message
    # Lines that name no position are not diagnostics
    assert all('make' not in d.message for d in found)


def test_the_line_index_covers_the_body_and_the_global(tmp_path):
    """
    Both blocks must be attributable.

    DaCe's own ``map_cpp.json`` keeps only the first contiguous run of lines per node, so for a
    tasklet with a ``code_global`` it describes the global block and omits the body -- which is
    where compile errors usually are. The index built here must not have that gap.
    """
    with sessions_in(tmp_path):
        sdfg, state = _two_slots(
            'index', {
                'n_A':
                TaskletSpec(code='_out = marker_helper(_in);',
                            code_global='static double marker_helper(double x) { return x; }'),
                'n_C':
                TaskletSpec(code=GOOD),
            })

    index = diagnose.build_line_index(sdfg)
    code = sdfg.generate_code()[0]
    lines = code.clean_code.split('\n')

    body_line = next(i for i, line in enumerate(lines, 1) if 'marker_helper(_in)' in line)
    global_line = next(i for i, line in enumerate(lines, 1) if 'return x;' in line)

    name = f'{code.name}.{code.language}'
    slots = {s.name: s for s in ai.sessions(sdfg)}
    target = slots['n_A'].tasklet
    for label, line in (('body', body_line), ('code_global', global_line)):
        located = index.get((name, line), set())
        assert any(diagnose._resolve(sdfg, loc) is target for loc in located), \
            f'the {label} of "n_A" is not attributable at line {line}'


if __name__ == '__main__':
    pytest.main([__file__])
