# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Tests that a generated tasklet is a tasklet in every respect that matters.

:class:`~dace.libraries.ai.nodes.ai_tasklet.AITasklet` adds one thing to
:class:`~dace.sdfg.nodes.Tasklet`: a record of what produced it. Everything else -- code
generation, execution, serialization -- has to behave identically, and the ways a node subclass can
quietly fail to do so are specific and worth pinning down.
"""

import os
import shutil
import subprocess
import sys
import textwrap

import numpy as np
import pytest

import dace
from dace import nodes
from dace.libraries.ai.nodes import AITasklet

needs_compiler = pytest.mark.skipif(shutil.which('c++') is None and shutil.which('g++') is None,
                                    reason='needs a host C++ compiler')

PROVENANCE = '{"session": "demo.double", "round": 2}'


def _build(name: str = 'aitasklet'):
    """
    Builds a mapped SDFG whose body is an :class:`AITasklet`.

    :param name: Name of the SDFG.
    :return: The SDFG.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [16], dace.float64)
    sdfg.add_array('B', [16], dace.float64)
    state = sdfg.add_state()
    tasklet = AITasklet('double', {'_in'}, {'_out'}, '_out = 2.0 * _in;', dace.dtypes.Language.CPP)
    tasklet.provenance = PROVENANCE
    state.add_node(tasklet)
    me, mx = state.add_map('m', {'i': '0:16'})
    state.add_memlet_path(state.add_read('A'), me, tasklet, dst_conn='_in', memlet=dace.Memlet('A[i]'))
    state.add_memlet_path(tasklet, mx, state.add_write('B'), src_conn='_out', memlet=dace.Memlet('B[i]'))
    return sdfg


@needs_compiler
def test_it_generates_code_and_runs():
    """
    The code generator dispatches on the exact class name, so a subclass needs its own entry.

    Without the ``_generate_AITasklet`` alias in :mod:`dace.codegen.targets.cpu` this raises
    ``AttributeError`` during code generation.
    """
    sdfg = _build()
    A = np.random.rand(16)
    B = np.zeros(16)
    sdfg(A=A, B=B)
    assert np.allclose(B, 2 * A)


def test_it_round_trips_as_itself(tmp_path):
    """
    Serialization must preserve the type, not just the code.

    :class:`~dace.sdfg.nodes.RTLTasklet` fails this: it reports ``__jsontype__`` as ``'Tasklet'``
    and comes back as one, silently dropping its own properties.
    """
    sdfg = _build()
    path = str(tmp_path / 'saved.sdfg')
    sdfg.save(path)

    loaded = dace.SDFG.from_file(path)
    node = next(n for s in loaded.states() for n in s.nodes() if isinstance(n, nodes.Tasklet))
    assert type(node) is AITasklet
    assert node.provenance == PROVENANCE


@needs_compiler
def test_a_fresh_process_can_load_it(tmp_path):
    """
    Loading must work in a process that has only imported ``dace``.

    Nothing imports ``dace.libraries.ai`` at startup -- libraries are loaded on demand -- so the
    class is not registered with the serializer and the node would come back as an unresolved stub,
    failing the whole load. The ``classpath`` recorded by ``AITasklet.to_json`` is what lets
    ``dace.serialize`` import the module and resolve it.
    """
    sdfg = _build('fresh_load')
    path = str(tmp_path / 'fresh.sdfg')
    sdfg.save(path)

    script = textwrap.dedent(f"""
        import numpy as np
        import dace
        assert 'dace.libraries.ai' not in __import__('sys').modules, 'the AI library was imported eagerly'
        sdfg = dace.SDFG.from_file({path!r})
        node = next(n for s in sdfg.states() for n in s.nodes() if isinstance(n, dace.nodes.Tasklet))
        assert type(node).__name__ == 'AITasklet', type(node).__name__
        assert node.provenance == {PROVENANCE!r}
        A = np.random.rand(16); B = np.zeros(16)
        sdfg(A=A, B=B)
        assert np.allclose(B, 2 * A)
        print('OK')
        """)

    env = dict(os.environ, DACE_debugprint='false')
    result = subprocess.run([sys.executable, '-c', script], capture_output=True, text=True, env=env, timeout=600)
    assert 'OK' in result.stdout, f'stdout:\n{result.stdout}\nstderr:\n{result.stderr}'


def test_a_plain_tasklet_is_unaffected(tmp_path):
    """ Adding the subclass must not change how an ordinary tasklet serializes. """
    sdfg = dace.SDFG('plain')
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    state = sdfg.add_state()
    tasklet = state.add_tasklet('t', {'_in'}, {'_out'}, '_out = _in;', dace.dtypes.Language.CPP)
    state.add_edge(state.add_read('A'), None, tasklet, '_in', dace.Memlet('A[0]'))
    state.add_edge(tasklet, '_out', state.add_write('B'), None, dace.Memlet('B[0]'))

    serialized = tasklet.to_json(state)
    assert serialized['type'] == 'Tasklet'
    assert 'classpath' not in serialized
    assert 'provenance' not in serialized['attributes']

    path = str(tmp_path / 'plain.sdfg')
    sdfg.save(path)
    loaded = dace.SDFG.from_file(path)
    node = next(n for s in loaded.states() for n in s.nodes() if isinstance(n, nodes.Tasklet))
    assert type(node) is nodes.Tasklet


def test_an_unresolvable_classpath_does_not_crash_the_load():
    """ A node whose defining module is gone must degrade, not take the SDFG with it. """
    from dace import serialize

    with pytest.raises(KeyError):
        serialize._resolve_serializer('NoSuchType', {'classpath': 'no.such.module.NoSuchType'})
    with pytest.raises(KeyError):
        serialize._resolve_serializer('NoSuchType', {})
    # A known type is returned without consulting the classpath at all
    assert serialize._resolve_serializer('Tasklet', {}) is nodes.Tasklet


if __name__ == '__main__':
    pytest.main([__file__])
