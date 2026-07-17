# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Per-nest translation units (``compiler.cpu.codegen_params.split_nsdfg_translation_units``).

Each top-level nest is outlined into a ``no_inline`` nested SDFG and emitted into its OWN ``.cpp``,
which the frame calls through a forward-declared ``DACE_HIDDEN`` in-binary function. The two things
worth guarding are that the flag is genuinely inert when off (byte-identical output, both CPU
generators) and that the split output actually LINKS and computes the same numbers.
"""
import re

import numpy as np
import pytest

import dace
from dace.codegen import codegen

M, N = 8, 16

IMPLEMENTATIONS = ['legacy', 'experimental_readable']


def two_nest_sdfg(name: str) -> dace.SDFG:
    """Two top-level maps that both read the SAME 2-D array ``A``.

    Sharing A is the point: under the readable generator every translation unit that indexes A needs
    its own ``A_idx`` helper, which is what the per-file dedup key has to get right once the nests
    stop sharing one file. 2-D on purpose -- a 1-D access needs no index helper at all.
    """
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [M, N], dace.float64)
    sdfg.add_array('B', [M, N], dace.float64)
    sdfg.add_array('C', [M, N], dace.float64)

    state = sdfg.add_state('s0', is_start_block=True)
    entry, exit_ = state.add_map('m0', dict(i='0:%d' % M, j='0:%d' % N))
    tasklet = state.add_tasklet('t0', {'a'}, {'b'}, 'b = a * 2.0')
    state.add_memlet_path(state.add_read('A'), entry, tasklet, dst_conn='a', memlet=dace.Memlet('A[i, j]'))
    state.add_memlet_path(tasklet, exit_, state.add_write('B'), src_conn='b', memlet=dace.Memlet('B[i, j]'))

    state2 = sdfg.add_state_after(state, 's1')
    entry2, exit2 = state2.add_map('m1', dict(k='0:%d' % M, l='0:%d' % N))
    tasklet2 = state2.add_tasklet('t1', {'a'}, {'c'}, 'c = a + 3.0')
    state2.add_memlet_path(state2.add_read('A'), entry2, tasklet2, dst_conn='a', memlet=dace.Memlet('A[k, l]'))
    state2.add_memlet_path(tasklet2, exit2, state2.add_write('C'), src_conn='c', memlet=dace.Memlet('C[k, l]'))
    return sdfg


def shared_inner_nest_sdfg(name: str) -> dace.SDFG:
    """Two DIFFERENT top-level nests that contain an IDENTICAL inner nested SDFG.

    Both top-level nests differ (one scales by 2, the other by 3) so both are emitted to their own
    TU; the inner nest (``io += 1``) is byte-identical between them, so under the default
    ``unique_functions='hash'`` it is generated once and deduplicated. If that dedup is keyed on the
    whole build rather than per-TU, the second TU gets a CALL to the inner function with no definition
    (it was emitted only into the first TU, and as ``inline`` it is not even linkable) -- so the split
    build fails to link. This is the fixture that exercises that path.
    """

    def inner() -> dace.SDFG:
        w = dace.SDFG('W')
        w.add_array('io', [1], dace.float64)
        st = w.add_state()
        t = st.add_tasklet('add1', {'i'}, {'o'}, 'o = i + 1.0')
        st.add_edge(st.add_read('io'), None, t, 'i', dace.Memlet('io[0]'))
        st.add_edge(t, 'o', st.add_write('io'), None, dace.Memlet('io[0]'))
        w.validate()
        return w

    def wrapper(mult: float) -> dace.SDFG:
        x = dace.SDFG('X')
        x.add_array('io', [1], dace.float64)
        s0 = x.add_state('w', is_start_block=True)
        n = s0.add_nested_sdfg(inner(), {'io'}, {'io'})
        n.no_inline = True
        s0.add_edge(s0.add_read('io'), None, n, 'io', dace.Memlet('io[0]'))
        s0.add_edge(n, 'io', s0.add_write('io'), None, dace.Memlet('io[0]'))
        s1 = x.add_state_after(s0, 'scale')
        t = s1.add_tasklet('mul', {'i'}, {'o'}, 'o = i * %s' % repr(mult))
        s1.add_edge(s1.add_read('io'), None, t, 'i', dace.Memlet('io[0]'))
        s1.add_edge(t, 'o', s1.add_write('io'), None, dace.Memlet('io[0]'))
        x.validate()
        return x

    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [1], dace.float64)
    sdfg.add_array('B', [1], dace.float64)
    s1 = sdfg.add_state('s0', is_start_block=True)
    n1 = s1.add_nested_sdfg(wrapper(2.0), {'io'}, {'io'})
    n1.no_inline = True
    s1.add_edge(s1.add_read('A'), None, n1, 'io', dace.Memlet('A[0]'))
    s1.add_edge(n1, 'io', s1.add_write('A'), None, dace.Memlet('A[0]'))
    s2 = sdfg.add_state_after(s1, 's1')
    n2 = s2.add_nested_sdfg(wrapper(3.0), {'io'}, {'io'})
    n2.no_inline = True
    s2.add_edge(s2.add_read('B'), None, n2, 'io', dace.Memlet('B[0]'))
    s2.add_edge(n2, 'io', s2.add_write('B'), None, dace.Memlet('B[0]'))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_nest_tu_defines_the_state_struct(implementation):
    """Every host TU gets the COMPLETE state struct, never a forward declaration.

    Streams and persistent-lifetime storage are fields of ``<name>_state_t``, so any nest may
    dereference ``__state``, and an incomplete type cannot be.
    """
    objects = generate('state_def_%s' % implementation, implementation, split=True)
    nests = nsdfg_objects(objects)
    assert nests, 'fixture produced no split nest TU -- the test would pass vacuously'
    for obj in nests:
        code = obj.clean_code or obj.code
        assert re.search(r'struct \w+_state_t\s*\{', code), 'nest TU must define the complete state struct'
        assert not re.search(r'struct \w+_state_t;', code), 'nest TU must not forward-declare it'
    frame = host_objects(objects)[0].clean_code
    assert re.search(r'struct \w+_state_t\s*\{', frame), 'frame TU still defines the state struct'


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_split_dedups_inner_nest_per_tu_and_links(implementation):
    """The default hash-dedup must not route an inner nest's call across a TU boundary without its
    definition. Compiling + linking + running is the proof -- a whole-build dedup key link-fails here."""
    with dace.config.set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         dace.config.set_temporary('compiler', 'cpu', 'codegen_params',
                                   'split_nsdfg_translation_units', value=True):
        sdfg = shared_inner_nest_sdfg('shared_inner_%s' % implementation)
        A = np.array([5.0])
        B = np.array([7.0])
        sdfg(A=A, B=B)
        assert np.allclose(A, (5.0 + 1.0) * 2.0)  # 12
        assert np.allclose(B, (7.0 + 1.0) * 3.0)  # 24


def generate(name: str, implementation: str, split: bool):
    with dace.config.set_temporary('compiler', 'cpu', 'implementation', value=implementation):
        with dace.config.set_temporary('compiler',
                                       'cpu',
                                       'codegen_params',
                                       'split_nsdfg_translation_units',
                                       value=split):
            return codegen.generate_code(two_nest_sdfg(name))


def host_objects(objects):
    """The frame ``.cpp`` (target_type '') -- the only host file when the split is off."""
    return [o for o in objects if o.language == 'cpp' and o.target_type == '']


def nsdfg_objects(objects):
    return [o for o in objects if o.target_type == 'nsdfg']


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_flag_off_single_host_tu(implementation):
    """Off (the default): one host TU carrying both nest bodies, and no split artifacts."""
    objects = generate('off_single', implementation, split=False)
    assert len(host_objects(objects)) == 1
    assert nsdfg_objects(objects) == []
    code = host_objects(objects)[0].clean_code
    assert 'DACE_HIDDEN' not in code
    # Both map bodies are still in this one file.
    assert code.count('#pragma omp parallel for') == 2


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_flag_off_is_byte_identical(implementation):
    """Off must not perturb a single byte. Every split path is gated on the flag, so generating with it
    explicitly off has to reproduce the generator's untouched default output exactly -- this is what
    lets the knob ship default-off without re-blessing any existing generated code."""
    with dace.config.set_temporary('compiler', 'cpu', 'implementation', value=implementation):
        baseline = host_objects(codegen.generate_code(two_nest_sdfg('ident')))[0].code
        off = host_objects(generate('ident', implementation, split=False))[0].code
    assert off == baseline


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_split_emits_one_tu_per_nest(implementation):
    """On: two nests -> two nsdfg CodeObjects; the frame keeps declarations + calls, not bodies."""
    objects = generate('split_two', implementation, split=True)
    nsdfgs = nsdfg_objects(objects)
    assert len(nsdfgs) == 2
    assert all(o.language == 'cpp' for o in nsdfgs)

    frame = host_objects(objects)[0].clean_code
    # Two forward declarations, each ending in ';' (a declaration, not a definition).
    decls = re.findall(r'DACE_HIDDEN\s+void\s+(\w+)\s*\([^;{]*\)\s*;', frame)
    assert len(decls) == 2, frame
    # ...and a call to each declared nest.
    for label in decls:
        assert re.search(r'^\s*%s\s*\(' % re.escape(label), frame, re.MULTILINE), label
    # No nest BODY in the frame: the loops moved out.
    assert '#pragma omp parallel for' not in frame
    for label in decls:
        assert not re.search(r'void\s+%s\s*\([^;{]*\)\s*\{' % re.escape(label), frame)


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_split_tus_are_self_contained(implementation):
    """Each nest TU must compile on its own: runtime include, the state struct it takes a pointer to,
    and its own function body. It must NOT include the frame-relative hash.h (wrong depth, and only
    frame code uses __HASH_*)."""
    objects = generate('selfcont', implementation, split=True)
    for obj in nsdfg_objects(objects):
        code = obj.clean_code
        assert '#include <dace/dace.h>' in code
        assert 'struct selfcont_state_t' in code
        assert 'hash.h' not in code
        assert re.search(r'DACE_HIDDEN\s+void\s+\w+\s*\([^;{]*\)\s*\{', code), obj.name


def test_split_readable_index_helpers_are_per_tu():
    """file_key regression (readable generator only).

    Both nests index the same array A. With the emitted-helper set keyed on the codegen instance,
    ``A_idx`` is recorded once for the whole host build, so the FIRST nest's TU defines it and every
    later nest's TU silently skips it -- leaving that TU calling an undefined ``A_idx``. Every TU that
    uses the helper must also define it.
    """
    objects = generate('peridx', 'experimental_readable', split=True)
    nsdfgs = nsdfg_objects(objects)
    assert len(nsdfgs) == 2
    for obj in nsdfgs:
        code = obj.clean_code
        uses = len(re.findall(r'\bA_idx\s*\(', code))
        defines = len(re.findall(r'constexpr\s+\w[\w\s]*\bA_idx\s*\(', code))
        assert uses > 0, f'{obj.name} should index A'
        assert defines == 1, f'{obj.name} uses A_idx {uses}x but defines it {defines}x'


@pytest.mark.parametrize('implementation', IMPLEMENTATIONS)
def test_split_compiles_links_and_matches_baseline(implementation):
    """The real gate: the split .so must LINK (cross-TU nest symbols resolve) and produce exactly the
    numbers the single-TU build produces."""
    rng = np.random.default_rng(0)
    A = rng.random((M, N))

    outputs = {}
    for split in (False, True):
        B = np.zeros((M, N))
        C = np.zeros((M, N))
        with dace.config.set_temporary('compiler', 'cpu', 'implementation', value=implementation):
            with dace.config.set_temporary('compiler',
                                           'cpu',
                                           'codegen_params',
                                           'split_nsdfg_translation_units',
                                           value=split):
                sdfg = two_nest_sdfg(f'link_{implementation}_{int(split)}')
                sdfg.compile()(A=A.copy(), B=B, C=C)
        outputs[split] = (B, C)

    # Correct against numpy, bit-exactly (no norm_error: these are exact fp operations).
    assert np.array_equal(outputs[True][0], A * 2.0)
    assert np.array_equal(outputs[True][1], A + 3.0)
    # ...and identical to the single-TU baseline.
    assert np.array_equal(outputs[True][0], outputs[False][0])
    assert np.array_equal(outputs[True][1], outputs[False][1])


if __name__ == '__main__':
    for impl in IMPLEMENTATIONS:
        test_flag_off_single_host_tu(impl)
        test_flag_off_is_byte_identical(impl)
        test_split_emits_one_tu_per_nest(impl)
        test_split_tus_are_self_contained(impl)
        test_split_compiles_links_and_matches_baseline(impl)
    test_split_readable_index_helpers_are_per_tu()
