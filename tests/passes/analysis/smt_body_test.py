# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for the tasklet-body-to-SMT encoder.

Every value test runs the SAME source through Python and through the encoder and compares. The
encoder's whole job is to agree with Python's semantics on the fragment it accepts, so an oracle
derived any other way would be testing the derivation.
"""
import pytest

from dace.transformation.passes.analysis import smt_body

pytestmark = pytest.mark.skipif(not smt_body.HAS_Z3, reason='needs z3')

WIDTH = 64


def evaluate(code: str, name: str, **inputs: int):
    """Encode ``code`` with concrete inputs and read ``name`` back as a Python int."""
    env = {k: smt_body.constant(v, WIDTH) for k, v in inputs.items()}
    after = smt_body.encode_body(code, env, WIDTH)
    if after is None or name not in after:
        return None
    folded = smt_body.z3.simplify(after[name])
    assert smt_body.z3.is_bv_value(folded), f'{name} did not fold to a value: {folded}'
    return folded.as_long()


def run_python(code: str, name: str, **inputs: int):
    scope = dict(inputs)
    exec(compile(code, '<body>', 'exec'), {}, scope)
    return scope[name]


def agree(code: str, name: str, **inputs: int):
    """Encoder and Python agree MODULO 2**WIDTH.

    Not an escape hatch: the encoder models fixed-width wrapping arithmetic, which is what the
    emitted unsigned C does, while a Python int is unbounded and goes negative. The two agree on
    every body that masks its own result -- the fragment this encoder is for -- and the wrap itself
    is pinned separately in :func:`test_subtraction_wraps_it_does_not_go_negative`.
    """
    encoded, native = evaluate(code, name, **inputs), run_python(code, name, **inputs)
    assert encoded == native % (1 << WIDTH), f'encoder {encoded} != python {native} for {inputs}'


@pytest.mark.parametrize('a,b', [(0, 0), (1, 0), (0xFF, 0x0F), (0x1234, 0x5678), (0xFFFF, 1)])
def test_bitwise_and_arithmetic_agree_with_python(a, b):
    for expr in ('x & y', 'x | y', 'x ^ y', 'x + y', 'x - y', 'x * y', 'x << 3'):
        agree(f'z = {expr}\n', 'z', x=a, y=b)


def test_subtraction_wraps_it_does_not_go_negative():
    """The encoder's arithmetic is the machine's, not Python's. ``2 - 3`` is ``2**WIDTH - 1`` here
    and ``-1`` in Python, so a caller reasoning about a body that can go negative WITHOUT masking is
    reasoning about a different program than the one that will run. Bodies in this fragment mask.
    """
    assert evaluate('z = x - y\n', 'z', x=2, y=3) == (1 << WIDTH) - 1
    assert run_python('z = x - y\n', 'z', x=2, y=3) == -1
    # Masked, the two agree again -- which is why the fragment is stated in terms of masking bodies.
    assert evaluate('z = (x - y) & 0xFF\n', 'z', x=2, y=3) == run_python('z = (x - y) & 0xFF\n', 'z', x=2, y=3)


def test_right_shift_is_logical_not_arithmetic():
    """The documented trap. Python's ``>>`` on a non-negative int shifts in ZEROS; z3's ``>>`` is
    arithmetic and smears the sign bit. With the top bit of the encoding set, the two differ in
    every bit -- and on a CRC that is every iteration after the first."""
    top = 1 << (WIDTH - 1)
    agree('z = x >> 1\n', 'z', x=top)
    assert evaluate('z = x >> 1\n', 'z', x=top) == top >> 1
    assert evaluate('z = x >> 4\n', 'z', x=top) == top >> 4


def test_augmented_assignment_reads_the_current_value():
    agree('s ^= d\ns += 1\ns &= 0xFF\n', 's', s=0x5A, d=0x0F)


@pytest.mark.parametrize('crc', [0x0000, 0x8000, 0x1234, 0xFFFF])
def test_branch_on_a_bit_agrees_with_python(crc):
    """A body may branch on ``crc & 0x8000`` directly -- a bitvector, not a bool. Truthiness is
    non-zero, as in Python."""
    body = ('if crc & 0x8000:\n'
            '    crc = ((crc << 1) ^ 0x1021) & 0xFFFF\n'
            'else:\n'
            '    crc = (crc << 1) & 0xFFFF\n')
    agree(body, 'crc', crc=crc)


def test_name_assigned_on_one_branch_only_keeps_the_incoming_value():
    """The merge must select per name. ``b`` is written only when the branch is taken, so on the
    other path it has to come out as it went in -- dropping it, or defaulting it to zero, is a
    silent miscompile that concrete tests on the taken path would never catch."""
    body = ('if d:\n'
            '    b = b ^ 0xFF\n')
    for d in (0, 1):
        agree(body, 'b', b=0x0F, d=d)


def test_nested_branches_agree_with_python():
    body = ('if x & 1:\n'
            '    if y & 1:\n'
            '        r = x + y\n'
            '    else:\n'
            '        r = x - y\n'
            'else:\n'
            '    r = x ^ y\n')
    for x in (0, 1, 2, 3):
        for y in (0, 1, 2, 3):
            agree(body, 'r', r=0, x=x, y=y)


def test_conditional_expression_agrees_with_python():
    for d in (0, 5):
        agree('z = 1 if d else 2\n', 'z', d=d)


def test_unsupported_constructs_are_refused():
    """Refusing is a correct answer for an oracle; guessing is not. A loop, a call, a subscript and
    a tuple target each leave the fragment."""
    env = {'s': smt_body.bitvec('s', WIDTH), 'd': smt_body.bitvec('d', WIDTH)}
    for code in ('for k in range(4):\n    s = s ^ 1\n', 'while s:\n    s = s - 1\n', 's = min(s, d)\n', 's = t[d]\n',
                 's, d = d, s\n', 's = s / d\n', 's = 1.5\n'):
        assert smt_body.encode_body(code, dict(env), WIDTH) is None, f'should have refused: {code!r}'


def test_reading_a_name_absent_from_the_environment_is_refused():
    """A body reading something the caller did not bind is unsupported, not implicitly zero."""
    assert smt_body.encode_body('s = s ^ unbound\n', {'s': smt_body.bitvec('s', WIDTH)}, WIDTH) is None


def test_syntax_error_is_refused_rather_than_raised():
    assert smt_body.encode_body('s = = 1\n', {'s': smt_body.bitvec('s', WIDTH)}, WIDTH) is None


def test_indented_body_is_accepted():
    """Tasklet sources arrive indented; the encoder strips the outer indentation."""
    assert evaluate('    z = x ^ y\n', 'z', x=0xF0, y=0x0F) == 0xFF
