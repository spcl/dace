# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The MPR lowering table, checked against a compiler rather than against itself.

A mapping table is easy to write and easy to get quietly wrong, so nothing here asserts that a
lowering equals some expected string. Instead each entry is compiled into a real translation unit
with a bare host compiler and RUN, and its result compared against the semantics the DaCe runtime
header documents. A rename to the wrong ``std`` function, a rewrite that loses a type, or a
definition that does not build all fail as a wrong number or a failed compile.

Both dialects are held to the same standard, and mostly by the same tests: the C tables are a
SECOND spelling of the same semantics, so a C entry that computes something else is exactly the
failure a table written twice invites. The value cases below are therefore parametrized over the
dialect rather than duplicated.

The coverage tests at the bottom are the ones that stop the tables rotting: every unqualified
runtime function the harness knows how to detect must be handled in some lane or explicitly refused,
every C definition must be reachable from something a printer emits, and every C++ entry must have a
C counterpart or an entry saying why it cannot.
"""
import ctypes
import textwrap

import numpy as np
import pytest

from dace import mpr_lowering
from dace.mpr_lowering import Dialect
from tests.codegen.mpr.conftest import (UNQUALIFIED_RUNTIME_FUNCTIONS, assert_standalone, build_standalone,
                                        compile_diagnostics, compile_standalone)

#: The two standalone dialects, and what the harness calls each one's compiler.
DIALECTS = (Dialect.STANDALONE, Dialect.STANDALONE_C)
LANGUAGE = {Dialect.STANDALONE: 'c++', Dialect.STANDALONE_C: 'c'}
DIALECT_IDS = [LANGUAGE[dialect] for dialect in DIALECTS]

#: ``(name, printed arguments, C++ literal the call must equal)``. The expected values come from
#: the runtime header's documented behaviour, not from re-running the lowering.
CASES = [
    ('Abs', ('-3.5', ), '3.5'),
    ('ceiling', ('2.25', ), '3.0'),
    ('floor', ('2.75', ), '2.0'),
    ('ROUND', ('2.5', ), '3.0'),
    ('ROUND', ('-2.5', ), '-3.0'),
    ('iround', ('2.5', ), '3'),
    ('reciprocal', ('4.0', ), '0.25'),
    ('sign', ('-2.5', ), '-1.0'),
    ('sign', ('0.0', ), '0.0'),
    ('sgn', ('7.5', ), '1.0'),
    ('ITE', ('1 > 0', '11.0', '22.0'), '11.0'),
    ('ITE', ('1 < 0', '11.0', '22.0'), '22.0'),
    ('IfExpr', ('1 < 0', '11.0', '22.0'), '22.0'),
    ('heaviside', ('-1.0', '0.5'), '0.0'),
    ('heaviside', ('0.0', '0.5'), '0.5'),
    ('heaviside', ('1.0', '0.5'), '1.0'),
    ('int_ceil', ('7', '3'), '3'),
    ('int_ceil', ('9', '3'), '3'),
    ('int_floor', ('7', '3'), '2'),
    ('mod', ('-1', '5'), '4'),
    ('mod', ('7', '5'), '2'),
    ('ipow', ('3', '4'), '81'),
    ('left_shift', ('3', '2'), '12'),
    ('right_shift', ('-8', '1'), '-4'),
    ('logical_right_shift', ('static_cast<int32_t>(-8)', '1'), '2147483644'),
    ('logical_left_shift', ('3', '2'), '12'),
    ('bitwise_and', ('12', '10'), '8'),
    ('bitwise_or', ('12', '10'), '14'),
    ('bitwise_xor', ('12', '10'), '6'),
    ('bitwise_invert', ('12', ), '-13'),
    ('Max', ('1.0', '2.0'), '2.0'),
    ('Max', ('1.0', '5.0', '3.0'), '5.0'),
    ('Min', ('1.0', '5.0', '3.0'), '1.0'),
    ('exp2', ('3.0', ), '8.0'),
    ('sqrt', ('16.0', ), '4.0'),
    ('pow', ('2.0', '10.0'), '1024.0'),
    ('int_floor_ni', ('-7', '3'), '-3'),
    ('int_floor_ni', ('7', '3'), '2'),
    ('py_floor', ('-7', '3'), '-3'),
    ('py_floor', ('-7.0', '3.0'), '-3.0'),
    ('py_mod', ('-1', '5'), '4'),
    ('py_mod', ('-1.0', '5.0'), '4.0'),
    ('floor_mod', ('-1', '5'), '4'),
    ('cpp_mod', ('-1', '5'), '-1'),
    ('cpp_mod', ('-1.0', '5.0'), '-1.0'),
    ('Mod', ('-1', '5'), '-1'),
    ('Mod_float', ('-1.0', '5.0'), '-1.0'),
    ('Modulo', ('-17', '3'), '1'),
    ('Modulo_float', ('-17.0', '3.0'), '1.0'),
    ('np_float_pow', ('2', '10'), '1024.0'),
    ('sign_numpy_2', ('-2.5', ), '-1.0'),
    ('heaviside', ('2.0', ), '1.0'),
]

#: ``(name, statement template, expected)`` for the out-parameter helpers. These return ``void``
#: and write through references, so they cannot be probed as an expression.
STATEMENT_CASES = [
    ('cpp_divmod', 'long q = 0, r = 0; cpp_divmod(-7L, 3L, q, r); out[0] = static_cast<double>(q);', -2.0),
    ('cpp_divmod', 'long q = 0, r = 0; cpp_divmod(-7L, 3L, q, r); out[0] = static_cast<double>(r);', -1.0),
    ('py_divmod', 'long q = 0, r = 0; py_divmod(-7L, 3L, q, r); out[0] = static_cast<double>(q);', -3.0),
    ('py_divmod', 'long q = 0, r = 0; py_divmod(-7L, 3L, q, r); out[0] = static_cast<double>(r);', 2.0),
    ('np_modf', 'double i = 0, f = 0; np_modf(2.5, i, f); out[0] = i + f * 10;', 7.0),
    ('np_frexp', 'double m = 0; int e = 0; np_frexp(8.0, m, e); out[0] = m * 100 + e;', 54.0),
]


def spell(text, dialect):
    """``text`` written for ``dialect``: the C++ casts in the case tables become C casts.

    The cases are written once, in C++, because that is what the DaCe runtime headers document. Only
    the CAST spelling differs between the dialects, and MPR already has the rewrite that fixes it --
    the same one it applies to a library expansion's hand-written body.
    """
    return mpr_lowering.c_cast_native_code(text) if dialect is Dialect.STANDALONE_C else text


def render(name, arguments, dialect):
    """The MPR call expression for ``name`` under ``dialect``, plus any definition it needs."""
    lowered = mpr_lowering.lowering_for(name, arguments, dialect)
    if lowered is None:
        assert mpr_lowering.needs_definition(name, dialect), (
            f'{name!r} has no MPR lowering and no inline definition in {dialect}, so it would be emitted as a '
            'bare call to a DaCe runtime function that MPR does not declare')
        lowered = '%s(%s)' % (name, ', '.join(arguments))
    return lowered


def preamble(used, dialect):
    """The includes and definitions a unit calling ``used`` needs, exactly as ``mpr.preamble`` does."""
    lines = ['#include %s' % header for header in mpr_lowering.headers_for(used, dialect)]
    if dialect is Dialect.STANDALONE_C:
        lines.append(mpr_lowering.C_UNDEF_LINE)
    return '\n'.join(lines) + '\n\n' + '\n\n'.join(mpr_lowering.definitions_for(used, dialect))


def entry(dialect, signature='double * __restrict__ out'):
    """The probe's entry line. C++ needs the linkage specifier; C's ABI is already C's."""
    if dialect is Dialect.STANDALONE_C:
        return 'void probe(%s)' % signature
    return 'extern "C" void probe(%s)' % signature


def translation_unit(name, arguments, dialect):
    """A standalone translation unit whose ``probe`` writes the lowered call's value into ``out``."""
    body = render(name, [spell(argument, dialect) for argument in arguments], dialect)
    # Exactly how ``mpr.preamble`` decides: from the FINISHED text, not from the SDFG. The call
    # itself may expand to a macro (C) whose name is nowhere in ``arguments``.
    used = {name} | mpr_lowering.helpers_used(body, dialect)
    cast = '(double)' if dialect is Dialect.STANDALONE_C else 'static_cast<double>'
    return textwrap.dedent("""
        {preamble}

        {entry}
        {{
            out[0] = {cast}({body});
        }}
        """).format(preamble=preamble(used, dialect), entry=entry(dialect), cast=cast, body=body)


def run_probe(code, name, dialect):
    """Build ``code``, call its ``probe``, and return the ``double`` it wrote."""
    library = build_standalone(code, name=name, language=LANGUAGE[dialect])
    out = np.zeros(1, dtype=np.float64)
    function = library.probe
    function.argtypes = [ctypes.c_void_p]
    function.restype = None
    function(ctypes.c_void_p(out.ctypes.data))
    return out[0]


@pytest.mark.parametrize('dialect', DIALECTS, ids=DIALECT_IDS)
@pytest.mark.parametrize('name,arguments,expected',
                         CASES,
                         ids=['%s(%s)' % (name, ','.join(args)) for name, args, _ in CASES])
def test_lowering_computes_the_runtime_value(name, arguments, expected, dialect):
    """Each lowering builds bare and produces the value the DaCe runtime header would."""
    code = translation_unit(name, arguments, dialect)
    assert_standalone(code, label=name, language=LANGUAGE[dialect])
    value = run_probe(code, 'mpr_%s_%s' % (name, dialect.value), dialect)
    reference = float(eval(expected))  # noqa: S307 - a literal from CASES, not external input
    assert value == pytest.approx(
        reference, rel=1e-12,
        abs=1e-12), (f'{name}({", ".join(arguments)}) lowered to {render(name, arguments, dialect)!r} under '
                     f'{dialect} and gave {value!r}, but the runtime semantics are {reference!r}')


#: Every definition EACH dialect carries, so a C macro that only the C tables have is exercised
#: too. Parametrized over the dialect's own set rather than over the C++ one: the two differ (C
#: needs the maths dispatch macros, C++ gets those from ``<cmath>`` overload resolution), and a
#: shared list would have to skip the difference instead of covering it.
DEFINITION_CASES = [(dialect, name) for dialect in DIALECTS
                    for name in sorted(mpr_lowering.TABLES[dialect].inline_definitions)]


@pytest.mark.parametrize('dialect,name',
                         DEFINITION_CASES,
                         ids=['%s-%s' % (LANGUAGE[dialect], name) for dialect, name in DEFINITION_CASES])
def test_every_inline_definition_builds_clean(dialect, name):
    """Warnings are errors, and a definition is emitted into every unit that calls its function."""
    code = preamble({name}, dialect) + '\n'
    assert_standalone(code, label=name, language=LANGUAGE[dialect])
    diagnostics = compile_diagnostics(code, name='mpr_def_%s_%s' % (name, dialect.value), language=LANGUAGE[dialect])
    assert diagnostics == '', f'{name}: inline definition produced compiler warnings\n{diagnostics}'


@pytest.mark.parametrize('dialect', DIALECTS, ids=DIALECT_IDS)
def test_no_unqualified_runtime_function_is_unhandled(dialect):
    """Every name the harness can detect is handled in some lane, or explicitly refused.

    This is the anti-rot check. The harness rejects a bare call to any of these names, so one that
    the table does not cover has no way to reach valid MPR output -- it would be caught only at the
    point where a kernel using it failed to build.
    """
    unhandled = sorted(UNQUALIFIED_RUNTIME_FUNCTIONS - mpr_lowering.TABLES[dialect].known)
    assert not unhandled, (f'{unhandled} are rejected by the MPR harness but have no lowering, no inline '
                           f'definition, and no refusal in the {dialect.value} tables')


@pytest.mark.parametrize('dialect', DIALECTS, ids=DIALECT_IDS)
@pytest.mark.parametrize('name,statement,expected',
                         STATEMENT_CASES,
                         ids=['%s-%d' % (name, index) for index, (name, _, _) in enumerate(STATEMENT_CASES)])
def test_out_parameter_helpers_compute_the_runtime_value(name, statement, expected, dialect):
    """The ``void`` helpers write through their out-parameters and match the runtime.

    C has no references, so its macros take the same LVALUES the printers already pass and apply
    ``&`` themselves -- which is why one statement serves both dialects unchanged.
    """
    code = '%s\n\n%s\n{\n    %s\n}\n' % (preamble({name}, dialect), entry(dialect), spell(statement, dialect))
    assert_standalone(code, label=name, language=LANGUAGE[dialect])
    value = run_probe(code, 'mpr_stmt_%s_%s' % (name, dialect.value), dialect)
    assert value == pytest.approx(expected, rel=1e-12,
                                  abs=1e-12), (f'{name}: {statement!r} gave {value!r}, expected {expected!r}')


def test_c_definitions_are_emitted_callees_first():
    """A C macro must be ``#define``d before the function body that expands it is compiled."""
    emitted = mpr_lowering.definitions_for({'py_mod'}, Dialect.STANDALONE_C)
    order = [text.rsplit('#define ', 1)[1].split('(')[0] for text in emitted]
    assert order.index('int_floor_ni') < order.index('py_floor') < order.index('py_mod'), order
    assert 'mpr_floor' in order, 'py_floor divides through the floor macro, so it must be carried too'


def test_definitions_are_emitted_callees_first():
    """A helper is declared before the helper that calls it, or the unit does not compile."""
    emitted = mpr_lowering.definitions_for({'py_mod'}, Dialect.STANDALONE)
    order = [text.split('inline auto ')[1].split('(')[0] for text in emitted]
    assert order == ['int_floor_ni', 'py_floor', 'py_mod'], order


def test_dependency_closure_pulls_transitive_callees():
    """Asking for one helper brings everything it reaches."""
    assert mpr_lowering.required_definitions({'floor_mod'},
                                             Dialect.STANDALONE) == {'floor_mod', 'py_mod', 'py_floor', 'int_floor_ni'}


@pytest.mark.parametrize('dialect', DIALECTS, ids=DIALECT_IDS)
def test_rewrite_arity_mismatch_is_an_error(dialect):
    """A caller disagreeing with the table about a function's shape must not be papered over."""
    with pytest.raises(ValueError, match='expects 3 arguments'):
        mpr_lowering.lowering_for('ITE', ('a', 'b'), dialect)


@pytest.mark.parametrize('dialect', DIALECTS, ids=DIALECT_IDS)
def test_rewrites_use_each_argument_once(dialect):
    """A rewrite repeating an argument would duplicate whatever expression the caller printed."""
    for name, (arity, template) in mpr_lowering.TABLES[dialect].rewrites.items():
        for index in range(arity):
            occurrences = template.count('{%d}' % index)
            assert occurrences == 1, (f'REWRITES[{name!r}] uses {{{index}}} {occurrences} times; a repeated argument '
                                      'duplicates the printed expression, so it belongs in INLINE_DEFINITIONS')


#: ``name -> a C++ statement that constant-evaluates a call to it``. ``constexpr`` on a definition
#: is a claim, and an unchecked claim rots: a helper that silently stopped being usable in a
#: constant expression would still compile everywhere it is called at runtime. ``static_assert``
#: fails the BUILD if the call cannot be folded, so each probe proves the keyword earns its place.
#:
#: The out-parameter helpers are wrapped in a ``constexpr`` function, which is the only way to
#: constant-evaluate something that writes through references.
CONSTEXPR_PROBES = {
    'mpr_max':
    'static_assert(mpr_max(2.0, 1.0, 3.0) == 3.0);\nstatic_assert(mpr_max(0.0, -0.0) == 0.0);',
    'mpr_min':
    'static_assert(mpr_min(2.0, 1.0, 3.0) == 1.0);\nstatic_assert(mpr_min(2, 1.5) == 1.5);',
    'ifloor':
    'static_assert(ifloor(-3.5) == -4);\nstatic_assert(ifloor(static_cast<int64_t>(7)) == 7);',
    'int_ceil':
    'static_assert(int_ceil(7, 3) == 3);',
    'int_floor_ni':
    'static_assert(int_floor_ni(-7, 3) == -3);',
    'mod':
    'static_assert(mod(-1, 5) == 4);',
    'ipow':
    'static_assert(ipow(3, 4) == 81);',
    'logical_left_shift':
    'static_assert(logical_left_shift(3, 2) == 12);',
    'logical_right_shift':
    'static_assert(logical_right_shift(static_cast<int32_t>(-8), 1) == 2147483644);',
    'sign':
    'static_assert(sign(-2) == -1);',
    'sgn':
    'static_assert(sgn(7) == 1);',
    'sign_numpy_2':
    'static_assert(sign_numpy_2(-2) == -1);',
    'heaviside':
    'static_assert(heaviside(2) == 1);',
    'py_floor':
    'static_assert(py_floor(-7, 3) == -3);',
    'py_mod':
    'static_assert(py_mod(-1, 5) == 4);',
    'floor_mod':
    'static_assert(floor_mod(-1, 5) == 4);',
    'cpp_mod':
    'static_assert(cpp_mod(-1, 5) == -1);',
    'Mod_float':
    'static_assert(Mod_float(-1.0, 5.0) == -1.0);',
    'cpp_divmod':
    'constexpr long cpp_divmod_probe() { long q = 0, r = 0; cpp_divmod(-7L, 3L, q, r); return q; }\n'
    'static_assert(cpp_divmod_probe() == -2);',
    'py_divmod':
    'constexpr long py_divmod_probe() { long q = 0, r = 0; py_divmod(-7L, 3L, q, r); return q; }\n'
    'static_assert(py_divmod_probe() == -3);',
}

#: Definitions that cannot be ``constexpr``, and why. ``std::modf`` and ``std::frexp`` write through
#: a POINTER out-parameter and are not ``constexpr`` in C++20; ``Modulo`` divides through
#: ``std::floor`` on a ``double`` for every instantiation, so no argument makes it foldable.
#: A prefix scan cannot fold: it writes through an output iterator, and its OpenMP ``inscan``
#: clause has no meaning in a constant expression. They are ``static inline`` for that reason, and
#: this table is what states it rather than leaving the omission to look like an oversight.
_SCAN_REASON = 'writes through an output iterator under an OpenMP inscan clause, which cannot be constant-evaluated'

NOT_CONSTEXPR = {
    'scan_incl_sum': _SCAN_REASON,
    'scan_incl_product': _SCAN_REASON,
    'scan_incl_min': _SCAN_REASON,
    'scan_incl_max': _SCAN_REASON,
    'scan_excl_sum': _SCAN_REASON,
    'scan_excl_product': _SCAN_REASON,
    'scan_excl_min': _SCAN_REASON,
    'scan_excl_max': _SCAN_REASON,
    'min_identity': 'reads std::numeric_limits<T>::infinity(), whose constexpr-ness varies by type and library',
    'max_identity': 'reads std::numeric_limits<T>::infinity(), whose constexpr-ness varies by type and library',
    'find_first_index': 'runs an OpenMP-parallel cancelling search over a predicate, which has no constant evaluation',
    'find_first_chunk': 'reads the OpenMP thread count, which only exists at run time',
    'np_modf': 'std::modf takes a pointer out-parameter and is not constexpr before C++23',
    'np_frexp': 'std::frexp takes a pointer out-parameter and is not constexpr before C++23',
    'Modulo': 'divides through std::floor on a double for every instantiation (GCC folds it, clang does not)',
    'Modulo_float': 'divides through std::floor for every instantiation (GCC folds it, clang does not)',
}


@pytest.mark.parametrize('name', sorted(CONSTEXPR_PROBES))
def test_definition_is_usable_in_a_constant_expression(name):
    """The ``constexpr`` on each definition is real: the call folds at compile time."""
    code = '%s\n\n%s\n' % (preamble({name}, Dialect.STANDALONE), CONSTEXPR_PROBES[name])
    diagnostics = compile_diagnostics(code, name='mpr_ce_%s' % name)
    assert diagnostics == '', f'{name}: constexpr probe produced warnings\n{diagnostics}'


def test_every_definition_is_constexpr_or_says_why_not():
    """No definition escapes the choice: it is either probed as constexpr or listed as unable."""
    classified = set(CONSTEXPR_PROBES) | set(NOT_CONSTEXPR)
    unclassified = sorted(set(mpr_lowering.INLINE_DEFINITIONS) - classified)
    assert not unclassified, (f'{unclassified} are neither proven constexpr by a probe nor listed in NOT_CONSTEXPR; '
                              'an unchecked constexpr claim is how the keyword rots')


@pytest.mark.parametrize('name', sorted(NOT_CONSTEXPR))
def test_non_constexpr_definitions_are_not_marked_constexpr(name):
    """A definition that cannot fold must not claim ``constexpr``: that is ill-formed, no diagnostic."""
    assert 'static constexpr' not in mpr_lowering.INLINE_DEFINITIONS[name], (
        f'{name} is declared constexpr but {NOT_CONSTEXPR[name]}, so no argument permits constant evaluation. '
        'GCC folds std::floor as a builtin and accepts it; clang rejects the same code.')


# -- the C tables, held to the same anti-rot standard as the C++ ones ----------------------------

#: Names the C tables deliberately do NOT cover, with the reason each one is refused rather than
#: guessed. Kept as a test-side copy so a silent addition to ``C_UNSUPPORTED`` fails here: a
#: refusal is a capability gap, and it has to be a decision rather than an omission.
#: Nothing: every C++ helper is answered in C by a definition or by a rewrite.
EXPECTED_C_REFUSALS = set()


def test_c_refuses_exactly_the_names_it_says_it_does():
    """The C dialect's refusal list is what it claims to be, and each entry says why."""
    assert set(mpr_lowering.C_UNSUPPORTED) == EXPECTED_C_REFUSALS
    # The scan identities were once refused and are now rewritten; a refusal that reappears for
    # them means the rewrite was lost.
    assert not (mpr_lowering.C_REWRITTEN_IN_NATIVE_CODE & set(mpr_lowering.C_UNSUPPORTED))
    for name, reason in mpr_lowering.C_UNSUPPORTED.items():
        assert len(reason) > 20, f'{name} is refused without saying why'


def test_every_cpp_definition_has_a_c_form_or_a_refusal():
    """No C++ helper escapes the choice: it is either spelled in C or listed as unspellable."""
    unclassified = sorted(
        set(mpr_lowering.INLINE_DEFINITIONS) - set(mpr_lowering.C_INLINE_DEFINITIONS) -
        set(mpr_lowering.C_UNSUPPORTED) - mpr_lowering.C_REWRITTEN_IN_NATIVE_CODE)
    assert not unclassified, (f'{unclassified} have a C++ inline definition but no C form, no rewrite and no entry '
                              'in C_UNSUPPORTED, so a kernel calling one would render C that does not build')


def test_every_cpp_std_rename_has_a_c_form():
    """Every ``std::`` rename is answered in C by a rename or a definition of MPR's own.

    ``std::gcd`` / ``std::lcm`` are the two with no C library counterpart at all, so they move lanes:
    a rename in C++, an emitted definition in C. Asserted explicitly, because "moved lanes" and
    "was forgotten" look identical from the C++ side.
    """
    answered = set(mpr_lowering.C_STD_RENAMES) | set(mpr_lowering.C_INLINE_DEFINITIONS)
    missing = sorted(set(mpr_lowering.STD_RENAMES) - answered)
    assert not missing, f'{missing} are renamed to std:: in C++ but have no C spelling'
    for name in ('gcd', 'lcm'):
        assert name not in mpr_lowering.C_STD_RENAMES, f'C has no library {name}'
        assert name in mpr_lowering.C_INLINE_DEFINITIONS, f'{name} must be emitted by MPR in C'


def test_every_c_definition_is_reachable():
    """No dead C macro: each one is named by a rename, a rewrite, a min/max, or another definition.

    A table that grows an entry nothing reaches is a table that has stopped being checked -- the
    unreachable entry never compiles, never runs, and never fails.
    """
    reachable = set(mpr_lowering.C_STD_RENAMES.values()) | set(mpr_lowering.C_VARIADIC_MINMAX.values())
    # A helper the printers call UNCHANGED -- ``lowering_for`` returns None and ``needs_definition``
    # says MPR emits the body. That is any C definition whose name is a runtime function, which
    # includes gcd/lcm: a std:: rename in C++, an emitted definition here.
    reachable |= {name for name in mpr_lowering.C_INLINE_DEFINITIONS if name in mpr_lowering.KNOWN}
    for dependencies in mpr_lowering.C_DEFINITION_DEPENDENCIES.values():
        reachable |= set(dependencies)
    for _, template in mpr_lowering.C_REWRITES.values():
        reachable |= mpr_lowering.helpers_used(template.replace('{0}', 'a').replace('{1}', 'b'), Dialect.STANDALONE_C)
    reachable |= mpr_lowering.helpers_used(mpr_lowering.rewrite_native_code(FIND_FIRST_STATEMENT, Dialect.STANDALONE_C),
                                           Dialect.STANDALONE_C)
    # The two arity-specific halves of ``heaviside`` are defined inside its own entry, not called
    # from anywhere else; every other name must be reached from outside.
    unreachable = sorted(set(mpr_lowering.C_INLINE_DEFINITIONS) - reachable)
    assert not unreachable, f'{unreachable} are emitted by no lowering, so nothing ever exercises them'


def test_every_c_definitions_dependencies_are_real():
    """A recorded dependency must be a definition, or the topological order is over a phantom."""
    for name, dependencies in mpr_lowering.C_DEFINITION_DEPENDENCIES.items():
        assert name in mpr_lowering.C_INLINE_DEFINITIONS, f'{name} has dependencies but no definition'
        for dependency in dependencies:
            assert dependency in mpr_lowering.C_INLINE_DEFINITIONS, f'{name} depends on the undefined {dependency}'
            assert dependency in mpr_lowering.C_INLINE_DEFINITIONS[name] or dependency.startswith('mpr_'), (
                f'{name} does not mention {dependency}')


@pytest.mark.parametrize('name', ['min_identity', 'max_identity'])
@pytest.mark.parametrize('ctype', sorted(mpr_lowering.C_SCAN_IDENTITIES))
def test_c_rewrites_the_scan_identities_to_constants(name, ctype):
    """The identity has no C function template, so C spells it as the constant for that type.

    Both dialects are asserted: C++ keeps the templated call it already emits, so a rewrite that
    leaked into C++ would show up here rather than as a numeric difference in a min/max scan.
    """
    call = '::dace::scan::detail::%s<%s>()' % (name, ctype)
    expected = mpr_lowering.C_SCAN_IDENTITIES[ctype][0 if name.startswith('min') else 1]
    assert mpr_lowering.rewrite_native_code(call, Dialect.STANDALONE_C) == expected
    assert mpr_lowering.rewrite_native_code(call, Dialect.STANDALONE).startswith(name)


#: The statement ``ExpandFindFirstPure`` / ``ExpandFindFirstOpenMP`` write, copied here rather than
#: imported so that a change to the expansion's spelling breaks this file instead of silently
#: turning the C rewrite into a no-op -- which would surface only as an unlowered ``dace::`` name in
#: some kernel that happens to search.
FIND_FIRST_STATEMENT = ('_out_idx = dace::find_first_index((0), (N), '
                        '[&](long long __i) -> bool { return (_a[__i] > 0.5); }, false);')


def test_c_rewrites_the_find_first_call_into_the_statement_macro():
    """C keeps the predicate, the index name and the bounds, and moves the target into the macro.

    The predicate cannot survive as an argument to anything callable in C, so the check is that it
    arrives verbatim and still reads the index under the name the expansion wrote its subscripts
    against: a rewrite that renamed either would build and then search the wrong elements.
    """
    rewritten = mpr_lowering.rewrite_native_code(FIND_FIRST_STATEMENT, Dialect.STANDALONE_C)
    assert rewritten == 'mpr_find_first(_out_idx, (0), (N), __i, false, (_a[__i] > 0.5));'
    assert mpr_lowering.helpers_used(rewritten, Dialect.STANDALONE_C) == {'mpr_find_first'}
    # C++ has the lambda, so it keeps the call and only drops the namespace.
    assert mpr_lowering.rewrite_native_code(FIND_FIRST_STATEMENT,
                                            Dialect.STANDALONE) == FIND_FIRST_STATEMENT.replace('dace::', '')


def test_c_leaves_an_unrecognized_find_first_call_for_verify():
    """A call shape the rewrite does not know stays ``dace::``-qualified rather than half-rewritten.

    The rewrite is textual and matches one statement form. If the expansion ever writes another,
    the honest outcome is MPR refusing to render -- a partial rewrite would produce C that does not
    compile, with nothing naming the construct responsible.
    """
    unknown = '_out_idx = dace::find_first_index(0, N, some_functor, false);'
    assert mpr_lowering.rewrite_native_code(unknown, Dialect.STANDALONE_C) == unknown


def test_c_refuses_a_scan_identity_it_cannot_order():
    """Complex has no ordered extreme, so a min/max scan over it must raise, not pick a wrong seed."""
    with pytest.raises(NotImplementedError, match='no ordered extreme'):
        mpr_lowering.rewrite_native_code('min_identity<double _Complex>()', Dialect.STANDALONE_C)


def test_variadic_minmax_nests_binary_calls_in_c():
    """C has no variadic macro to fold over, so a three-way ``Max`` nests the binary one; the C++
    template takes all three directly, and both associate left to right."""
    arguments = ('a', 'b', 'c')
    assert mpr_lowering.variadic_minmax('Max', arguments, Dialect.STANDALONE_C) == 'mpr_max(mpr_max(a, b), c)'
    assert mpr_lowering.variadic_minmax('Max', arguments, Dialect.STANDALONE) == 'mpr_max(a, b, c)'


def test_c_scan_helpers_keep_the_parallel_inscan_form():
    """The scan is the reason the eight helpers exist; a serial loop would not be a parallel scan."""
    for kind, clause in (('incl', 'inclusive'), ('excl', 'exclusive')):
        for operation, reduction in (('sum', '+'), ('product', '*'), ('min', 'min'), ('max', 'max')):
            definition = mpr_lowering.C_INLINE_DEFINITIONS['scan_%s_%s' % (kind, operation)]
            assert '#pragma omp simd reduction(inscan, %s:acc)' % reduction in definition
            assert '#pragma omp scan %s(acc)' % clause in definition


#: A dispatch macro whose argument has a SIDE EFFECT. ``_Generic``'s controlling expression is
#: unevaluated, so the argument must be evaluated exactly once -- but the argument is written twice
#: in the macro's expansion, and nothing but the standard says the first one does not run.
_SINGLE_EVALUATION_PROBE = """
#include <stdint.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <complex.h>
#undef I
{definitions}
static int calls = 0;
static double bump(void) {{ calls += 1; return 16.0; }}
void probe(double * out) {{ calls = 0; out[0] = {call}; out[0] += calls; }}
"""


@pytest.mark.parametrize('name,call,expected', [
    ('mpr_sqrt', 'mpr_sqrt(bump())', 5.0),
    ('mpr_max', 'mpr_max(bump(), 1.0)', 17.0),
    ('mpr_min', 'mpr_min(bump(), 1.0)', 2.0),
],
                         ids=['sqrt', 'max', 'min'])
def test_c_dispatch_macros_evaluate_each_argument_once(name, call, expected):
    """One evaluation, proven by counting -- a duplicated argument would double a stateful call."""
    definitions = '\n'.join(mpr_lowering.definitions_for({name}, Dialect.STANDALONE_C))
    code = _SINGLE_EVALUATION_PROBE.format(definitions=definitions, call=call)
    library = ctypes.CDLL(compile_standalone(code, 'mpr_once_%s' % name, language='c'))
    out = np.zeros(1, dtype=np.float64)
    library.probe.argtypes = [ctypes.c_void_p]
    library.probe.restype = None
    library.probe(ctypes.c_void_p(out.ctypes.data))
    assert out[0] == pytest.approx(expected), (f'{call} gave {out[0]!r}; the trailing digit is the number of times '
                                               'the argument ran, and it must be 1')
