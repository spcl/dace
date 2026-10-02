# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``dace::math::pow`` on two integral operands and the global ``ipow`` in ``dace/math.h``, compiled with the
configured host compiler under UBSan and compared with NumPy.

``dace::math::pow`` is ``std::pow``, so two integral operands give a double, including a negative exponent's
reciprocal. ``ipow`` keeps the base type and is only ever called with an exponent proven non-negative.
"""
import itertools
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dace
from dace.config import Config

INCLUDE = Path(dace.__file__).parent / 'runtime' / 'include'

INTEGRAL_BASES = (-3, -2, -1, 0, 1, 2, 3)
FLOATING_BASES = (-2.0, -0.5, 0.5, 2.0, 3.0)
NEGATIVE_EXPONENTS = (-5, -3, -2, -1)
NONNEGATIVE_EXPONENTS = (0, 1, 2, 5)
INTEGRAL_TYPES = ('int', 'long long')


def build_cases():
    """``id -> (C++ expression, kind, expected)``; kind is ``i`` for an integer result, ``f`` for a double."""
    cases = {}
    for ctype, exp_ctype in itertools.product(INTEGRAL_TYPES, INTEGRAL_TYPES):
        for base, exponent in itertools.product(INTEGRAL_BASES, NEGATIVE_EXPONENTS + NONNEGATIVE_EXPONENTS):
            name = f'pow_{ctype}_{exp_ctype}_{base}_{exponent}'.replace(' ', '_').replace('-', 'm')
            with np.errstate(divide='ignore'):
                expected = float(np.float_power(base, exponent))
            cases[name] = (f'dace::math::pow(({ctype}){base}, ({exp_ctype}){exponent})', 'f', expected)
        for base, exponent in itertools.product(INTEGRAL_BASES, NONNEGATIVE_EXPONENTS):
            name = f'ipow_{ctype}_{exp_ctype}_{base}_{exponent}'.replace(' ', '_').replace('-', 'm')
            cases[name] = (f'ipow(({ctype}){base}, ({exp_ctype}){exponent})', 'i',
                           int(np.power(np.int64(base), exponent)))
    for ctype, base, exponent in itertools.product(('double', 'float'), FLOATING_BASES, NONNEGATIVE_EXPONENTS):
        name = f'ipow_{ctype}_{base}_{exponent}'.replace('.', 'p').replace('-', 'm')
        cases[name] = (f'ipow(({ctype}){base!r}, (int){exponent})', 'f', float(np.float_power(base, exponent)))
    cases['ipow_double_zero_zero'] = ('ipow(0.0, 0)', 'f', 1.0)
    cases['ipow_unsigned_exponent'] = ('ipow(2.0, 5u)', 'f', 32.0)
    cases['ipow_unsigned_exponent_integral_base'] = ('ipow(3, 4u)', 'i', 81)
    for exponent in NONNEGATIVE_EXPONENTS:
        cases[f'ipow_complex_{exponent}'] = (f'ipow(std::complex<double>(0.0, 1.0), {exponent})', 'c',
                                             complex(np.power(1j * 1.0, exponent)))
    return cases


CASES = build_cases()
EMIT = {
    'i': 'std::printf("%s %lld\\n", "{name}", (long long)({expression}));',
    'f': 'std::printf("%s %.17g\\n", "{name}", (double)({expression}));',
    'c': 'std::printf("%s %.17g %.17g\\n", "{name}", ({expression}).real(), ({expression}).imag());',
}


def run_driver(directory: Path) -> dict:
    """Compiles all cases into one program under UBSan, which aborts on the first undefined behaviour."""
    lines = ['#include <cstdio>', '#include <complex>', '#include <dace/dace.h>', 'int main() {']
    lines += [EMIT[kind].format(name=name, expression=expression) for name, (expression, kind, _) in CASES.items()]
    lines += ['return 0;', '}']
    (directory / 'driver.cpp').write_text('\n'.join(lines))
    compiler = Config.get('compiler', 'cpu', 'executable') or os.environ.get('CXX') or 'c++'
    command = [
        compiler, f'-std=c++{Config.get("compiler", "cpp_standard")}', '-fopenmp', '-fsanitize=undefined',
        '-fno-sanitize-recover=all', '-O1', f'-I{INCLUDE}',
        str(directory / 'driver.cpp'), '-o',
        str(directory / 'driver')
    ]
    build = subprocess.run(command, capture_output=True, text=True, timeout=600)
    assert build.returncode == 0, build.stderr
    run = subprocess.run([str(directory / 'driver')], capture_output=True, text=True, timeout=600)
    assert run.returncode == 0, run.stderr
    return {line.split()[0]: line.split()[1:] for line in run.stdout.splitlines()}


@pytest.fixture(scope='module')
def results(tmp_path_factory):
    return run_driver(tmp_path_factory.mktemp('integer_power'))


@pytest.mark.parametrize('name', CASES)
def test_the_power_matches_numpy(name, results):
    _, kind, expected = CASES[name]
    got = results[name]
    if kind == 'i':
        assert int(got[0]) == expected
    elif kind == 'f':
        np.testing.assert_allclose(float(got[0]), expected, rtol=1e-6)
    else:
        np.testing.assert_allclose(complex(float(got[0]), float(got[1])), expected, atol=1e-12)


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        all_results = run_driver(Path(directory))
    for case in CASES:
        test_the_power_matches_numpy(case, all_results)
