# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
``dace::math::pow`` and ``dace::math::ipow`` in ``dace/math.h``, compiled with the
configured host compiler under UBSan and compared with NumPy.

``dace::math::pow`` with an integral exponent multiplies, and takes the reciprocal for a negative one; an
integral base with a signed exponent gives a double. ``dace::math::ipow`` takes an unsigned exponent and keeps the
base type.
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

    def key(*parts):
        return '_'.join(str(p) for p in parts).replace(' ', '_').replace('.', 'p').replace('-', 'm')

    for ctype, exp_ctype in itertools.product(INTEGRAL_TYPES, INTEGRAL_TYPES):
        for base, exponent in itertools.product(INTEGRAL_BASES, NEGATIVE_EXPONENTS + NONNEGATIVE_EXPONENTS):
            with np.errstate(divide='ignore'):
                expected = float(np.float_power(base, exponent))
            cases[key('pow', ctype, exp_ctype, base,
                      exponent)] = (f'dace::math::pow(({ctype}){base}, ({exp_ctype}){exponent})', 'f', expected)
    for ctype, base, exponent in itertools.product(INTEGRAL_TYPES, INTEGRAL_BASES, NONNEGATIVE_EXPONENTS):
        expected = int(np.power(np.int64(base), exponent))
        cases[key('pow_unsigned', ctype, base,
                  exponent)] = (f'dace::math::pow(({ctype}){base}, {exponent}u)', 'i', expected)
        cases[key('ipow', ctype, base, exponent)] = (f'dace::math::ipow(({ctype}){base}, {exponent}u)', 'i', expected)
    for ctype, base in itertools.product(('double', 'float'), FLOATING_BASES):
        for exponent in NEGATIVE_EXPONENTS + NONNEGATIVE_EXPONENTS:
            cases[key('pow', ctype, base, exponent)] = (f'dace::math::pow(({ctype}){base!r}, (int){exponent})', 'f',
                                                        float(np.float_power(base, exponent)))
        for exponent in NONNEGATIVE_EXPONENTS:
            cases[key('ipow', ctype, base, exponent)] = (f'dace::math::ipow(({ctype}){base!r}, {exponent}u)', 'f',
                                                         float(np.float_power(base, exponent)))
        cases[key('pow_fractional', ctype, base)] = (f'dace::math::pow(({ctype}){base!r}, 0.5)', 'f',
                                                     float(np.sqrt(abs(base))) if base > 0 else float('nan'))
    for exponent in NEGATIVE_EXPONENTS + NONNEGATIVE_EXPONENTS:
        cases[key('pow_complex', exponent)] = (f'dace::math::pow(std::complex<double>(0.0, 1.0), {exponent})', 'c',
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
        np.testing.assert_allclose(float(got[0]), expected, rtol=1e-6, equal_nan=True)
    else:
        np.testing.assert_allclose(complex(float(got[0]), float(got[1])), expected, atol=1e-12)


if __name__ == '__main__':
    import tempfile
    with tempfile.TemporaryDirectory() as directory:
        all_results = run_driver(Path(directory))
    for case in CASES:
        test_the_power_matches_numpy(case, all_results)
