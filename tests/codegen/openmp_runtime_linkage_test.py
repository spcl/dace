# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A library that calls into OpenMP names the runtime among its own NEEDED entries, whatever configured its
build before.

CMake's OpenMP target links the runtime only through the libraries FindOpenMP detects, and the detection skips a
library the compiler already links implicitly. A configure with ``-fopenmp`` in the C++ flags therefore caches no
OpenMP library at all, and a later build under other flags that reuses it -- the same build folder, or the
configure and command caches that folder publishes -- linked the runtime nowhere: the library failed to load on an
undefined ``GOMP_parallel``.
"""
import pathlib
import subprocess
from collections.abc import Iterator

import numpy as np
import pytest

import dace
from dace.codegen import compiler

N = dace.symbol('N')

#: Entry points GCC (``GOMP_``) and Clang (``__kmpc_``) emit for a parallel map, beside the ``omp_`` API.
OPENMP_PREFIXES = ('omp_', 'GOMP_', '__kmpc_')


@dace.program
def doubled(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[0:N]:
        y[i] = x[i] * 2.0


@pytest.fixture
def private_cache(tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setattr(compiler, 'build_cache_root', lambda: str(tmp_path / 'cache'))
    with dace.config.set_temporary('compiler', 'precompiled_header', value=False):
        with dace.config.set_temporary('cache', value='name'):
            with dace.config.set_temporary('default_build_folder', value=str(tmp_path / 'build')):
                yield


def default_args() -> str:
    return dace.Config.get('compiler', 'cpu', 'args')


def build(name: str, cpu_args: str) -> dace.SDFG:
    sdfg = doubled.to_sdfg()
    sdfg.name = name
    with dace.config.set_temporary('compiler', 'cpu', 'args', value=cpu_args):
        sdfg.compile(return_program_handle=False)
    return sdfg


def dynamic_symbols(library: pathlib.Path, selection: str) -> set[str]:
    listing = subprocess.run(['nm', '-D', selection, str(library)], capture_output=True, text=True, check=True)
    return {line.split()[-1].split('@')[0] for line in listing.stdout.splitlines() if line.strip()}


def assert_carries_its_openmp_runtime(sdfg: dace.SDFG) -> None:
    library = compiler.get_binary_name(sdfg.build_folder, sdfg.name)
    called = {name for name in dynamic_symbols(library, '--undefined-only') if name.startswith(OPENMP_PREFIXES)}
    assert called, f'{library.name} calls no OpenMP entry point, so nothing is tested'

    dynamic = subprocess.run(['readelf', '-d', str(library)], capture_output=True, text=True, check=True).stdout
    needed = [line.split('[')[1].rstrip(']') for line in dynamic.splitlines() if '(NEEDED)' in line]
    resolved = subprocess.run(['ldd', str(library)], capture_output=True, text=True, check=True).stdout
    providers = [
        pathlib.Path(fields[2]) for fields in map(str.split, resolved.splitlines())
        if len(fields) > 2 and fields[0] in needed and fields[1] == '=>'
    ]
    provided = set().union(*(dynamic_symbols(provider, '--defined-only') for provider in providers))
    assert called <= provided, f'{library.name} calls {sorted(called - provided)}, defined by no NEEDED entry of {needed}'

    x, y = np.arange(16, dtype=np.float64), np.zeros(16)
    compiler.load_precompiled_sdfg(sdfg.build_folder, sdfg)(x=x, y=y, N=16)
    assert np.array_equal(y, x * 2.0)


def test_a_program_rebuilt_without_openmp_in_its_flags_links_the_runtime(private_cache: None):
    build('reconfigured', f'{default_args()} -fopenmp')

    sut = build('reconfigured', default_args())

    assert_carries_its_openmp_runtime(sut)


def test_a_program_built_from_the_caches_of_such_a_rebuild_links_the_runtime(private_cache: None):
    build('publisher', f'{default_args()} -fopenmp')
    build('publisher', default_args())

    sut = build('consumer', default_args())

    assert_carries_its_openmp_runtime(sut)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
