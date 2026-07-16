# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Reproducer: ``compiler.cpu.codegen_params.index_ctype`` picks the integer type the readable CPU
    code generator's ``<array>_idx`` / ``<array>_size`` helpers compute a flat index in -- ``int64``
    (``long long``, the default, byte-identical to the historical signature) vs ``int32`` (``int``).

    int32 is OPT-IN ONLY and is never selected automatically. It wraps past 2**31, and the bound is on
    the ELEMENT COUNT, not the byte size -- so an int8 array overflows at just 2 GiB of data, which is
    reachable, silent, and unprovable by the generator for a symbolic shape. This script prints that
    bound per dtype alongside the emitted signatures.

    The knob exists to be swept, not because int32 wins: measured on x86-64 (Zen 4) most kernels sit
    inside a 0.5-6% noise floor, and the deltas that clear it disagree on sign. Templating the helpers
    to let each call site pick was measured and rejected -- GCC's identical-code folding collapses the
    instantiations onto one body, and C++ deduction would pick 32-bit math off the generated
    ``for (auto i = 0; ...)`` call sites with no diagnostic. See CODEGEN_STYLE_PERFORMANCE.md section 5.

    The legacy generator inlines its offset arithmetic instead of emitting helpers, so it has no index
    type to pick, ignores the key, and stays byte-identical.

    Run:  python samples/codegen/index_ctype_demo.py
"""
import re

import dace
from dace.config import set_temporary

N = dace.symbol('N')
M = dace.symbol('M')


@dace.program
def transpose_add(A: dace.float64[N, M], B: dace.float64[M, N]):
    """ Transposed and non-square, so the two arrays need genuinely different index helpers. """
    for i, j in dace.map[0:N, 0:M]:
        B[j, i] = A[i, j] + 1.0


def emit(implementation, index_ctype):
    sdfg = transpose_add.to_sdfg(simplify=True)
    with set_temporary('compiler', 'cpu', 'implementation', value=implementation), \
         set_temporary('compiler', 'cpu', 'codegen_params', 'index_ctype', value=index_ctype):
        return '\n'.join((obj.clean_code or obj.code) for obj in sdfg.generate_code() if obj.language == 'cpp')


def index_helpers(code):
    """ The generated ``<array>_idx`` helper definition lines. """
    return [line.strip() for line in code.splitlines() if re.search(r'\w+_idx\(.*\)\s*\{', line)]


if __name__ == '__main__':
    print('compiler.cpu.codegen_params.index_ctype -- emitted index helpers\n')
    for implementation in ('experimental_readable', 'legacy'):
        for index_ctype in ('int64', 'int32'):
            helpers = index_helpers(emit(implementation, index_ctype))
            print(f'  implementation={implementation:22s} index_ctype={index_ctype}')
            for helper in helpers or ['<none emitted -- offset math is inlined at each access>']:
                print(f'    {helper}')
            print()

    print('Legacy is IDENTICAL for both values: it inlines its offset arithmetic, so it has no index\n'
          'type to pick and its output stays byte-identical.\n')

    print('Why int32 is opt-in: it wraps at 2**31 ELEMENTS -- a bound on the element count, not the\n'
          'byte size, so a narrow dtype hits it at a size you can actually allocate:\n')
    for dtype in (dace.int8, dace.float32, dace.float64):
        gib = (2**31) * dtype.bytes / 2**30
        print(f'    {dtype.to_string():9s} ({dtype.bytes} B/element)  overflows at {gib:5.0f} GiB')
