# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The CPF-rendered prefix scan (``cpf_lowering.blocked_scan_body``) over several tiles: every op, inclusive and
exclusive, built with the system C compiler and run on four threads against a Python reference."""

import itertools
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from dace.cpf_lowering import blocked_scan_body

#: A thread's block holds at most 512 KB (65536 int64), so four threads see a tile of 262144 elements; this many
#: elements take three full tiles and a ragged fourth.
N = 3 * 65536 * 4 + 17

IDENTITY = {"sum": 0, "product": 1, "min": 10**9, "max": -(10**9)}
FOLD = {"sum": lambda a, x: a + x, "product": lambda a, x: a * x, "min": min, "max": max}


def reference(values: list[int], kind: str, operation: str, seed: int) -> list[int]:
    out, acc = [], seed
    for x in values:
        if kind == "incl":
            acc = FOLD[operation](acc, x)
            out.append(acc)
        else:
            out.append(acc)
            acc = FOLD[operation](acc, x)
    return out


@pytest.mark.skipif(shutil.which("gcc") is None, reason="needs gcc")
@pytest.mark.parametrize("kind, operation", list(itertools.product(["incl", "excl"], ["sum", "product", "min", "max"])))
def test_rendered_scan_over_several_tiles(kind: str, operation: str, tmp_path: Path):
    body = "\n".join(blocked_scan_body(kind, operation, "int64_t", c_dialect=True))
    source = (
        "#include <stdint.h>\n#include <stdio.h>\n#include <stdlib.h>\n#include <omp.h>\n"
        f"static void scan(const int64_t *f, int64_t *o, long lo, long hi, int64_t seed) {{\n{body}\n}}\n"
        "int main(void) {\n"
        f"    long n = {N};\n"
        "    int64_t *f = malloc(sizeof(int64_t) * n), *o = malloc(sizeof(int64_t) * n);\n"
        '    for (long i = 0; i < n; ++i) if (scanf("%ld", &f[i]) != 1) return 2;\n'
        f"    scan(f, o, 0, n, {IDENTITY[operation]});\n"
        '    for (long i = 0; i < n; ++i) printf("%ld\\n", o[i]);\n'
        "    return 0;\n}\n"
    )
    (tmp_path / "scan.c").write_text(source)
    subprocess.run(["gcc", "-O2", "-fopenmp", "-o", str(tmp_path / "scan"), str(tmp_path / "scan.c")], check=True)
    rng = np.random.default_rng(5)
    values = (rng.choice([-1, 1], N) if operation == "product" else rng.integers(-1000, 1000, N)).tolist()
    run = subprocess.run(
        [str(tmp_path / "scan")],
        input="\n".join(map(str, values)),
        capture_output=True,
        text=True,
        check=True,
        env={"OMP_NUM_THREADS": "4"},
    )
    assert [int(line) for line in run.stdout.split()] == reference(values, kind, operation, IDENTITY[operation])


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
