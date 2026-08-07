# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``std::numeric_limits`` must be specialized for ``dace::half`` and ``dace::bfloat16``. Unspecialized
class types get the primary template, whose ``max()``/``lowest()``/``infinity()`` are all ``T()`` --
zero -- so identity-seeded min/max reductions silently produce zeros instead of failing to build."""
import os
import shutil
import subprocess

import pytest

import dace
from dace.config import Config

INCLUDE = os.path.join(os.path.dirname(os.path.abspath(dace.__file__)), 'runtime', 'include')

#: Every check would pass trivially against the primary template if it compared to zero, so each one
#: is a value the primary template cannot produce.
PROBE_SOURCE = r'''
#include "dace/types.h"
template <typename T>
static bool ok() {
    using L = std::numeric_limits<T>;
    return L::is_specialized && (float)L::max() > 0.0f && (float)L::lowest() < 0.0f &&
           (float)L::infinity() > (float)L::max() && (float)L::denorm_min() > 0.0f;
}
int main() {
    if (!ok<dace::half>() || !ok<dace::bfloat16>()) return 1;
    // The exact finite bounds of IEEE binary16 and bfloat16.
    if ((float)std::numeric_limits<dace::half>::max() != 65504.0f) return 2;
    if ((float)std::numeric_limits<dace::bfloat16>::lowest() != -3.38953139e+38f) return 3;
    return 0;
}
'''


def test_numeric_limits_is_specialized(tmp_path):
    executable = Config.get('compiler', 'cpu', 'executable') or 'c++'
    assert shutil.which(executable), f'configured compiler {executable!r} is not on PATH'
    source, binary = tmp_path / 'probe.cpp', tmp_path / 'probe'
    source.write_text(PROBE_SOURCE)
    build = subprocess.run([
        executable, f'-std=c++{Config.get("compiler", "cpp_standard")}', '-I', INCLUDE,
        str(source), '-o',
        str(binary)
    ],
                           capture_output=True,
                           text=True,
                           timeout=300)
    assert build.returncode == 0, f'probe did not compile:\n{build.stderr}'
    assert subprocess.run([str(binary)], timeout=60).returncode == 0, \
        'std::numeric_limits is unspecialized for a low-precision type, so its identities are zero'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
