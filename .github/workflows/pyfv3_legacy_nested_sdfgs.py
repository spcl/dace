# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Pytest plugin converting gt4py's origin/domain wrapper to the nested SDFG contract.

``gt4py.cartesian.backend.dace_backend.freeze_origin_domain_sdfg`` wraps a stencil SDFG in a nested
SDFG whose connectors describe the window the origin and domain select out of the caller's fields,
with the memlets inside written relative to that window -- the DaCe 1.x nested SDFG semantics. It
then calls ``inline_sdfgs`` on the wrapper. Under the current contract a connector is the container
it is connected to, so DaCe refuses to inline such a nested SDFG (it would drop the window and
produce wrong accesses).

This plugin patches ``inline_sdfgs`` as gt4py imported it so that
``dace.sdfg.dealias.convert_legacy_nested_sdfgs`` restates the wrapper just before it is inlined,
which lets the pyFV3 regression tests run against an unmodified gt4py. It is meant to be removed once
gt4py builds contract-conforming SDFGs; gt4py's own TODO in that function points at
https://github.com/GridTools/gt4py/issues/2082.

Usage: put this directory on ``PYTHONPATH`` and pass ``-p pyfv3_legacy_nested_sdfgs`` to pytest. The
patch happens at import, so no pytest hooks are needed.
"""
# ndsl refuses to be imported once gt4py's configuration has been read, so it goes first
import ndsl.dsl  # noqa: F401
import gt4py.cartesian.backend.dace_backend as dace_backend

from dace.sdfg import dealias

_MARKER = '_dace_converts_legacy_nested_sdfgs'


def _inline_sdfgs_with_conversion(sdfg, *args, **kwargs):
    """Converts the legacy nested SDFGs of ``sdfg`` before handing it to gt4py's ``inline_sdfgs``."""
    converted = dealias.convert_legacy_nested_sdfgs(sdfg)
    if converted:
        print(f'[pyfv3_legacy_nested_sdfgs] converted {len(converted)} legacy connector(s) in {sdfg.name}')
    return _original_inline_sdfgs(sdfg, *args, **kwargs)


_original_inline_sdfgs = dace_backend.inline_sdfgs
if not getattr(_original_inline_sdfgs, _MARKER, False):
    setattr(_inline_sdfgs_with_conversion, _MARKER, True)
    dace_backend.inline_sdfgs = _inline_sdfgs_with_conversion
