# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Which dtype casts the tile-access classifier collapses before it reads a subset's affine
structure.

A cast is a no-op for index arithmetic but a :class:`sympy.Function` node in the parse, so
``int64(i) + 3`` only classifies as LINEAR once the cast is stripped. The stripper used to work off
a hand-written list of dtype names, and a name missing from that list failed silently in the "did
nothing" direction: the Function survived, the dim read as opaque, and the access dropped to a
weaker kind with no diagnostic. The list is now the dtype registry, longest name first.

Order matters for the prefix-strip regex the same names build: ``float`` must not get the chance to
match inside ``float64``. Both halves are pinned -- the ordering as a structural property, and the
``float`` / ``float64`` pair by behavior.
"""
import pytest

from dace import dtypes, symbolic
from dace.transformation.passes.vectorization.utils import tile_access

#: The dtype classes the hand-written list never held, by name. Hard-coded from the diff, not
#: derived: two low-precision floats, both fp8 encodings, all three complex spellings, and the two
#: bare Python-level names the frontend also emits.
DROPPED_BY_THE_HAND_WRITTEN_LIST = ("bfloat16", "float8_e4m3fn", "float8_e5m2", "complex", "complex64", "complex128",
                                    "int", "float")


@pytest.mark.parametrize("name", sorted(dtypes.TYPECLASS_STRINGS))
def test_a_cast_in_any_registered_dtype_collapses_to_its_argument(name):
    """Every name the dtype registry knows is recognised as a cast and stripped."""
    assert tile_access._sympify_tasklet_rhs(f"dace.{name}(i) + 3") == symbolic.pystr_to_symbolic("i + 3")


@pytest.mark.parametrize("name", DROPPED_BY_THE_HAND_WRITTEN_LIST)
@pytest.mark.parametrize("prefix", ["", "np.", "numpy."])
def test_a_dropped_dtype_collapses_under_every_spelling_the_frontend_emits(prefix, name):
    """The classes the old list missed collapse bare and under either numpy prefix too, so the
    fix covers the Function-name test and the prefix-strip regex alike."""
    assert tile_access._sympify_tasklet_rhs(f"{prefix}{name}(i) + 3") == symbolic.pystr_to_symbolic("i + 3")


def test_a_shorter_dtype_name_never_precedes_a_longer_one_it_prefixes():
    """Longest-first ordering: no name in the alternation can shadow one that starts with it."""
    names = tile_access._CAST_NAMES_LONGEST_FIRST
    shadowed = [(short, long) for i, short in enumerate(names) for long in names[i + 1:] if long.startswith(short)]
    assert shadowed == []


def test_the_bare_float_alternative_does_not_shadow_float64():
    """The pair that motivates the ordering: both spellings collapse, neither eats the other."""
    assert tile_access._sympify_tasklet_rhs("dace.float64(q) + 2") == symbolic.pystr_to_symbolic("q + 2")
    assert tile_access._sympify_tasklet_rhs("dace.float(q) + 2") == symbolic.pystr_to_symbolic("q + 2")


@pytest.mark.parametrize("text", ["int_floor(i, 4) + 3", "foo(i) + 3", "int_ceil(i, 4) + 3"])
def test_a_call_that_is_not_a_dtype_cast_survives_the_strip(text):
    """The empty-bracket control: the stripper leaves every non-cast call in place, so a collapse
    above is the stripper acting rather than sympy folding the call away on its own."""
    assert tile_access._sympify_tasklet_rhs(text) == symbolic.pystr_to_symbolic(text)
