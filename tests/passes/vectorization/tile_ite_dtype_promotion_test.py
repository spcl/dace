# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``TileITE.validate()`` dtype promotion for a Tile-kind ``_t`` / ``_e`` arm.

Before this fix, a Tile-kind arm whose dtype differed from ``_o``'s -- even a widening,
value-preserving promotion (``float16`` -> ``float64``) -- raised ``NotImplementedError``
unconditionally: "TileITE requires uniform dtype across _t, _e and _o ... cast via separate
tasklet first", telling the caller to do work the node should do itself.

``TileBinop`` / ``TileFma`` / ``TileUnop`` already promote a Tile operand this way (design
6.2, ``tile_binop._promotion_ok``): same dtype, or a WIDENING conversion (int -> float/double,
narrower int -> wider int, float -> double, numeric -> bool) is allowed and resolved by the
pure/ISA expansion's own arithmetic-conversion context; only a genuinely narrowing conversion
(``double -> float``, ``float -> int``, int narrowing) still raises. ``TileITE`` now follows the
SAME rule, via the SAME ``_promotion_ok`` helper, so the three tile lib nodes with a Tile operand
agree on one promotion policy instead of ``TileITE`` carrying a stricter, second opinion.

This is a separate, narrower change from the ``np.where`` literal-arm fix covered by
``branched_tail_remainder_cudatest.py``'s ``test_branched_tail_where_literal_arm_typed_not_bare_double``
-- that reproducer's TileITE arms are already uniform-dtype by the time they reach ``validate()``
(the mismatch it hits is downstream, in a scalar tasklet that never becomes a TileITE), so this
relaxation is not exercised by it. It is tested standalone here, against hand-built SDFGs, because
provoking a genuinely mismatched-but-widening Tile arm end-to-end would need a kernel shape no
current frontend lowering produces.
"""
import dace
from dace.libraries.tileops import TileITE


def _build_sdfg(kind_t: str, t_dtype, e_dtype, out_dtype):
    """A minimal SDFG with one ``TileITE(name='probe_ite', widths=(2,), kind_t=kind_t,
    kind_e='Tile')`` node: a bool ``mask`` array, an ``e`` array (``_e``, always Tile-kind),
    a ``t`` array (``_t``, only when ``kind_t == 'Tile'``) and an ``o`` output array. Returns
    ``(sdfg, state, ite)`` with the node wired but NOT validated -- the caller validates."""
    sdfg = dace.SDFG("probe")
    sdfg.add_array("mask", (2, ), dace.bool_, transient=False)
    if kind_t == "Tile":
        sdfg.add_array("t", (2, ), t_dtype, transient=False)
    sdfg.add_array("e", (2, ), e_dtype, transient=False)
    sdfg.add_array("o", (2, ), out_dtype, transient=False)
    state = sdfg.add_state()
    ite = TileITE(name="probe_ite",
                  widths=(2, ),
                  kind_t=kind_t,
                  kind_e="Tile",
                  expr_t=("0.0" if kind_t == "Symbol" else None))
    state.add_node(ite)
    mask_an = state.add_access("mask")
    e_an = state.add_access("e")
    o_an = state.add_access("o")
    state.add_edge(mask_an, None, ite, "_mask", dace.Memlet("mask"))
    if kind_t == "Tile":
        t_an = state.add_access("t")
        state.add_edge(t_an, None, ite, "_t", dace.Memlet("t"))
    state.add_edge(e_an, None, ite, "_e", dace.Memlet("e"))
    state.add_edge(ite, "_o", o_an, None, dace.Memlet("o"))
    return sdfg, state, ite


def test_tile_ite_allows_widening_tile_arm():
    """A Tile ``_t`` arm narrower than ``_o`` (``float16`` -> ``float64``, a widening, exact
    conversion) must validate cleanly -- the old strict uniform-dtype check would have raised."""
    sdfg, state, ite = _build_sdfg("Tile", dace.float16, dace.float16, dace.float64)
    ite.validate(sdfg, state)  # must not raise


def test_tile_ite_still_rejects_narrowing_tile_arm():
    """A Tile ``_t`` arm WIDER than ``_o`` (``float64`` -> ``float16``, a narrowing, lossy
    conversion) must still raise -- the promotion relaxation is widening-only, matching
    ``TileBinop`` / ``TileFma`` / ``TileUnop``."""
    sdfg, state, ite = _build_sdfg("Tile", dace.float64, dace.float16, dace.float16)
    try:
        ite.validate(sdfg, state)
    except NotImplementedError as ex:
        assert "narrowing" in str(ex)
        return
    raise AssertionError("narrowing Tile _t arm (float64 -> float16) did not raise")


def test_tile_ite_symbol_arm_stays_exempt():
    """A Symbol arm (``kind_t='Symbol'``) is cast to ``_o``'s dtype inline at expansion, so it
    stays exempt from the Tile-arm promotion check regardless of ``_o``'s dtype -- unchanged by
    this fix, checked here so a future edit cannot fold Symbol arms into the same check by
    mistake."""
    sdfg, state, ite = _build_sdfg("Symbol", None, dace.float16, dace.float64)
    ite.validate(sdfg, state)  # must not raise


def test_tile_ite_treats_a_narrow_float_as_a_float():
    """``bfloat16`` and the two fp8 types are ml_dtypes scalars, and ``np.issubdtype(bfloat16,
    np.floating)`` is False, so the promotion rule read them as neither integer nor float and
    refused EVERY conversion off them -- a widening bfloat16 -> float32 included."""
    sdfg, state, ite = _build_sdfg("Tile", dace.bfloat16, dace.bfloat16, dace.float32)
    ite.validate(sdfg, state)  # must not raise


def test_tile_ite_takes_a_narrow_float_to_bool():
    """The shape the GPU vectorizer actually emits: a comparison over a bfloat16 array answers
    into a bool tile, which is a numeric -> bool truthiness cast and not a narrowing."""
    sdfg, state, ite = _build_sdfg("Tile", dace.bfloat16, dace.bfloat16, dace.bool_)
    ite.validate(sdfg, state)  # must not raise


def test_tile_ite_rejects_a_same_width_float_swap():
    """``float16`` -> ``bfloat16`` is two bytes either way, but they spend those bytes
    differently -- neither direction round-trips -- so a width comparison alone must not admit
    it. Only a STRICTLY wider float, or the same dtype, is a promotion."""
    sdfg, state, ite = _build_sdfg("Tile", dace.float16, dace.bfloat16, dace.bfloat16)
    try:
        ite.validate(sdfg, state)
    except NotImplementedError as ex:
        assert "narrowing" in str(ex)
        return
    raise AssertionError("float16 -> bfloat16 Tile arm did not raise")


if __name__ == "__main__":
    test_tile_ite_allows_widening_tile_arm()
    test_tile_ite_still_rejects_narrowing_tile_arm()
    test_tile_ite_symbol_arm_stays_exempt()
    test_tile_ite_treats_a_narrow_float_as_a_float()
    test_tile_ite_takes_a_narrow_float_to_bool()
    test_tile_ite_rejects_a_same_width_float_swap()
    print("all tile_ite dtype promotion tests passed")
