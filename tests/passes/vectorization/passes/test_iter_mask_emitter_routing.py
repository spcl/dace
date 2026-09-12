# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Unit tests for the ``_iter_mask`` consumer side of the CPP emit helpers
(``_generate_code`` — emitter routing).

Verifies that when a per-op tasklet has an ``_iter_mask`` input connector and
``EmitCtx.mask_connector`` is set, the emitter selects the ``op + "_masked"``
template entry and passes the mask connector name through to ``.format()``.
The unsuffixed entries are untouched when ``mask_connector is None``.

Wiring of ``_iter_mask`` to per-op tasklets (the pipeline-level integration)
is a follow-up slice; these tests construct ``EmitCtx`` directly so the
consumer logic can be validated in isolation.
"""
import dace

from dace.transformation.passes.vectorization.utils.tasklets import (
    EmitCtx,
    _generate_code,
    _template_key,
)

N = dace.symbol("N")


def _make_ctx(templates: dict, mask_connector=None) -> EmitCtx:
    """Build an EmitCtx with a stub state/node so _generate_code can read out_edges."""
    sdfg = dace.SDFG("emit_routing_probe")
    sdfg.add_array("a", (8, ), dace.float64)
    sdfg.add_array("b", (8, ), dace.float64)
    sdfg.add_array("out", (8, ), dace.float64)
    if mask_connector:
        sdfg.add_array("_iter_mask", (8, ), dace.bool_, transient=True)
    state = sdfg.add_state()
    a, b, out_n = state.add_access("a"), state.add_access("b"), state.add_access("out")
    t = state.add_tasklet("t", {"a", "b"}, {"out"}, "out = a + b")
    state.add_edge(a, None, t, "a", dace.Memlet("a[0:8]"))
    state.add_edge(b, None, t, "b", dace.Memlet("b[0:8]"))
    state.add_edge(t, "out", out_n, None, dace.Memlet("out[0:8]"))
    if mask_connector:
        m = state.add_access("_iter_mask")
        t.add_in_connector(mask_connector)
        state.add_edge(m, None, t, mask_connector, dace.Memlet("_iter_mask[0:8]"))
    return EmitCtx(state=state,
                   node=t,
                   templates=templates,
                   vector_dtype=dace.float64,
                   vector_width=8,
                   vector_map_param="i",
                   is_commutative=True,
                   fallbackcode_due_to_types=False,
                   mask_connector=mask_connector)


def test_template_key_no_mask_returns_base():
    templates = {"+": "vector_add<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});"}
    ctx = _make_ctx(templates, mask_connector=None)
    assert _template_key(ctx, "+") == "+"


def test_template_key_mask_set_but_masked_variant_absent_returns_base():
    """If only the unsuffixed entry exists, ``_template_key`` falls back gracefully."""
    templates = {"+": "vector_add<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});"}
    ctx = _make_ctx(templates, mask_connector="_iter_mask")
    assert _template_key(ctx, "+") == "+"


def test_template_key_mask_set_returns_masked_variant():
    templates = {
        "+": "vector_add<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
        "+_masked": "vector_add_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
    }
    ctx = _make_ctx(templates, mask_connector="_iter_mask")
    assert _template_key(ctx, "+") == "+_masked"


def test_generate_code_routes_to_masked_template_with_mask_arg():
    """When mask_connector is set, _generate_code emits the masked template
    with the mask connector name passed through .format()."""
    templates = {
        "+": "vector_add<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
        "+_masked": "vector_add_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
    }
    ctx = _make_ctx(templates, mask_connector="_iter_mask")
    code = _generate_code(ctx, rhs1_="a", rhs2_="b", const1_=None, const2_=None, lhs_="out", op_="+")
    assert "vector_add_av_masked" in code
    assert "_iter_mask" in code
    assert "vector_add<" not in code


def test_generate_code_without_mask_uses_base_template():
    templates = {
        "+": "vector_add<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2});",
        "+_masked": "vector_add_av_masked<{dtype}, {vector_width}>({lhs}, {rhs1}, {rhs2}, {mask});",
    }
    ctx = _make_ctx(templates, mask_connector=None)
    code = _generate_code(ctx, rhs1_="a", rhs2_="b", const1_=None, const2_=None, lhs_="out", op_="+")
    assert "vector_add<" in code
    assert "_masked" not in code
    assert "_iter_mask" not in code
