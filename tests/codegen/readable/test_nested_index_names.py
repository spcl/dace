# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
A nested SDFG can bind a sub-block of a parent array to a connector of the same name, which gives it other strides.
Each index function must be defined once per translation unit, and arrays that differ in strides must not share one.
"""
import collections
import re

import numpy as np

import dace
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, run_isolated,
                                             use_implementation)

N = dace.symbol("N")


def nested_view_sdfg(inner_shape, inner_strides, inner_name="A"):
    """Parent A[N,N] element-wise map, plus a no_inline nested SDFG whose connector ``inner_name`` is a
    view of A with the given (shape, strides). Returns the top SDFG."""
    sdfg = dace.SDFG("nested_view")
    sdfg.add_array("A", [N, N], dace.float64)

    nsdfg = dace.SDFG("inner")
    nsdfg.add_array(inner_name, inner_shape, dace.float64, strides=inner_strides)
    ns = nsdfg.add_state("n")
    an = ns.add_access(inner_name)
    me, mx = ns.add_map("im", {"i": "0:2", "j": "0:2"})
    tk = ns.add_tasklet("t", {"x"}, {"o"}, "o = x + 1.0")
    ns.add_memlet_path(an, me, tk, dst_conn="x", memlet=dace.Memlet(f"{inner_name}[i,j]"))
    an2 = ns.add_access(inner_name)
    ns.add_memlet_path(tk, mx, an2, src_conn="o", memlet=dace.Memlet(f"{inner_name}[i,j]"))

    st = sdfg.add_state("main")
    # parent element-wise access -> outer A_idx (strides N,1)
    pr, pw = st.add_access("A"), st.add_access("A")
    pme, pmx = st.add_map("pm", {"i": "0:N", "j": "0:N"})
    ptk = st.add_tasklet("pt", {"x"}, {"o"}, "o = x * 2.0")
    st.add_memlet_path(pr, pme, ptk, dst_conn="x", memlet=dace.Memlet("A[i,j]"))
    st.add_memlet_path(ptk, pmx, pw, src_conn="o", memlet=dace.Memlet("A[i,j]"))
    # nested SDFG bound to a 2x2 sub-block of A
    ar, aw = st.add_access("A"), st.add_access("A")
    nn = st.add_nested_sdfg(nsdfg, {inner_name}, {inner_name}, symbol_mapping={"N": N})
    nn.no_inline = True
    st.add_edge(ar, None, nn, inner_name, dace.Memlet("A[0:2, 0:2]"))
    st.add_edge(nn, inner_name, aw, None, dace.Memlet("A[0:2, 0:2]"))
    sdfg.validate()
    return sdfg


INDEX_FUNCTION = re.compile(r"static\s+DACE_HDFI\s+constexpr\s+long\s+long\s+(\w+_idx(?:_\d+)?)\s*\(")


def index_functions(sdfg):
    with use_implementation(EXPERIMENTAL):
        code = "\n".join(obj.code for obj in sdfg.generate_code() if obj.language == "cpp")
    return collections.Counter(INDEX_FUNCTION.findall(code))


def test_a_nested_view_with_the_parent_name_gets_its_own_index_function():
    names = index_functions(nested_view_sdfg([2, 2], [2, 1]))
    assert len(names) >= 2, names
    assert set(names.values()) == {1}, names


def test_a_nested_array_with_the_parent_signature_shares_its_index_function():
    names = index_functions(nested_view_sdfg([N, N], [N, 1]))
    assert names["A_idx"] == 1, names


def test_a_nested_view_computes_the_same_result_as_the_legacy_generator():
    a = np.random.default_rng(0).random((6, 6))

    def run(implementation):

        def build_and_run():
            with use_implementation(implementation):
                compiled = nested_view_sdfg([2, 2], [2, 1]).compile()
            out = a.copy()
            compiled(A=out, N=6)
            return {"A": out}

        return run_isolated(build_and_run)

    assert_outputs_equivalent(run(LEGACY), run(EXPERIMENTAL), "cpu", label="nested_view")


if __name__ == "__main__":
    test_a_nested_view_with_the_parent_name_gets_its_own_index_function()
    test_a_nested_array_with_the_parent_signature_shares_its_index_function()
    test_a_nested_view_computes_the_same_result_as_the_legacy_generator()
