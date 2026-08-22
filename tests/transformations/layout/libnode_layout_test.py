# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Layout changes over the remaining native library nodes: an Einsum operand permutation (the P3
``transform_einsum`` rewrite, end-to-end compile+run), and the layout-agnostic copy / memset.

The P3 rewrites (``rewrite_libnodes.py``) make a layout change reach a libnode's SEMANTIC indices:
Gemm->TensorDot, einsum-subscript, Reduce.axes, Scan.stride are covered in ``rewrite_libnodes_test``.
This file adds the missing end-to-end coverage:

  * **Einsum operand permute** -- storing an operand transposed and rewriting its subscripts keeps
    the contraction bit-exact (previously only string-level unit tests existed).
  * **memset** (a zero-init tasklet) -- layout-agnostic: permuting an array through it stays exact.
  * **copy** (a ``CopyLibraryNode``) -- layout-agnostic ONLY while both operands keep the SAME
    layout; a layout change that leaves the two operands with DIFFERENT layouts turns the copy into
    a transpose, which the copy libnode cannot express (it asks for a Transpose libnode). That case
    is an xfail: the missing P3 rule is Copy->TensorTranspose (the analog of Gemm->TensorDot).
"""
import numpy
import pytest
import dace

from dace.libraries.blas.nodes.einsum import Einsum
from dace.transformation.layout.rewrite_libnodes import (transform_einsum, RewriteCopyForLayout, copy_permutation_axes)
from dace.transformation.layout.permute_dimensions import PermuteDimensions
from dace.transformation.layout.prepare import prepare_for_layout

M, K, Nn = (dace.symbol(s) for s in ("M", "K", "Nn"))


# --------------------------------------------------------------------------- #
#  Einsum operand permutation -- end-to-end compile + run
# --------------------------------------------------------------------------- #
def _einsum_sdfg(einsum_str, sa, sb, so):
    sdfg = dace.SDFG("es_" + einsum_str.replace(",", "_").replace("->", "to"))
    sdfg.add_array("A", sa, dace.float64)
    sdfg.add_array("B", sb, dace.float64)
    sdfg.add_array("C", so, dace.float64)
    st = sdfg.add_state("s", is_start_block=True)
    node = Einsum("einsum")
    node.einsum_str = einsum_str
    node.add_in_connector("_i0")
    node.add_in_connector("_i1")
    node.add_out_connector("_o")
    st.add_node(node)
    st.add_edge(st.add_read("A"), None, node, "_i0", dace.Memlet.from_array("A", sdfg.arrays["A"]))
    st.add_edge(st.add_read("B"), None, node, "_i1", dace.Memlet.from_array("B", sdfg.arrays["B"]))
    st.add_edge(node, "_o", st.add_write("C"), None, dace.Memlet.from_array("C", sdfg.arrays["C"]))
    return sdfg


def test_einsum_permute_operand0_bitexact():
    """Store operand A transposed ([K,M]) and rewrite ij->ji: the contraction is still A @ B."""
    _M, _K, _N = 4, 5, 6
    A = numpy.random.default_rng(0).random((_M, _K))
    B = numpy.random.default_rng(1).random((_K, _N))

    new_str = transform_einsum("ij,jk->ik", 0, (1, 0))
    assert new_str == "ji,jk->ik"
    sdfg = _einsum_sdfg(new_str, [_K, _M], [_K, _N], [_M, _N])
    sdfg.validate()
    C = numpy.zeros((_M, _N))
    sdfg(A=A.T.copy(), B=B.copy(), C=C)  # A stored transposed
    assert numpy.allclose(C, A @ B)


def test_einsum_permute_operand1_bitexact():
    """Store operand B transposed ([N,K]) and rewrite jk->kj: still A @ B."""
    _M, _K, _N = 4, 5, 6
    A = numpy.random.default_rng(2).random((_M, _K))
    B = numpy.random.default_rng(3).random((_K, _N))

    new_str = transform_einsum("ij,jk->ik", 1, (1, 0))
    assert new_str == "ij,kj->ik"
    sdfg = _einsum_sdfg(new_str, [_M, _K], [_N, _K], [_M, _N])
    sdfg.validate()
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), B=B.T.copy(), C=C)
    assert numpy.allclose(C, A @ B)


# --------------------------------------------------------------------------- #
#  memset / copy under a layout change
# --------------------------------------------------------------------------- #
@dace.program
def memset_add(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = 0.0
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = C[i, j] + A[i, j]


@dace.program
def elementwise_copy(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j]


@dace.program
def transient_copy(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    B = A.copy()  # an implicit copy A -> B (a transient staging buffer)
    for i, j in dace.map[0:M, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j] = B[i, j] * 2.0


def test_memset_permute_bitexact():
    """A memset (zero-init tasklet) is layout-agnostic: permuting the input array stays exact."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(0).random((_M, _N))
    sdfg = memset_add.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)
    assert numpy.allclose(C, A)


def test_copy_libnode_consistent_permute_bitexact():
    """An elementwise copy lifts to a CopyLibraryNode; permuting BOTH operands the same way keeps
    it an identity copy (same layout on each side) -- bit-exact."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(1).random((_M, _N))
    sdfg = elementwise_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0], "C": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)
    assert numpy.allclose(C, A)


def _has(sdfg, typename: str) -> bool:
    """True if any node of that type name exists anywhere in the SDFG (incl. nested)."""
    return any(type(n).__name__ == typename for n, _ in sdfg.all_nodes_recursive())


def test_copy_libnode_transposing_rewrite_bitexact():
    """Permuting only ONE side of a copy makes it transposing, and a plain copy cannot express that.
    PermuteDimensions converts the copy to a TensorTranspose ITSELF -- it is the pass that knows the
    permutation -- so nothing transposing is left for RewriteCopyForLayout to find. Bit-exact."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(2).random((_M, _N))
    sdfg = elementwise_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    assert _has(sdfg, "TensorTranspose"), "permute must replace the copy it made transposing"
    assert RewriteCopyForLayout().apply_pass(sdfg, {}) == 0  # already converted by the permute
    sdfg.validate()
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)
    assert numpy.allclose(C, A)


@dace.program
def elementwise_copy_square(A: dace.float64[M, M], C: dace.float64[M, M]):
    C[:] = A


def test_same_symbol_square_copy_is_not_silently_transposed():
    """The case that forced the conversion into the permute pass. With ONE symbol on both dims the
    relaid-out operand and the destination end up with identical shapes, identical strides AND
    identical subsets, so no downstream pass can tell an elementwise copy from a transposing one:
    copy_permutation_axes sees [M, M] vs [M, M] and reports 'no transpose'. Left to
    RewriteCopyForLayout this silently produced C = A^T instead of C = A."""
    _M = 5
    A = numpy.random.default_rng(7).random((_M, _M))
    sdfg = elementwise_copy_square.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    # the shapes carry nothing: the permutation is only knowable from the pass that applied it
    assert copy_permutation_axes([_M, _M], [_M, _M]) is None
    assert _has(sdfg, "TensorTranspose"), "permute must convert the copy it made transposing"
    sdfg.validate()
    C = numpy.zeros((_M, _M))
    sdfg(A=A.copy(), C=C, M=_M)
    assert numpy.allclose(C, A), "square same-symbol copy came out transposed"


def test_implicit_copy_normalized_and_layout_correct():
    """prepare_for_layout lifts the implicit A->B copy to a CopyLibraryNode (InsertExplicitCopies),
    so a layout change over A stays correct: the relaid-out copy becomes transposing and
    RewriteCopyForLayout converts it to a TensorTranspose. Bit-exact."""
    _M, _N = 6, 4
    A = numpy.random.default_rng(4).random((_M, _N))
    sdfg = transient_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)  # normalizes the implicit copy to a CopyLibraryNode
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    RewriteCopyForLayout().apply_pass(sdfg, {})
    C = numpy.zeros((_M, _N))
    sdfg(A=A.copy(), C=C, M=_M, Nn=_N)
    assert numpy.allclose(C, 2.0 * A)


@dace.program
def copy3(A: dace.float64[M, K, Nn], C: dace.float64[M, K, Nn]):
    for i, j, k in dace.map[0:M, 0:K, 0:Nn] @ dace.ScheduleType.Sequential:
        C[i, j, k] = A[i, j, k]


@pytest.mark.parametrize("perm", [[2, 0, 1], [1, 2, 0], [2, 1, 0], [0, 2, 1]])
def test_copy_libnode_transposing_3d_rewrite_bitexact(perm):
    """A 3D copy whose input is permuted becomes a genuine (non-self-inverse) transpose; the permute
    pass must emit the right axes for every 3D permutation, not just the self-inverse 2D swap. This
    is what pins the ``axes = P^-1`` convention for an input-side relayout."""
    _M, _K, _N = 3, 4, 5
    A = numpy.random.default_rng(sum(perm)).random((_M, _K, _N))
    sdfg = copy3.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": perm}, add_permute_maps=True).apply_pass(sdfg, {})
    assert RewriteCopyForLayout().apply_pass(sdfg, {}) == 0  # the permute already converted it
    sdfg.validate()
    C = numpy.zeros((_M, _K, _N))
    sdfg(A=A.copy(), C=C, M=_M, K=_K, Nn=_N)
    assert numpy.allclose(C, A)


@dace.program
def slice_copy(A: dace.float64[M, K, Nn], C: dace.float64[K, Nn]):
    # Copies only a SUB-REGION of A: one 2-D slice, not the whole 3-D array.
    C[:] = A[0, :, :]


def test_subregion_copy_is_refused_not_transposed_over_the_whole_array():
    """A copy that spans only part of the array is NOT a whole-array permute. TensorTranspose reads
    the array DESCRIPTORS, so emitting it here would transpose data the copy never touched; the
    correct answer is the permutation induced on the spanned dimensions, which a whole-array
    transpose cannot express. The pass must refuse rather than silently move the wrong data."""
    sdfg = slice_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    with pytest.raises(NotImplementedError, match="SUB-REGION"):
        PermuteDimensions(permute_map={"A": [2, 0, 1]}, add_permute_maps=True).apply_pass(sdfg, {})


@dace.program
def column_copy(A: dace.float64[M, Nn], C: dace.float64[M]):
    C[:] = A[:, 1]  # a 1-D region: only ONE spanned dimension


@dace.program
def half_dim_copy(A: dace.float64[M, Nn], C: dace.float64[M, Nn]):
    C[0:2, :] = A[2:4, :]  # 2 spanned dims, but a SUB-region (half of dim 0)


NSym = dace.symbol("NSym")
MSym = dace.symbol("MSym")


@dace.program
def matvec_wcr(A: dace.float64[NSym, MSym], v: dace.float64[MSym], out: dace.float64[NSym]):
    # The WCR privatisation of `out` produces a UNIT-ELEMENT copy (_wcr_priv_assign[0] -> out[i]).
    for i, j in dace.map[0:NSym, 0:MSym]:
        out[i] += A[i, j] * v[j]


def test_column_subset_copy_is_refused():
    """A column copy is a SUBSET copy, and subset copies are invalid under a layout change: the
    transpose it needs is the permutation induced on the single dimension it spans, not the array's
    permutation. Only a full-array copy or a unit-element copy is valid."""
    sdfg = column_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    with pytest.raises(NotImplementedError, match="SUB-REGION"):
        PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})


def test_unit_element_copy_is_valid():
    """A unit-element copy maps through the whole algebra trivially -- one element is one element
    whatever the permutation does to the strides around it -- so it is relaid out without a
    transpose. This is the shape a WCR privatisation produces (``_wcr_priv_assign[0] -> out[i]``),
    so refusing it would block permuting any reduction operand."""
    _N, _M = 6, 4
    A = numpy.random.default_rng(5).random((_N, _M))
    v = numpy.random.default_rng(6).random(_M)
    sdfg = matvec_wcr.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})
    sdfg.validate()
    out = numpy.zeros(_N)
    sdfg(A=A.copy(), v=v.copy(), out=out, NSym=_N, MSym=_M)
    assert numpy.allclose(out, A @ v)


def test_multidim_subregion_copy_is_refused():
    """Copying half of one dimension into another half is not a permutation of the array. The copy
    spans several dimensions of a SUB-region, so it needs the permutation induced on those
    dimensions -- a whole-array TensorTranspose (which reads the descriptors) would move data the
    copy never touched. Refuse rather than corrupt."""
    sdfg = half_dim_copy.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    with pytest.raises(NotImplementedError, match="SUB-REGION"):
        PermuteDimensions(permute_map={"A": [1, 0]}, add_permute_maps=True).apply_pass(sdfg, {})


def test_full_shape_copy_converts_with_the_inverse_axes():
    """The convention, pinned directly: relaying out the INPUT of a full-shape copy by P gives the
    transpose axes P^-1 (out_sizes[k] == in_sizes[axes[k]]). For the 3D permutation [2,0,1] the
    inverse is [1,2,0]."""
    sdfg = copy3.to_sdfg(simplify=True)
    prepare_for_layout(sdfg, validate=False)
    PermuteDimensions(permute_map={"A": [2, 0, 1]}, add_permute_maps=True).apply_pass(sdfg, {})
    transposes = [n for n, _ in sdfg.all_nodes_recursive() if type(n).__name__ == "TensorTranspose"]
    assert transposes, "full-shape copy should have become a TensorTranspose"
    assert list(transposes[0].axes) == [1, 2, 0], f"axes {transposes[0].axes} != P^-1 of [2,0,1]"


def test_copy_permutation_axes_helper():
    """The axis-inference used by RewriteCopyForLayout: a distinct-size permutation is recovered; an
    identity / reshape yields None; a repeated-size permutation is ambiguous and raises."""
    assert copy_permutation_axes([4, 6], [6, 4]) == [1, 0]
    assert copy_permutation_axes([2, 3, 4], [4, 2, 3]) == [2, 0, 1]
    assert copy_permutation_axes([4, 6], [4, 6]) is None  # identity order
    assert copy_permutation_axes([4, 4], [4, 4]) is None  # square, identity order -> plain copy
    assert copy_permutation_axes([4, 6], [4, 6, 1]) is None  # rank change (reshape)
    # a genuine repeated-size permutation (order differs, sizes repeat) is ambiguous -> raises:
    with pytest.raises(NotImplementedError):
        copy_permutation_axes([4, 4, 6], [4, 6, 4])


if __name__ == "__main__":
    test_einsum_permute_operand0_bitexact()
    test_einsum_permute_operand1_bitexact()
    test_memset_permute_bitexact()
    test_copy_libnode_consistent_permute_bitexact()
    test_copy_libnode_transposing_rewrite_bitexact()
    test_implicit_copy_normalized_and_layout_correct()
    test_copy_permutation_axes_helper()
    print("libnode layout tests PASS")
