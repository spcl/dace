# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""A layout transform must apply EXACTLY ONCE to an array a nested SDFG both reads and writes.

DaCe's standard in-place representation passes such an array with the SAME inner connector name on
the nested node's in-edge (``dst_conn``) and out-edge (``src_conn``). Every transform here iterates
in-edges then out-edges to find its nested targets, so the pair used to yield the inner array TWICE
and the rewrite was applied twice: sigma^{-1} composed on itself, a pad added twice, a split index
split again. Each transform's own suite passes the array as input-only (distinct in/out names), so
none of them covered this. Bit-exactness cannot catch a doubled pad either -- it is a descriptor
inconsistency -- so these assert on the descriptors and memlets directly."""
import dace
import sympy as sp
from dace import nodes

from dace.transformation.layout.pad_dimensions import PadDimensions
from dace.transformation.layout.shuffle_elements import ShuffleElements
from dace.transformation.layout.split_dimensions import SplitDimensions
from dace.transformation.layout.unblock_dimensions import UnblockDimensions
from dace.libraries.layout.shuffle import register_shuffle

N = dace.symbol("N")

try:
    register_shuffle("rw_xor3", "i ^ 3", "i ^ 3")
except Exception:  # already registered by another test module
    pass


def _readwrite_nested(shape=(N, ), name="X"):
    """An SDFG whose array flows into a NestedSDFG as BOTH input and output (one inner name)."""
    inner = dace.SDFG("inner_rw")
    inner.add_array(name, list(shape), dace.float64)
    ist = inner.add_state("i", is_start_block=True)
    params = {f"i{d}": f"0:{s}" for d, s in enumerate(shape)}
    sub = ",".join(params)
    me, mx = ist.add_map("m", params)
    t = ist.add_tasklet("t", {"a"}, {"b"}, "b = a + 1.0")
    ist.add_memlet_path(ist.add_read(name), me, t, dst_conn="a", memlet=dace.Memlet(f"{name}[{sub}]"))
    ist.add_memlet_path(t, mx, ist.add_write(name), src_conn="b", memlet=dace.Memlet(f"{name}[{sub}]"))

    outer = dace.SDFG("outer_rw")
    outer.add_array(name, list(shape), dace.float64)
    ost = outer.add_state("o", is_start_block=True)
    nsdfg = ost.add_nested_sdfg(inner, {name}, {name})  # same inner name on both sides
    whole = ",".join(f"0:{s}" for s in shape)
    ost.add_edge(ost.add_read(name), None, nsdfg, name, dace.Memlet(f"{name}[{whole}]"))
    ost.add_edge(nsdfg, name, ost.add_write(name), None, dace.Memlet(f"{name}[{whole}]"))
    return outer, nsdfg.sdfg


def test_pad_grows_a_readwrite_nested_array_once():
    """The inner descriptor must match the outer one. Growing it per in-edge AND per out-edge left
    the inner array padded twice (outer N+5, inner N+10) -- an allocation and bounds mismatch."""
    outer, inner = _readwrite_nested()
    PadDimensions(pad_map={"X": [5]}).apply_pass(outer, {})
    assert sp.simplify(outer.arrays["X"].shape[0] - (N + 5)) == 0
    assert sp.simplify(inner.arrays["X"].shape[0] - outer.arrays["X"].shape[0]) == 0
    assert sp.simplify(inner.arrays["X"].total_size - outer.arrays["X"].total_size) == 0


def test_shuffle_composes_the_inverse_once_on_a_readwrite_nested_array():
    """sigma^{-1} applied twice reads the wrong element: A'[sigma^{-1}(sigma^{-1}(e))]. The inner
    body memlet must carry exactly one inverse application."""
    outer, inner = _readwrite_nested()
    ShuffleElements(shuffle_map={"X": ("rw_xor3", 0)}).apply_pass(outer, {})
    subsets = [str(e.data.subset) for st in inner.states() for e in st.edges() if e.data and e.data.data == "X"]
    composed = [s for s in subsets if "shuffle_inv" in s]
    assert composed, "the inner body memlet was never composed with the inverse"
    for s in composed:
        assert s.count("shuffle_inv") == 1, f"inverse applied more than once: {s}"


def test_split_dimensions_splits_a_readwrite_nested_array_once():
    """A second recursion would split the already-split index again."""
    outer, inner = _readwrite_nested(shape=(N, ))
    SplitDimensions(split_map={"X": ([True], [8])}).apply_pass(outer, {})
    # the inner descriptor gained exactly one dimension (rank 1 -> 2), not two
    assert len(inner.arrays["X"].shape) == 2, f"inner rank {len(inner.arrays['X'].shape)} != 2"
    assert len(outer.arrays["X"].shape) == len(inner.arrays["X"].shape)


def test_unblock_reverses_a_readwrite_nested_array_once():
    """Block then Unblock must round-trip the read-write nested array back to its original rank."""
    outer, inner = _readwrite_nested(shape=(N, ))
    SplitDimensions(split_map={"X": ([True], [8])}).apply_pass(outer, {})
    UnblockDimensions(unblock_map={"X": ([True], [8])}).apply_pass(outer, {})
    assert len(outer.arrays["X"].shape) == 1
    assert len(inner.arrays["X"].shape) == 1, "inner array not unblocked back to rank 1"


if __name__ == "__main__":
    test_pad_grows_a_readwrite_nested_array_once()
    test_shuffle_composes_the_inverse_once_on_a_readwrite_nested_array()
    test_split_dimensions_splits_a_readwrite_nested_array_once()
    test_unblock_reverses_a_readwrite_nested_array_once()
    print("nested read-write tests PASS")
