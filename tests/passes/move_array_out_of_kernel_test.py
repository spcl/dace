# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Tests for :class:`MoveArrayOutOfKernel`."""

import ast
import re

import numpy as np
import pytest
import sympy

import dace
from dace import dtypes
from dace.sdfg.state import LoopRegion
from dace.transformation.helpers import nest_state_subgraph
from dace.transformation.passes.move_array_out_of_kernel import (
    MoveArrayOutOfKernel,
    lifted_layout,
    prepend_subscript_indices,
    tile_extent,
)

NX, NZ = (dace.symbol(s, dtype=dace.int64) for s in ("NX", "NZ"))
GLOBAL = dtypes.StorageType.GPU_Global
REGISTER = dtypes.StorageType.Register
GPU_DEVICE = dtypes.ScheduleType.GPU_Device
SEQUENTIAL = dtypes.ScheduleType.Sequential
LIMIT = MoveArrayOutOfKernel().register_limit
LAYOUTS = ["coalesced", "contiguous"]
ROWS, COLS = 8, 8

B_I, N = sympy.symbols("b_i N")


@pytest.mark.parametrize(
    "max_elem, min_elem, expected",
    [
        pytest.param(sympy.Min(N - 1, B_I + 31), B_I, 32, id="min_bounded_tile_has_its_static_width"),
        pytest.param(N - 1, sympy.Integer(0), N, id="plain_range_has_its_symbolic_extent"),
    ],
)
def test_tile_extent(max_elem, min_elem, expected):
    """A ``Min``-bounded tile must not leak the outer tile origin into its extent."""
    assert sympy.simplify(tile_extent(max_elem, min_elem) - expected) == 0


@pytest.mark.parametrize(
    "ranges, shape, strides, expected_shape, coalesced_strides, contiguous_strides",
    [
        (dict(i="0:128", j="0:32"), [64], None, [128, 32, 64], [32, 1, 4096], [2048, 64, 1]),
        (dict(i="0:8"), [4, 16], [1, 4], [8, 4, 16], [1, 8, 32], [64, 1, 4]),
        (dict(i="0:8"), [4, 16], [32, 2], None, None, None),
    ],
)
def test_lifted_dimensions_are_prepended_keeping_the_own_layout(
    ranges, shape, strides, expected_shape, coalesced_strides, contiguous_strides
):
    """Map dimensions vary fastest when coalesced, the last (``threadIdx.x``) innermost, and slowest when contiguous;
    C or Fortran keep the order of their own axes, any other layout is refused."""
    state = dace.SDFG("move_array_strides").add_state("s")
    me, _ = state.add_map("kernel", ranges, schedule=GPU_DEVICE)
    arr = dace.data.Array(dace.float64, shape, strides=strides)
    if expected_shape is None:
        with pytest.raises(NotImplementedError):
            lifted_layout(arr, [me])
        return

    for coalesced, expected_strides in ((True, coalesced_strides), (False, contiguous_strides)):
        layout = lifted_layout(arr, [me], coalesced)

        assert [int(s) for s in layout.shape] == expected_shape, layout.shape
        assert [int(s) for s in layout.strides] == expected_strides, layout.strides
        assert int(layout.total_size) == int(np.prod(expected_shape)), layout.total_size


def kernel_with_internal_transient(shape=(1024,), storage=GLOBAL) -> dace.SDFG:
    """``A[i] = buf[0]`` in a ``GPU_Device`` map, through a kernel-local ``buf``."""
    sdfg = dace.SDFG("flat_lift")
    sdfg.add_array("A", [128], dace.float64, storage=GLOBAL)
    sdfg.add_transient("buf", shape, dace.float64, storage=storage)

    state = sdfg.add_state("s")
    me, mx = state.add_map("kernel", dict(i="0:128"), schedule=GPU_DEVICE)
    buf = state.add_access("buf")
    produce = state.add_tasklet("produce", {}, {"o": None}, "o = 1.0")
    state.add_edge(me, None, produce, None, dace.Memlet())
    state.add_edge(produce, "o", buf, None, dace.Memlet("buf[0]"))
    consume = state.add_tasklet("consume", {"b": None}, {"o": None}, "o = b")
    state.add_edge(buf, None, consume, "b", dace.Memlet("buf[0]"))
    state.add_memlet_path(consume, mx, state.add_write("A"), src_conn="o", memlet=dace.Memlet("A[i]"))
    sdfg.validate()
    return sdfg


def transient_body() -> dace.SDFG:
    """Nested body writing ``a_out[i]`` of a 128-element array through its own ``buf[1024]`` transient."""
    inner = dace.SDFG("inner")
    inner.add_symbol("i", dace.int64)
    inner.add_array("a_out", [128], dace.float64, storage=GLOBAL)
    inner.add_transient("buf", [1024], dace.float64, storage=GLOBAL)
    inner_state = inner.add_state("i", is_start_block=True)
    buf = inner_state.add_access("buf")
    produce = inner_state.add_tasklet("produce", {}, {"o": None}, "o = 1.0")
    inner_state.add_edge(produce, "o", buf, None, dace.Memlet("buf[0]"))
    consume = inner_state.add_tasklet("consume", {"b": None}, {"o": None}, "o = b")
    inner_state.add_edge(buf, None, consume, "b", dace.Memlet("buf[0]"))
    inner_state.add_edge(consume, "o", inner_state.add_write("a_out"), None, dace.Memlet("a_out[i]"))
    return inner


def kernel_with_transient_behind_a_nested_sdfg() -> dace.SDFG:
    """The same transient, one nested-SDFG boundary below the kernel."""
    inner = transient_body()

    sdfg = dace.SDFG("nested_lift")
    sdfg.add_array("A", [128], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("s")
    me, mx = state.add_map("kernel", dict(i="0:128"), schedule=GPU_DEVICE)
    nsdfg = state.add_nested_sdfg(inner, {}, {"a_out": None})
    state.add_edge(me, None, nsdfg, None, dace.Memlet())
    state.add_memlet_path(nsdfg, mx, state.add_write("A"), src_conn="a_out", memlet=dace.Memlet("A[i]"))
    sdfg.validate()
    return sdfg


def buf_scopes(sdfg: dace.SDFG, name: str = "buf") -> list:
    return [
        state.scope_dict()[node]
        for sub in sdfg.all_sdfgs_recursive()
        for state in sub.states()
        for node in state.data_nodes()
        if node.data == name
    ]


def lift(sdfg: dace.SDFG, layout: str = "coalesced") -> int | None:
    sut = MoveArrayOutOfKernel()
    sut.layout = layout
    return sut.apply_pass(sdfg, {})


@pytest.mark.parametrize("storage", [REGISTER, dtypes.StorageType.Default])
def test_a_constant_size_per_thread_array_within_the_limit_stays_in_place(storage):
    sdfg = kernel_with_internal_transient((LIMIT,), storage)

    assert lift(sdfg) is None

    assert sdfg.arrays["buf"].storage == storage and tuple(sdfg.arrays["buf"].shape) == (LIMIT,)


def test_a_constant_size_register_array_over_the_limit_is_moved_out():
    sdfg = kernel_with_internal_transient((LIMIT + 1,), REGISTER)

    assert lift(sdfg) == 1

    desc = sdfg.arrays["buf"]
    assert desc.storage == GLOBAL and desc.transient and tuple(desc.shape) == (128, LIMIT + 1), desc
    assert None in buf_scopes(sdfg), "nothing carries the array out of the kernel"
    sdfg.validate()


def test_the_register_limit_is_configurable():
    sdfg = kernel_with_internal_transient((LIMIT + 1,), REGISTER)
    sut = MoveArrayOutOfKernel()
    sut.register_limit = LIMIT + 1

    assert sut.apply_pass(sdfg, {}) is None

    assert sdfg.arrays["buf"].storage == REGISTER and tuple(sdfg.arrays["buf"].shape) == (LIMIT + 1,)


def test_a_small_global_array_is_demoted_to_registers():
    sdfg = kernel_with_internal_transient((8,))

    assert lift(sdfg) == 1

    assert sdfg.arrays["buf"].storage == REGISTER and tuple(sdfg.arrays["buf"].shape) == (8,)


@pytest.mark.parametrize("lifetime", [dace.AllocationLifetime.Persistent, dace.AllocationLifetime.External])
def test_a_small_transient_outliving_the_invocation_is_lifted_not_demoted(lifetime):
    """A register cannot outlive the invocation a persistent or external array is kept across."""
    sdfg = kernel_with_internal_transient((8,))
    sdfg.arrays["buf"].lifetime = lifetime

    assert lift(sdfg) == 1

    assert sdfg.arrays["buf"].storage == GLOBAL and tuple(sdfg.arrays["buf"].shape) == (128, 8)
    assert sdfg.arrays["buf"].lifetime == lifetime
    sdfg.validate()


def test_a_reference_is_not_moved():
    """A reference names memory its ``set`` connector gives it (e.g. a memlet access policy's cursor), so it owns
    nothing to move."""
    sdfg = kernel_with_internal_transient((NX,), REGISTER)
    sdfg.arrays["buf"] = dace.data.ArrayReference(dace.float64, (NX,), storage=REGISTER, transient=True)

    assert lift(sdfg) is None

    assert isinstance(sdfg.arrays["buf"], dace.data.ArrayReference) and sdfg.arrays["buf"].storage == REGISTER


def test_a_symbolically_sized_register_array_is_moved_to_global_memory():
    """GPUs cannot hold a variable-length array in registers."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    body.arrays["tmp"].storage = REGISTER

    assert lift(sdfg) == 1

    assert sdfg.arrays["tmp"].storage == GLOBAL and len(sdfg.arrays["tmp"].shape) == 2, sdfg.arrays["tmp"]
    assert body.arrays["tmp"].storage == GLOBAL and not body.arrays["tmp"].transient
    sdfg.validate()


def test_a_default_storage_array_is_left_to_storage_inference():
    """``Default`` is not known to be per-thread; code generation resolves it before the pass runs."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    body.arrays["tmp"].storage = dtypes.StorageType.Default

    assert lift(sdfg) is None

    assert "tmp" not in sdfg.arrays and body.arrays["tmp"].transient


def test_transient_behind_a_nested_sdfg_is_lifted_through_the_boundary():
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    assert lift(sdfg) == 1

    assert tuple(sdfg.arrays["buf"].shape) == (128, 1024), sdfg.arrays["buf"].shape
    assert sdfg.arrays["buf"].transient
    inner = next(s for s in sdfg.all_sdfgs_recursive() if s is not sdfg)
    assert not inner.arrays["buf"].transient, "the inner copy must be passed in, not allocated in-kernel"
    assert buf_scopes(sdfg) == [None, None], buf_scopes(sdfg)
    sdfg.validate()


def kernel_with_a_tasklet_naming_the_buffer(language: dtypes.Language) -> dace.SDFG:
    """``B[i] = buf[1]`` read by a tasklet whose body names the buffer instead of a connector."""
    sdfg = kernel_with_internal_transient()
    sdfg.add_array("B", [128], dace.float64, storage=GLOBAL)
    state = sdfg.start_state
    entry = next(n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry))
    code = "o = buf[1]" if language == dtypes.Language.Python else "o = buf[1];"
    peek = state.add_tasklet("peek", {}, {"o": None}, code, language=language)
    state.add_edge(entry, None, peek, None, dace.Memlet())
    state.add_memlet_path(peek, state.exit_node(entry), state.add_write("B"), src_conn="o", memlet=dace.Memlet("B[i]"))
    sdfg.validate()
    return sdfg


def test_a_tasklet_body_naming_a_lifted_buffer_gains_the_kernel_index():
    sdfg = kernel_with_a_tasklet_naming_the_buffer(dtypes.Language.Python)
    lift(sdfg)

    peek = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.Tasklet) and n.label == "peek")
    assert peek.code.as_string == "o = buf[i, 1]", peek.code.as_string


def test_a_cpp_tasklet_subscripting_a_lifted_buffer_is_refused():
    sdfg = kernel_with_a_tasklet_naming_the_buffer(dtypes.Language.CPP)

    with pytest.raises(NotImplementedError, match="subscripts 'buf'"):
        lift(sdfg)


def flat_offsets(desc: dace.data.Array, subset: dace.subsets.Range, values: dict) -> tuple[int, int]:
    strides = [int(dace.symbolic.evaluate(s, values)) for s in desc.strides]
    corners = (subset.min_element(), subset.max_element())
    return tuple(
        sum(int(dace.symbolic.evaluate(b, values)) * s for b, s in zip(c, strides, strict=True)) for c in corners
    )


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("start, end, step, trips", [(1, 126, 1, 126), (0, 127, 2, 64)])
def test_the_lifted_dimension_holds_one_row_per_kernel_iteration(start, end, step, trips, layout):
    """Iteration ``i`` of ``start:end+1:step`` owns row ``(i - start) // step``: no row is skipped or shared."""
    sdfg = kernel_with_internal_transient()
    kernel = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.MapEntry))
    kernel.map.range = dace.subsets.Range([(start, end, step)])
    lift(sdfg, layout)

    desc = sdfg.arrays["buf"]
    assert tuple(desc.shape) == (trips, 1024), desc.shape
    total = int(desc.total_size)
    inner = [e for e in sdfg.start_state.edges() if e.data.data == "buf" and isinstance(e.src, dace.nodes.Tasklet)]
    firsts = {flat_offsets(desc, inner[0].data.subset, {"i": i})[0] for i in range(start, end + 1, step)}
    assert len(firsts) == trips and min(firsts) == 0 and max(firsts) < total, (len(firsts), max(firsts), total)


KERNEL_EXTENT, BLOCK_EXTENT, SCRATCH_EXTENT = 3, 5, LIMIT + 1


def kernel_with_a_thread_block_around_its_scratch() -> dace.SDFG:
    """``a[i, t] = tmp[2]`` with ``tmp[2] = i + 10 * t``; ``tmp`` sits in thread-block map ``t`` in kernel ``i``."""
    sdfg = dace.SDFG("scratch_inside_a_thread_block")
    sdfg.add_array("a", [KERNEL_EXTENT, BLOCK_EXTENT], dace.float64, storage=GLOBAL)
    sdfg.add_array("tmp", [SCRATCH_EXTENT], dace.float64, transient=True, storage=GLOBAL)
    state = sdfg.add_state("grid", is_start_block=True)
    kernel_entry, kernel_exit = state.add_map("kernel", dict(i=f"0:{KERNEL_EXTENT}"), schedule=GPU_DEVICE)
    block_entry, block_exit = state.add_map(
        "block", dict(t=f"0:{BLOCK_EXTENT}"), schedule=dtypes.ScheduleType.GPU_ThreadBlock
    )
    fill = state.add_tasklet("fill", {}, {"__out": None}, "__out = i + 10.0 * t")
    use = state.add_tasklet("use", {"__in": None}, {"__out": None}, "__out = __in")
    scratch = state.add_access("tmp")
    state.add_nedge(kernel_entry, block_entry, dace.Memlet())
    state.add_nedge(block_entry, fill, dace.Memlet())
    state.add_edge(fill, "__out", scratch, None, dace.Memlet("tmp[2]"))
    state.add_edge(scratch, None, use, "__in", dace.Memlet("tmp[2]"))
    state.add_memlet_path(
        use, block_exit, kernel_exit, state.add_write("a"), src_conn="__out", memlet=dace.Memlet("a[i, t]")
    )
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("layout", LAYOUTS)
def test_each_thread_owns_one_slice_and_moves_it_out_of_the_kernel(layout):
    """Shape gains ``[KERNEL, BLOCK]`` outermost first; every access leads with ``[i, t]`` and no map exit carries data
    (an edge between two accesses of one array is a copy into the whole array, which races)."""
    sdfg = kernel_with_a_thread_block_around_its_scratch()
    lift(sdfg, layout)

    desc = sdfg.arrays["tmp"]
    assert [int(s) for s in desc.shape] == [KERNEL_EXTENT, BLOCK_EXTENT, SCRATCH_EXTENT], desc.shape
    # Coalesced: neighboring threads ``t`` touch neighboring elements; contiguous: one thread's slice is consecutive
    expected = {
        "coalesced": [BLOCK_EXTENT, 1, KERNEL_EXTENT * BLOCK_EXTENT],
        "contiguous": [BLOCK_EXTENT * SCRATCH_EXTENT, SCRATCH_EXTENT, 1],
    }[layout]
    assert [int(s) for s in desc.strides] == expected, desc.strides
    accesses = [
        e
        for e in sdfg.start_state.edges()
        if e.data.data == "tmp" and (isinstance(e.src, dace.nodes.Tasklet) or isinstance(e.dst, dace.nodes.Tasklet))
    ]
    assert len(accesses) == 2, [str(e.data) for e in accesses]
    for edge in accesses:
        offsets = {
            flat_offsets(desc, edge.data.subset, {"i": i, "t": t})[0]
            for i in range(KERNEL_EXTENT)
            for t in range(BLOCK_EXTENT)
        }
        assert len(offsets) == KERNEL_EXTENT * BLOCK_EXTENT, f"{edge.data}: two iterations share one slice"
        assert min(offsets) >= 0 and max(offsets) < int(desc.total_size), (edge.data, min(offsets), max(offsets))
    moved = [str(e.data) for e in sdfg.start_state.edges() if isinstance(e.dst, dace.nodes.MapExit) and e.data.data]
    assert all(not m.startswith("tmp") for m in moved), moved
    outside = [n for n in sdfg.start_state.data_nodes() if n.data == "tmp" and sdfg.start_state.entry_node(n) is None]
    assert len(outside) == 1, "nothing places the array outside the kernel"


def add_sequential_map(state: dace.SDFGState, param: str, extent, read: str, write: str, *, code: str):
    state.add_mapped_tasklet(
        f"map_{param}",
        {param: f"0:{extent}"},
        {"__in": dace.Memlet(read)},
        f"__out = {code}",
        {"__out": dace.Memlet(write)},
        schedule=SEQUENTIAL,
        external_edges=True,
    )


def kernel_with_a_locally_named_scratch_extent() -> dace.SDFG:
    """Transient extent ``M`` is local to the nested SDFG, bound to the outer ``NZ - 1``."""
    inner = dace.SDFG("inner_local_scratch")
    inner.add_symbol("M", dace.int64)
    inner.add_array("a_in", [NZ], dace.float64, storage=GLOBAL)
    inner.add_array("out_in", [NZ], dace.float64, storage=GLOBAL)
    inner.add_array("tmp", ["M"], dace.float64, transient=True, storage=GLOBAL)
    fill = inner.add_state("fill", is_start_block=True)
    add_sequential_map(fill, "m", "M", "a_in[m]", "tmp[m]", code="__in * 2.0")
    add_sequential_map(inner.add_state_after(fill, "drain"), "m", "M", "tmp[m]", "out_in[m]", code="__in + 1.0")

    sdfg = dace.SDFG("locally_named_scratch_extent")
    sdfg.add_array("a", [NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array("out", [NZ], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("grid", is_start_block=True)
    kernel_entry, kernel_exit = state.add_map("kernel", dict(i="0:1"), schedule=GPU_DEVICE)
    nsdfg = state.add_nested_sdfg(inner, {"a_in": None}, {"out_in": None}, symbol_mapping={"M": NZ - 1})
    state.add_memlet_path(state.add_read("a"), kernel_entry, nsdfg, dst_conn="a_in", memlet=dace.Memlet("a[0:NZ]"))
    state.add_memlet_path(
        nsdfg, kernel_exit, state.add_write("out"), src_conn="out_in", memlet=dace.Memlet("out[0:NZ]")
    )
    sdfg.validate()
    return sdfg


def test_lift_translates_a_locally_named_shape_symbol_through_symbol_mapping():
    """The lifted descriptor must say ``NZ - 1``, not the inner-only ``M``."""
    sdfg = kernel_with_a_locally_named_scratch_extent()
    lift(sdfg)

    desc = sdfg.arrays["tmp"]
    assert sympy.simplify(desc.shape[-1] - (NZ - 1)) == 0, desc.shape
    assert "M" not in sdfg.free_symbols
    assert "M" not in sdfg.arglist()
    sdfg.validate()


def kernel_with_scratch_below_a_nested_sdfg(storage=GLOBAL) -> dace.SDFG:
    """``out[i, k] = 2 * a[i, NZ - 1 - k] + 1`` through ``tmp[NZ]`` defined in ``body`` and passed further down."""
    producer = dace.SDFG("producer")
    producer.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    producer.add_array("tmp", [NZ], dace.float64, storage=storage)
    add_sequential_map(producer.add_state("fill", is_start_block=True), "k", NZ, "a[i, k]", "tmp[k]", code="__in * 2.0")

    consumer = dace.SDFG("consumer")
    consumer.add_array("tmp", [NZ], dace.float64, storage=storage)
    consumer.add_array("out", [NX, NZ], dace.float64, storage=GLOBAL)
    add_sequential_map(
        consumer.add_state("drain", is_start_block=True), "k", NZ, "tmp[NZ - 1 - k]", "out[i, k]", code="__in + 1.0"
    )

    body = dace.SDFG("body")
    body.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    body.add_array("out", [NX, NZ], dace.float64, storage=GLOBAL)
    body.add_array("tmp", [NZ], dace.float64, transient=True, storage=storage)
    mapping = dict(i="i", NX=NX, NZ=NZ)
    fill = body.add_state("call_producer", is_start_block=True)
    pnode = fill.add_nested_sdfg(producer, {"a": None}, {"tmp": None}, symbol_mapping=mapping)
    fill.add_edge(fill.add_read("a"), None, pnode, "a", dace.Memlet("a[0:NX, 0:NZ]"))
    fill.add_edge(pnode, "tmp", fill.add_write("tmp"), None, dace.Memlet("tmp[0:NZ]"))
    drain = body.add_state_after(fill, "call_consumer")
    cnode = drain.add_nested_sdfg(consumer, {"tmp": None}, {"out": None}, symbol_mapping=mapping)
    drain.add_edge(drain.add_read("tmp"), None, cnode, "tmp", dace.Memlet("tmp[0:NZ]"))
    drain.add_edge(cnode, "out", drain.add_write("out"), None, dace.Memlet("out[0:NX, 0:NZ]"))

    sdfg = dace.SDFG(f"scratch_below_a_nested_sdfg_{storage.name}")
    sdfg.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array("out", [NX, NZ], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("grid", is_start_block=True)
    entry, exit_node = state.add_map("kernel", dict(i="0:NX"), schedule=GPU_DEVICE)
    entry.map.gpu_block_size = [32, 1, 1]
    nsdfg = state.add_nested_sdfg(body, {"a": None}, {"out": None}, symbol_mapping=mapping)
    state.add_memlet_path(state.add_read("a"), entry, nsdfg, dst_conn="a", memlet=dace.Memlet("a[0:NX, 0:NZ]"))
    state.add_memlet_path(
        nsdfg, exit_node, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:NX, 0:NZ]")
    )
    sdfg.validate()
    return sdfg


def test_lift_gives_descendant_nested_sdfgs_the_lifted_descriptor():
    """A nested SDFG's data has the shape of the data connected to it (No-View nested SDFGs), so every nest below
    the definition takes the lifted descriptor and its memlets the slice index."""
    sdfg = kernel_with_scratch_below_a_nested_sdfg()
    lift(sdfg)

    ranks = {}
    for nested in sdfg.all_sdfgs_recursive():
        if "tmp" not in nested.arrays:
            continue
        rank = ranks[nested.name] = len(nested.arrays["tmp"].shape)
        for state in nested.states():
            for edge in state.edges():
                if edge.data.data == "tmp":
                    assert edge.data.subset.dims() == rank, (nested.name, str(edge.data))
    assert ranks == {sdfg.name: 2, "body": 2, "producer": 2, "consumer": 2}, ranks
    sdfg.validate()


def test_a_renamed_lift_gives_a_descendant_the_lifted_descriptor():
    """The owner is renamed around an unrelated outer ``tmp``; the nests below take the lifted rank-2 descriptor."""
    sdfg = kernel_with_scratch_below_a_nested_sdfg()
    sdfg.add_array("tmp", [4], dace.float64, storage=GLOBAL)
    lift(sdfg)

    assert tuple(sdfg.arrays["tmp"].shape) == (4,), "the colliding outer descriptor was overwritten"
    lifted = [name for name, desc in sdfg.arrays.items() if name != "tmp" and len(desc.shape) == 2 and desc.transient]
    assert len(lifted) == 1, sorted(sdfg.arrays)
    for nested in sdfg.all_sdfgs_recursive():
        if nested.name in ("producer", "consumer"):
            assert tuple(map(str, nested.arrays["tmp"].shape)) == ("NX", "NZ"), nested.arrays["tmp"].shape
    sdfg.validate()


def wrap_kernel_around_a_body_it_never_reaches() -> dace.SDFG:
    """``out[k] = 2 * a[NZ - 1 - k] + 1`` in a body under a size-1 kernel ``w`` the body never names."""
    body = dace.SDFG("wrapped_body")
    body.add_array("a", [NZ], dace.float64, storage=GLOBAL)
    body.add_array("out", [NZ], dace.float64, storage=GLOBAL)
    body.add_array("tmp", [NZ], dace.float64, transient=True, storage=GLOBAL)
    fill = body.add_state("fill", is_start_block=True)
    add_sequential_map(fill, "k", NZ, "a[k]", "tmp[k]", code="__in * 2.0")
    add_sequential_map(body.add_state_after(fill, "drain"), "k", NZ, "tmp[NZ - 1 - k]", "out[k]", code="__in + 1.0")

    sdfg = dace.SDFG("wrap_kernel_scratch")
    sdfg.add_array("a", [NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array("out", [NZ], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("grid", is_start_block=True)
    entry, exit_node = state.add_map("wrap", dict(w="0:1"), schedule=GPU_DEVICE)
    nsdfg = state.add_nested_sdfg(body, {"a": None}, {"out": None}, symbol_mapping=dict(NZ=NZ))
    state.add_memlet_path(state.add_read("a"), entry, nsdfg, dst_conn="a", memlet=dace.Memlet("a[0:NZ]"))
    state.add_memlet_path(nsdfg, exit_node, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:NZ]"))
    sdfg.validate()
    return sdfg


def test_a_lift_binds_the_kernel_parameter_it_indexes_by():
    """The new slice index is ``w``, which the body never used, so the nest must be bound to it."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    lift(sdfg)

    nest = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    assert str(nest.symbol_mapping.get("w")) == "w", nest.symbol_mapping
    sdfg.validate()


def test_a_nest_looping_over_the_kernel_parameter_name_is_lifted_with_its_loop_renamed():
    """Inside a loop over ``w`` the lifted index would name the loop counter, so the loop is renamed instead."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    loop = LoopRegion("again", "w < 1", "w", "w = 0", "w = w + 1")
    fill, drain = body.start_block, body.sink_nodes()[0]
    body.remove_nodes_from([fill, drain])
    loop.add_node(fill, is_start_block=True)
    loop.add_node(drain)
    loop.add_edge(fill, drain, dace.InterstateEdge())
    body.add_node(loop, is_start_block=True)
    sdfg.validate()

    assert lift(sdfg) == 1

    assert loop.loop_variable != "w", loop.loop_variable
    assert len(body.arrays["tmp"].shape) == 2 and not body.arrays["tmp"].transient
    accesses = [e.data for s in body.states() for e in s.edges() if e.data.data == "tmp"]
    assert accesses and all(str(m.subset[0][0]) == "w" for m in accesses), [str(m) for m in accesses]
    sdfg.validate()


def test_a_nest_below_a_body_that_never_named_the_kernel_parameter_is_indexed_by_it():
    """The owner gains ``w`` only through the lift, and the nest reading its buffer must see it too."""
    sdfg = wrap_kernel_around_a_body_it_never_reaches()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    drain = body.sink_nodes()[0]
    nest = nest_state_subgraph(body, drain, dace.sdfg.graph.SubgraphView(drain, list(drain.nodes())))
    sdfg.validate()

    assert lift(sdfg) == 1

    assert str(nest.symbol_mapping.get("w")) == "w", nest.symbol_mapping
    assert len(nest.sdfg.arrays["tmp"].shape) == 2, nest.sdfg.arrays["tmp"].shape
    sdfg.validate()


def kernel_over_k_beside_a_nested_k_loop() -> dace.SDFG:
    """Kernel ``k`` lifts a symbolic scratch; a second kernel's nest owns a ``k`` loop of its own."""
    scratch_body = dace.SDFG("scratch_body")
    scratch_body.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    scratch_body.add_array("mid", [NX, NZ], dace.float64, storage=GLOBAL)
    scratch_body.add_array("tmp", [NX], dace.float64, transient=True, storage=GLOBAL)
    fill = scratch_body.add_state("fill", is_start_block=True)
    add_sequential_map(fill, "i", NX, "a[i, k]", "tmp[i]", code="__in * 2.0")
    add_sequential_map(
        scratch_body.add_state_after(fill, "drain"), "i", NX, "tmp[NX - 1 - i]", "mid[i, k]", code="__in"
    )

    sweep = dace.SDFG("sweep")
    sweep.add_array("mid", [NX, NZ], dace.float64, storage=GLOBAL)
    sweep.add_array("out", [NX, NZ], dace.float64, storage=GLOBAL)
    loop = LoopRegion(
        "k_sweep", condition_expr="k < NZ", loop_var="k", initialize_expr="k = 0", update_expr="k = k + 1"
    )
    sweep.add_node(loop, is_start_block=True)
    step = loop.add_state("step", is_start_block=True)
    bump = step.add_tasklet("bump", {"__in": None}, {"__out": None}, "__out = __in + 1.0")
    step.add_edge(step.add_read("mid"), None, bump, "__in", dace.Memlet("mid[i, k]"))
    step.add_edge(bump, "__out", step.add_write("out"), None, dace.Memlet("out[i, k]"))

    sdfg = dace.SDFG("kernel_over_k_beside_a_nested_k_loop")
    sdfg.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array("mid", [NX, NZ], dace.float64, transient=True, storage=GLOBAL)
    sdfg.add_array("out", [NX, NZ], dace.float64, storage=GLOBAL)

    scratch = sdfg.add_state("scratch", is_start_block=True)
    entry, exit_node = scratch.add_map("kernel_k", dict(k="0:NZ"), schedule=GPU_DEVICE)
    entry.map.gpu_block_size = [32, 1, 1]
    snode = scratch.add_nested_sdfg(scratch_body, {"a": None}, {"mid": None}, symbol_mapping=dict(k="k", NX=NX, NZ=NZ))
    scratch.add_memlet_path(scratch.add_read("a"), entry, snode, dst_conn="a", memlet=dace.Memlet("a[0:NX, 0:NZ]"))
    scratch.add_memlet_path(
        snode, exit_node, scratch.add_write("mid"), src_conn="mid", memlet=dace.Memlet("mid[0:NX, 0:NZ]")
    )

    sweep_state = sdfg.add_state_after(scratch, "sweep")
    sentry, sexit = sweep_state.add_map("kernel_i", dict(i="0:NX"), schedule=GPU_DEVICE)
    sentry.map.gpu_block_size = [32, 1, 1]
    wnode = sweep_state.add_nested_sdfg(sweep, {"mid": None}, {"out": None}, symbol_mapping=dict(i="i", NX=NX, NZ=NZ))
    sweep_state.add_memlet_path(
        sweep_state.add_read("mid"), sentry, wnode, dst_conn="mid", memlet=dace.Memlet("mid[0:NX, 0:NZ]")
    )
    sweep_state.add_memlet_path(
        wnode, sexit, sweep_state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:NX, 0:NZ]")
    )
    sdfg.validate()
    return sdfg


def test_lift_does_not_bind_a_name_the_nest_assigns_itself():
    """A nest owning a ``k`` loop must not be bound to the kernel's ``k``: codegen then declares neither."""
    sdfg = kernel_over_k_beside_a_nested_k_loop()
    lift(sdfg)

    sweep = next(
        n for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG) and n.sdfg.name == "sweep"
    )
    assert "k" not in sweep.symbol_mapping, sweep.symbol_mapping
    sdfg.validate()
    code = "".join(obj.clean_code for obj in sdfg.generate_code())
    assert re.search(r"\w+ k;|for \(\w+ k = ", code), "the loop counter is used without a declaration in scope"


def kernel_with_interstate_buffer_read() -> dace.SDFG:
    """``out[i] = a[i, order[0]]``: an interstate edge reads one element of a kernel-local buffer."""
    inner = dace.SDFG("pick_body")
    inner.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    inner.add_array("out", [NX], dace.float64, storage=GLOBAL)
    inner.add_array("order", [NZ], dace.int64, transient=True, storage=GLOBAL)
    inner.add_symbol("sel", dace.int64)
    fill = inner.add_state("fill", is_start_block=True)
    fill.add_mapped_tasklet(
        "rank",
        {"k": "0:NZ"},
        {"__in": dace.Memlet("a[i, k]")},
        "__out = (NZ - 1 - k) if (__in > 0.5) else k",
        {"__out": dace.Memlet("order[k]")},
        schedule=SEQUENTIAL,
        external_edges=True,
    )
    use = inner.add_state("use")
    inner.add_edge(fill, use, dace.InterstateEdge(assignments={"sel": "order[0]"}))
    pick = use.add_tasklet("pick", {"__in": None}, {"__out": None}, "__out = __in")
    use.add_edge(use.add_read("a"), None, pick, "__in", dace.Memlet("a[i, sel]"))
    use.add_edge(pick, "__out", use.add_write("out"), None, dace.Memlet("out[i]"))

    sdfg = dace.SDFG("kernel_with_interstate_buffer_read")
    sdfg.add_array("a", [NX, NZ], dace.float64, storage=GLOBAL)
    sdfg.add_array("out", [NX], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("body", is_start_block=True)
    entry, exit_node = state.add_map("grid", dict(i="0:NX"), schedule=GPU_DEVICE)
    entry.map.gpu_block_size = [32, 1, 1]
    nsdfg = state.add_nested_sdfg(inner, {"a": None}, {"out": None}, symbol_mapping=dict(i="i", NX=NX, NZ=NZ))
    state.add_memlet_path(state.add_read("a"), entry, nsdfg, dst_conn="a", memlet=dace.Memlet("a[0:NX, 0:NZ]"))
    state.add_memlet_path(nsdfg, exit_node, state.add_write("out"), src_conn="out", memlet=dace.Memlet("out[0:NX]"))
    sdfg.validate()
    return sdfg


def test_an_interstate_read_of_a_lifted_buffer_gains_the_kernel_index():
    """``order[0]`` on an interstate edge must keep naming one element of the now rank-2 buffer."""
    sdfg = kernel_with_interstate_buffer_read()
    lift(sdfg)

    body = next(s for s in sdfg.all_sdfgs_recursive() if s.name == "pick_body")
    assert len(body.arrays["order"].shape) == 2, body.arrays["order"].shape
    reads = [
        node
        for edge in body.all_interstate_edges()
        for value in edge.data.assignments.values()
        for node in ast.walk(ast.parse(str(value)))
        if isinstance(node, ast.Subscript)
    ]
    assert [ast.unparse(r) for r in reads] == ["order[i, 0]"], [ast.unparse(r) for r in reads]
    sdfg.generate_code()


def test_a_symbol_mapping_read_of_a_lifted_buffer_gains_the_kernel_index():
    """A nest bound to ``order[1]`` must keep reading one element of the now rank-2 buffer."""
    sdfg = kernel_with_interstate_buffer_read()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    leaf = dace.SDFG("leaf")
    leaf.add_symbol("s", dace.int64)
    leaf.add_array("o", [1], dace.float64, storage=GLOBAL)
    leaf_state = leaf.add_state()
    write = leaf_state.add_tasklet("write", {}, {"x": None}, "x = s")
    leaf_state.add_edge(write, "x", leaf_state.add_write("o"), None, dace.Memlet("o[0]"))
    body.add_array("picked", [1], dace.float64, transient=True, storage=REGISTER)
    use = next(s for s in body.states() if s.label == "use")
    nest = use.add_nested_sdfg(leaf, {}, {"o": None}, symbol_mapping={"s": "order[1]"})
    use.add_edge(nest, "o", use.add_write("picked"), None, dace.Memlet("picked[0]"))

    lift(sdfg)

    assert str(nest.symbol_mapping["s"]) == "order[i, 1]", nest.symbol_mapping
    sdfg.validate()


def test_control_flow_reading_a_per_thread_buffer_is_refused():
    """Written inside a thread-block map of its own SDFG, the buffer has no one element for an interstate edge."""
    sdfg = kernel_with_interstate_buffer_read()
    body = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    fill = body.start_block
    fill.remove_nodes_from(list(fill.nodes()))
    me, mx = fill.add_map("lanes", dict(k="0:NZ"), schedule=dtypes.ScheduleType.GPU_ThreadBlock)
    write = fill.add_tasklet("write", {}, {"__out": None}, "__out = k")
    order = fill.add_access("order")
    fill.add_edge(me, None, write, None, dace.Memlet())
    fill.add_edge(write, "__out", order, None, dace.Memlet("order[k]"))
    fill.add_edge(order, None, mx, None, dace.Memlet())
    sdfg.validate()

    with pytest.raises(NotImplementedError, match="varies per GPU thread"):
        lift(sdfg)


def test_two_nests_defining_the_same_name_are_each_lifted_once():
    """Two sibling definitions of ``buf`` in one kernel are two arrays, each gaining one kernel dimension."""
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    state = sdfg.start_state
    first = next(n for n in state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    sdfg.add_array("B", [128], dace.float64, storage=GLOBAL)
    second = state.add_nested_sdfg(transient_body(), {}, {"a_out": None})
    kernel = state.entry_node(first)
    state.add_edge(kernel, None, second, None, dace.Memlet())
    state.add_memlet_path(
        second, state.exit_node(kernel), state.add_write("B"), src_conn="a_out", memlet=dace.Memlet("B[i]")
    )
    sdfg.validate()

    lift(sdfg)

    lifted = sorted(tuple(desc.shape) for name, desc in sdfg.arrays.items() if name not in ("A", "B"))
    assert lifted == [(128, 1024), (128, 1024)], {n: d.shape for n, d in sdfg.arrays.items()}
    sdfg.validate()


def test_a_transient_inside_a_loop_of_the_nested_body_is_lifted():
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    inner = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    body = inner.start_block
    loop = LoopRegion("repeat", "r < 2", "r", "r = 0", "r = r + 1")
    inner.remove_node(body)
    loop.add_node(body, is_start_block=True)
    inner.add_node(loop, is_start_block=True)
    sdfg.validate()

    assert lift(sdfg) == 1

    assert tuple(sdfg.arrays["buf"].shape) == (128, 1024), sdfg.arrays["buf"].shape
    sdfg.validate()


def kernel_with_a_transient_shared_by_two_kernels() -> dace.SDFG:
    """``buf`` of :func:`kernel_with_internal_transient`, written again by a second kernel."""
    sdfg = kernel_with_internal_transient()
    second = sdfg.add_state_after(sdfg.start_state, "again")
    me, mx = second.add_map("kernel2", dict(i="0:128"), schedule=GPU_DEVICE)
    produce = second.add_tasklet("produce", {}, {"o": None}, "o = 2.0")
    second.add_edge(me, None, produce, None, dace.Memlet())
    buf = second.add_access("buf")
    second.add_edge(produce, "o", buf, None, dace.Memlet("buf[1]"))
    second.add_edge(buf, None, mx, None, dace.Memlet())
    sdfg.validate()
    return sdfg


def test_a_transient_shared_by_two_kernels_is_refused():
    """One allocation used by two kernels has no single per-iteration slicing."""
    sdfg = kernel_with_a_transient_shared_by_two_kernels()

    with pytest.raises(NotImplementedError, match="shared by the kernels"):
        lift(sdfg)


def test_code_mentioning_the_name_without_subscripting_it_is_returned_verbatim():
    """A name that only appears as a substring must not count as a rewrite, not even through reformatting."""
    code = "x+1 if tmp_flag else 0"
    assert prepend_subscript_indices(code, "tmp", ["i"]) is code
    assert prepend_subscript_indices("tmp[0]+1", "tmp", ["i"]) == "tmp[i, 0] + 1"


def kernel_with_accumulator(rows: int = ROWS) -> dace.SDFG:
    """``out[i] = sum_j A[i, j]`` through a kernel-local one-element ``acc`` accumulated with a WCR."""
    sdfg = dace.SDFG("kernel_accumulator")
    sdfg.add_array("A", [rows, COLS], dace.float64, storage=GLOBAL)
    sdfg.add_array("out", [rows], dace.float64, storage=GLOBAL)
    sdfg.add_transient("acc", [1], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("s")
    kernel_entry, kernel_exit = state.add_map("kernel", dict(i=f"0:{rows}"), schedule=GPU_DEVICE)
    kernel_entry.map.gpu_block_size = [32, 1, 1]
    row_entry, row_exit = state.add_map("row", dict(j=f"0:{COLS}"), schedule=SEQUENTIAL)

    init = state.add_tasklet("init", {}, {"o": None}, "o = 0.0")
    state.add_nedge(kernel_entry, init, dace.Memlet())
    zeroed = state.add_access("acc")
    state.add_edge(init, "o", zeroed, None, dace.Memlet("acc[0]"))

    add = state.add_tasklet("add", {"a": None}, {"o": None}, "o = a")
    state.add_memlet_path(
        state.add_read("A"), kernel_entry, row_entry, add, dst_conn="a", memlet=dace.Memlet("A[i, j]")
    )
    state.add_nedge(zeroed, row_entry, dace.Memlet())
    total = state.add_access("acc")
    state.add_memlet_path(add, row_exit, total, src_conn="o", memlet=dace.Memlet("acc[0]", wcr="lambda x, y: x + y"))

    copy = state.add_tasklet("copy", {"a": None}, {"o": None}, "o = a")
    state.add_edge(total, None, copy, "a", dace.Memlet("acc[0]"))
    state.add_memlet_path(copy, kernel_exit, state.add_write("out"), src_conn="o", memlet=dace.Memlet("out[i]"))
    sdfg.validate()
    return sdfg


def test_a_small_accumulator_is_lifted_not_demoted():
    """One element is within the register limit, yet the WCR keeps it in memory: it is lifted instead."""
    sdfg = kernel_with_accumulator()

    assert lift(sdfg) == 1

    desc = sdfg.arrays["acc"]
    assert desc.storage == GLOBAL and desc.transient and tuple(desc.shape) == (ROWS, 1), desc
    assert None in [sdfg.start_state.entry_node(n) for n in sdfg.start_state.data_nodes() if n.data == "acc"]
    sdfg.validate()


@pytest.mark.gpu
def test_a_contiguous_lifted_accumulator_sums_each_row():
    """Code generation lifts coalesced; this covers the contiguous layout end to end."""
    import cupy  # Only present on GPU runners.

    sdfg = kernel_with_accumulator()
    lift(sdfg, "contiguous")
    A = cupy.arange(ROWS * COLS, dtype=cupy.float64).reshape(ROWS, COLS)
    out = cupy.zeros(ROWS, dtype=cupy.float64)

    sdfg(A=A, out=out)

    assert np.array_equal(cupy.asnumpy(out), cupy.asnumpy(A).sum(axis=1))


@pytest.mark.gpu
def test_a_contiguous_symbolic_scratch_below_a_nested_sdfg_computes_the_right_values():
    """Code generation lifts coalesced; this covers the contiguous layout through nests."""
    import cupy  # Only present on GPU runners.

    nx, nz = 5, 7
    host_a = np.random.default_rng(0).random((nx, nz))
    sdfg = kernel_with_scratch_below_a_nested_sdfg(REGISTER)
    lift(sdfg, "contiguous")
    out = cupy.zeros((nx, nz))

    sdfg(a=cupy.asarray(host_a), out=out, NX=nx, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), host_a[:, ::-1] * 2.0 + 1.0)


def test_a_transient_shared_by_two_kernels_is_left_in_place_by_code_generation():
    """Code generation runs the pass without raising: what it cannot slice keeps its current lowering."""
    sdfg = kernel_with_a_transient_shared_by_two_kernels()
    sut = MoveArrayOutOfKernel()
    sut.raise_on_unsupported = False

    with pytest.warns(UserWarning, match="shared by the kernels"):
        assert sut.apply_pass(sdfg, {}) is None

    assert tuple(sdfg.arrays["buf"].shape) == (1024,)


@pytest.mark.gpu
def test_code_generation_gives_every_thread_its_own_scratch():
    """Unlifted, every thread accumulates its row into one shared ``acc``; many blocks make the race show."""
    import cupy  # Only present on GPU runners.

    rows = 4096
    sdfg = kernel_with_accumulator(rows)
    A = cupy.arange(rows * COLS, dtype=cupy.float64).reshape(rows, COLS)
    out = cupy.zeros(rows, dtype=cupy.float64)

    sdfg(A=A, out=out)

    assert np.array_equal(cupy.asnumpy(out), cupy.asnumpy(A).sum(axis=1))


@pytest.mark.gpu
def test_code_generation_computes_through_a_symbolic_scratch_below_a_nested_sdfg():
    """The code generator lifts the scratch itself and types the connectors the lift adds."""
    import cupy  # Only present on GPU runners.

    nx, nz = 300, 7
    host_a = np.random.default_rng(0).random((nx, nz))
    sdfg = kernel_with_scratch_below_a_nested_sdfg(REGISTER)
    out = cupy.zeros((nx, nz))

    sdfg(a=cupy.asarray(host_a), out=out, NX=nx, NZ=nz)

    assert np.allclose(cupy.asnumpy(out), host_a[:, ::-1] * 2.0 + 1.0)


@pytest.mark.parametrize("schedule", [GPU_DEVICE, dtypes.ScheduleType.GPU_Persistent])
def test_a_symbolic_per_thread_array_is_moved_out_of_any_kernel(schedule):
    sdfg = kernel_with_internal_transient((NX,), REGISTER)
    (kernel,) = [n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.MapEntry)]
    kernel.map.schedule = schedule

    assert lift(sdfg) == 1

    buf = sdfg.arrays["buf"]
    assert buf.storage == GLOBAL and [str(d) for d in buf.shape] == ["128", "NX"], buf
    assert buf_scopes(sdfg).count(None) == 1, "the moved array is allocated outside the kernel"
    inner = [e.data for e in sdfg.start_state.edges() if e.data.data == "buf" and isinstance(e.src, dace.nodes.Tasklet)]
    assert [str(m.subset[0][0]) for m in inner] == ["i"], [str(m) for m in inner]
    sdfg.validate()


def flat_symbolic_scratch() -> dace.SDFG:
    sdfg = kernel_with_internal_transient((NX,), REGISTER)
    next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.MapEntry)).map.gpu_block_size = [32, 1, 1]
    return sdfg


CODEGEN_CASES = {
    "flat": (flat_symbolic_scratch, "buf", "NX"),
    "nested": (lambda: kernel_with_scratch_below_a_nested_sdfg(REGISTER), "tmp", "NZ"),
}


@pytest.mark.parametrize("case", list(CODEGEN_CASES))
def test_code_generation_moves_a_symbolic_scratch_out_of_the_kernel(case):
    """C++ has no variable-length arrays: the CUDA code generator runs the pass itself, so the scratch is one
    device allocation outside the kernel, not a per-thread one."""
    build, name, extent = CODEGEN_CASES[case]

    objects = build().generate_code()
    host = "\n".join(obj.clean_code for obj in objects)
    (device,) = [obj.clean_code for obj in objects if obj.language == "cu"]

    assert not re.search(rf"double\s+{name}\s*\[\s*{extent}", device), f"{name} is declared per thread"
    # A persistent allocation is a state field, ``&__state->__0_buf``
    assert re.search(rf"(cuda|hip)Malloc\(\(void\*\*\)&(__state->__\d+_)?{name}\b", host), (
        f"{name} is not allocated on the device"
    )
    assert "aligned_new_array" not in host
    assert not re.search(rf"Copy\w*\([^;]*\b{name}\b", device), f"the kernel copies {name} onto itself"


def kernel_computing_the_size_of_its_scratch() -> dace.SDFG:
    """The nested body sets ``m = i + 1`` on an interstate edge and sizes its ``Register`` scratch ``buf[m]`` by it, as
    a CSR row's length is read inside the kernel (npbench spmv)."""
    sdfg = kernel_with_transient_behind_a_nested_sdfg()
    inner = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    inner.add_symbol("m", dace.int64)
    body = inner.start_block
    size = inner.add_state_before(body, "size", is_start_block=True)
    inner.edges_between(size, body)[0].data.assignments["m"] = "i + 1"
    inner.arrays["buf"].set_shape(new_shape=[dace.symbol("m", dace.int64)])
    inner.arrays["buf"].storage = REGISTER
    sdfg.validate()
    return sdfg


def test_a_scratch_the_kernel_sizes_stays_where_it_is():
    """An allocation before the launch cannot use a size the kernel computes."""
    sdfg = kernel_computing_the_size_of_its_scratch()

    assert lift(sdfg) is None

    inner = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG)).sdfg
    assert inner.arrays["buf"].storage == REGISTER and [str(d) for d in inner.arrays["buf"].shape] == ["m"]
    sdfg.validate()


def test_a_scratch_sized_by_the_kernel_parameter_stays_where_it_is():
    """``buf[i + 1]`` in kernel ``i`` has no size before the launch."""
    sdfg = kernel_with_internal_transient()
    sdfg.arrays["buf"].set_shape(new_shape=[dace.symbol("i", dace.int64) + 1])
    sdfg.arrays["buf"].storage = REGISTER

    assert lift(sdfg) is None

    assert sdfg.arrays["buf"].storage == REGISTER


@pytest.mark.gpu
def test_a_symbolic_per_thread_array_moved_by_code_generation_computes_the_right_values():
    import cupy  # Only present on GPU runners.

    sdfg = flat_symbolic_scratch()
    out = cupy.zeros(128)

    sdfg(A=out, NX=4)

    assert np.array_equal(cupy.asnumpy(out), np.ones(128))


def test_a_kernel_level_access_above_a_per_thread_scratch_is_refused():
    """A kernel-level write of ``tmp`` would span every thread's slice, yet the tasklet produces one value."""
    sdfg = kernel_with_a_thread_block_around_its_scratch()
    state = sdfg.start_state
    entries = {n.map.schedule: n for n in state.nodes() if isinstance(n, dace.nodes.MapEntry)}
    init = state.add_tasklet("init", {}, {"__out": None}, "__out = 0.0")
    state.add_nedge(entries[GPU_DEVICE], init, dace.Memlet())
    zeroed = state.add_access("tmp")
    state.add_edge(init, "__out", zeroed, None, dace.Memlet("tmp[0]"))
    state.add_nedge(zeroed, entries[dtypes.ScheduleType.GPU_ThreadBlock], dace.Memlet())
    sdfg.validate()

    with pytest.raises(NotImplementedError, match="outside the kernel levels"):
        lift(sdfg)


def kernel_with_a_per_thread_count(indirect: bool = False) -> dace.SDFG:
    """``for b: n = counts[b]; buf = empty(n + 1); buf[n] = 2; out[b] = buf[n]``: the extent of ``buf`` is a
    count each thread reads (lavamd's ``pointers = np.empty(1 + neighbor_counts[l])``). ``indirect`` reads it
    through ``r = rows[b]; n = counts[r]`` (spgemm_hash's ``row_bin[rows_in_bins[b]]``)."""
    body = dace.SDFG("count_body")
    body.add_symbol("b", dace.int64)
    body.add_array("counts", [NX], dace.int32, storage=GLOBAL)
    body.add_array("o", [NX], dace.float64, storage=GLOBAL)
    body.add_transient("buf", ["n + 1"], dace.float64, storage=dtypes.StorageType.Register)
    load = body.add_state("load", is_start_block=True)
    use = body.add_state("use")
    index = "b"
    if indirect:
        body.add_array("rows", [NX], dace.int32, storage=GLOBAL)
        pick = body.add_state("pick")
        body.add_edge(load, pick, dace.InterstateEdge(assignments={"r": "rows[b]"}))
        load, index = pick, "r"
    body.add_edge(load, use, dace.InterstateEdge(assignments={"n": f"counts[{index}]"}))
    buf = use.add_access("buf")
    fill = use.add_tasklet("fill", {}, {"v"}, "v = 2.0")
    use.add_edge(fill, "v", buf, None, dace.Memlet("buf[n]"))
    take = use.add_tasklet("take", {"x"}, {"y"}, "y = x")
    use.add_edge(buf, None, take, "x", dace.Memlet("buf[n]"))
    use.add_edge(take, "y", use.add_write("o"), None, dace.Memlet("o[b]"))

    sdfg = dace.SDFG("per_thread_indirect_count" if indirect else "per_thread_count")
    sdfg.add_array("out", [NX], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("kernel")
    me, mx = state.add_map("kernel", {"b": "0:NX"}, schedule=GPU_DEVICE)
    me.map.gpu_block_size = [32, 1, 1]
    inputs = ("counts", "rows") if indirect else ("counts",)
    nsdfg = state.add_nested_sdfg(body, dict.fromkeys(inputs), {"o": None}, symbol_mapping={"b": "b", "NX": "NX"})
    for name in inputs:
        sdfg.add_array(name, [NX], dace.int32, storage=GLOBAL)
        state.add_memlet_path(state.add_read(name), me, nsdfg, dst_conn=name, memlet=dace.Memlet(f"{name}[0:NX]"))
    state.add_memlet_path(nsdfg, mx, state.add_write("out"), src_conn="o", memlet=dace.Memlet("out[b]"))
    sdfg.validate()
    return sdfg


def test_an_extent_a_thread_reads_is_sized_by_its_maximum_over_the_kernel():
    """The lifted ``buf`` is sized by ``n_max``, the largest ``counts[b]``, folded before the kernel: ``n`` is
    assigned inside the kernel, so no host allocation can name it."""
    sdfg = kernel_with_a_per_thread_count()
    assert lift(sdfg) == 1
    shape_symbols = {
        str(sym) for dim in sdfg.arrays["buf"].shape for sym in dace.symbolic.pystr_to_symbolic(dim).free_symbols
    }
    assert "n" not in shape_symbols and "n_max" in shape_symbols, sdfg.arrays["buf"].shape
    into_kernel = list(sdfg.in_edges(next(s for s in sdfg.states() if s.label == "kernel")))
    assert [e.data.assignments.get("n_max") for e in into_kernel] == ["n_max_host[0]"]
    sdfg.validate()


@pytest.mark.gpu
def test_an_extent_a_thread_reads_runs_on_the_device():
    cp = pytest.importorskip("cupy")
    sdfg = kernel_with_a_per_thread_count()
    counts = cp.asarray(np.array([3, 0, 7, 1, 5], dtype=np.int32))
    out = cp.zeros(5)
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        sdfg(counts=counts, out=out, NX=5)
    assert np.allclose(cp.asnumpy(out), 2.0)


def test_an_extent_a_thread_reads_through_an_index_it_reads_is_folded_through_both_reads():
    """``n = counts[rows[b]]``: the fold before the kernel reads ``counts`` at ``rows[b]``, not at the nest-local ``r``,
    which no host code defines."""
    sdfg = kernel_with_a_per_thread_count(indirect=True)
    assert lift(sdfg) == 1
    folds = [
        node
        for state in sdfg.states()
        for node in state.nodes()
        if isinstance(node, dace.nodes.Tasklet) and node.label == "fold_n_max"
    ]
    assert [node.code.as_string.strip() for node in folds] == ["o = __read_counts[__read_rows[b]]"]
    assert "r" not in sdfg.free_symbols and "n" not in sdfg.free_symbols, sdfg.free_symbols
    sdfg.validate()


@pytest.mark.gpu
def test_an_extent_a_thread_reads_through_an_index_runs_on_the_device():
    cp = pytest.importorskip("cupy")
    sdfg = kernel_with_a_per_thread_count(indirect=True)
    counts = cp.asarray(np.array([3, 0, 7, 1, 5], dtype=np.int32))
    rows = cp.asarray(np.array([4, 2, 0, 3, 1], dtype=np.int32))
    out = cp.zeros(5)
    with dace.config.set_temporary("compiler", "cuda", "implementation", value="experimental"):
        sdfg(counts=counts, rows=rows, out=out, NX=5)
    assert np.allclose(cp.asnumpy(out), 2.0)


def test_a_tasklet_naming_an_array_the_lifted_name_prefixes_is_left_alone():
    """``buf_0`` contains ``buf`` but is another array: a host tasklet reading it after the kernel is no reader of the
    lifted buffer, so it neither gains an index nor stops the lift."""
    sdfg = kernel_with_internal_transient()
    sdfg.add_array("buf_0", [4], dace.float64)
    sdfg.add_array("C", [1], dace.float64)
    after = sdfg.add_state_after(sdfg.start_state, "after")
    peek = after.add_tasklet("peek", {}, {"o": None}, "o = buf_0[1]")
    after.add_edge(peek, "o", after.add_write("C"), None, dace.Memlet("C[0]"))

    assert lift(sdfg) == 1
    assert peek.code.as_string == "o = buf_0[1]", peek.code.as_string
    sdfg.validate()


def kernel_reading_its_scratch_through_a_flat_view() -> dace.SDFG:
    """``A[i] = flat[10]`` with ``buf[1, 2] = i``; ``flat`` views ``buf[4, 8]`` as 32 elements, in kernel ``i``."""
    sdfg = dace.SDFG("scratch_behind_a_flat_view")
    sdfg.add_array("A", [128], dace.float64, storage=GLOBAL)
    sdfg.add_transient("buf", [4, 8], dace.float64, storage=GLOBAL)
    sdfg.add_view("flat", [32], dace.float64, storage=GLOBAL)
    state = sdfg.add_state("s")
    me, mx = state.add_map("kernel", dict(i="0:128"), schedule=GPU_DEVICE)
    buf = state.add_access("buf")
    produce = state.add_tasklet("produce", {}, {"o": None}, "o = i")
    state.add_edge(me, None, produce, None, dace.Memlet())
    state.add_edge(produce, "o", buf, None, dace.Memlet("buf[1, 2]"))
    flat = state.add_access("flat")
    state.add_edge(buf, None, flat, None, dace.Memlet("buf[0:4, 0:8]"))
    consume = state.add_tasklet("consume", {"b": None}, {"o": None}, "o = b")
    state.add_edge(flat, None, consume, "b", dace.Memlet("flat[10]"))
    state.add_memlet_path(consume, mx, state.add_write("A"), src_conn="o", memlet=dace.Memlet("A[i]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_view_of_a_lifted_array_steps_over_the_other_threads_slices(layout):
    """Coalesced, one element of a thread's slice lies 128 apart from the next; the view kept stride 1 and read the
    other threads' elements (lavamd's per-box neighbor offsets)."""
    sdfg = kernel_reading_its_scratch_through_a_flat_view()
    lift(sdfg, layout)

    buf, flat = sdfg.arrays["buf"], sdfg.arrays["flat"]
    view_stride = int(flat.strides[0])
    assert view_stride == int(buf.strides[-1]), (flat.strides, buf.strides)
    # flat[10] is buf[1, 2] of the same thread
    assert 10 * view_stride == int(buf.strides[1] + 2 * buf.strides[2])


def kernel_filling_its_scratch_through_a_whole_container_nest() -> dace.SDFG:
    """``A[i] = buf[1]`` after a nest writes ``buf[1:3]``; the nest's ``buf`` is the whole container, its
    memlets address it as the kernel does. ``buf`` is larger than registers hold."""
    sdfg = dace.SDFG("scratch_filled_by_a_whole_container_nest")
    sdfg.add_array("A", [128], dace.float64, storage=GLOBAL)
    sdfg.add_transient("buf", [LIMIT + 4], dace.float64, storage=GLOBAL)
    body = dace.SDFG("fill")
    body.add_array("buf", [LIMIT + 4], dace.float64, storage=GLOBAL)
    inner = body.add_state("fill", is_start_block=True)
    write = inner.add_access("buf")
    for k in (1, 2):
        tasklet = inner.add_tasklet(f"set_{k}", {}, {"o"}, f"o = {k}.0")
        inner.add_edge(tasklet, "o", write, None, dace.Memlet(f"buf[{k}]"))
    state = sdfg.add_state("s")
    me, mx = state.add_map("kernel", dict(i="0:128"), schedule=GPU_DEVICE)
    nest = state.add_nested_sdfg(body, set(), {"buf"})
    state.add_edge(me, None, nest, None, dace.Memlet())
    buf = state.add_access("buf")
    state.add_edge(nest, "buf", buf, None, dace.Memlet("buf[1:3]"))
    consume = state.add_tasklet("consume", {"b": None}, {"o": None}, "o = b")
    state.add_edge(buf, None, consume, "b", dace.Memlet("buf[1]"))
    state.add_memlet_path(consume, mx, state.add_write("A"), src_conn="o", memlet=dace.Memlet("A[i]"))
    sdfg.validate()
    return sdfg


@pytest.mark.parametrize("layout", LAYOUTS)
def test_a_whole_container_nest_keeps_writing_the_elements_it_wrote(layout):
    """The nest's memlets already address ``buf`` as the kernel does; read as a window starting at ``buf[1]`` they
    were shifted once more and wrote ``buf[2:4]`` (lavamd's neighbor ``pointers`` ran past the end)."""
    sdfg = kernel_filling_its_scratch_through_a_whole_container_nest()
    lift(sdfg, layout)

    nest = next(n for n in sdfg.start_state.nodes() if isinstance(n, dace.nodes.NestedSDFG))
    written = sorted(int(e.data.subset[-1][0]) for st in nest.sdfg.states() for e in st.edges() if e.data.data == "buf")
    assert written == [1, 2], written


if __name__ == "__main__":
    test_tile_extent(sympy.Min(N - 1, B_I + 31), B_I, 32)
    test_tile_extent(N - 1, sympy.Integer(0), N)
    test_lifted_dimensions_are_prepended_keeping_the_own_layout(
        dict(i="0:128", j="0:32"), [64], None, [128, 32, 64], [32, 1, 4096], [2048, 64, 1]
    )
    test_lifted_dimensions_are_prepended_keeping_the_own_layout(
        dict(i="0:8"), [4, 16], [1, 4], [8, 4, 16], [1, 8, 32], [64, 1, 4]
    )
    test_lifted_dimensions_are_prepended_keeping_the_own_layout(dict(i="0:8"), [4, 16], [32, 2], None, None, None)
    test_a_constant_size_per_thread_array_within_the_limit_stays_in_place(REGISTER)
    test_a_constant_size_per_thread_array_within_the_limit_stays_in_place(dtypes.StorageType.Default)
    test_a_constant_size_register_array_over_the_limit_is_moved_out()
    test_the_register_limit_is_configurable()
    test_a_small_global_array_is_demoted_to_registers()
    test_a_small_transient_outliving_the_invocation_is_lifted_not_demoted(dace.AllocationLifetime.Persistent)
    test_a_small_transient_outliving_the_invocation_is_lifted_not_demoted(dace.AllocationLifetime.External)
    test_a_reference_is_not_moved()
    test_a_symbolically_sized_register_array_is_moved_to_global_memory()
    test_a_default_storage_array_is_left_to_storage_inference()
    test_transient_behind_a_nested_sdfg_is_lifted_through_the_boundary()
    test_a_tasklet_body_naming_a_lifted_buffer_gains_the_kernel_index()
    test_a_cpp_tasklet_subscripting_a_lifted_buffer_is_refused()
    for layout in LAYOUTS:
        test_the_lifted_dimension_holds_one_row_per_kernel_iteration(1, 126, 1, 126, layout)
        test_the_lifted_dimension_holds_one_row_per_kernel_iteration(0, 127, 2, 64, layout)
        test_each_thread_owns_one_slice_and_moves_it_out_of_the_kernel(layout)
    test_lift_translates_a_locally_named_shape_symbol_through_symbol_mapping()
    test_lift_gives_descendant_nested_sdfgs_the_lifted_descriptor()
    test_a_renamed_lift_gives_a_descendant_the_lifted_descriptor()
    test_a_lift_binds_the_kernel_parameter_it_indexes_by()
    test_a_nest_looping_over_the_kernel_parameter_name_is_lifted_with_its_loop_renamed()
    test_a_nest_below_a_body_that_never_named_the_kernel_parameter_is_indexed_by_it()
    test_lift_does_not_bind_a_name_the_nest_assigns_itself()
    test_an_interstate_read_of_a_lifted_buffer_gains_the_kernel_index()
    test_a_symbol_mapping_read_of_a_lifted_buffer_gains_the_kernel_index()
    test_control_flow_reading_a_per_thread_buffer_is_refused()
    test_two_nests_defining_the_same_name_are_each_lifted_once()
    test_a_transient_inside_a_loop_of_the_nested_body_is_lifted()
    test_a_transient_shared_by_two_kernels_is_refused()
    test_code_mentioning_the_name_without_subscripting_it_is_returned_verbatim()
    test_a_small_accumulator_is_lifted_not_demoted()
    test_a_contiguous_lifted_accumulator_sums_each_row()
    test_a_contiguous_symbolic_scratch_below_a_nested_sdfg_computes_the_right_values()
    test_a_transient_shared_by_two_kernels_is_left_in_place_by_code_generation()
    test_code_generation_gives_every_thread_its_own_scratch()
    test_code_generation_computes_through_a_symbolic_scratch_below_a_nested_sdfg()
    test_a_symbolic_per_thread_array_is_moved_out_of_any_kernel(GPU_DEVICE)
    test_a_symbolic_per_thread_array_is_moved_out_of_any_kernel(dtypes.ScheduleType.GPU_Persistent)
    test_code_generation_moves_a_symbolic_scratch_out_of_the_kernel("flat")
    test_code_generation_moves_a_symbolic_scratch_out_of_the_kernel("nested")
    test_a_scratch_the_kernel_sizes_stays_where_it_is()
    test_a_scratch_sized_by_the_kernel_parameter_stays_where_it_is()
    test_a_symbolic_per_thread_array_moved_by_code_generation_computes_the_right_values()
    test_a_kernel_level_access_above_a_per_thread_scratch_is_refused()
    test_an_extent_a_thread_reads_is_sized_by_its_maximum_over_the_kernel()
    test_an_extent_a_thread_reads_through_an_index_it_reads_is_folded_through_both_reads()
    test_an_extent_a_thread_reads_runs_on_the_device()
    test_an_extent_a_thread_reads_through_an_index_runs_on_the_device()
