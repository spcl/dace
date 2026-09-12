# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""``dace::float16`` is a bare alias for CUDA's ``__half`` on the device
(``dace/runtime/include/dace/types.h``). ``__half`` declares SEVERAL
simultaneously non-``explicit`` conversions to built-in types (``float``,
``short``, ``unsigned short``, ``int``, ``unsigned int``, ``long long``,
``unsigned long long``, ``bool`` -- all gated by one CUDA macro, never
individually). A bare half value handed to a mixed-type infix operator, or
to an overloaded function with no half overload (``std::sqrt`` and friends),
is an ambiguous-conversion compile error: nvcc cannot pick a single
conversion function on its own.

The ``pure`` expansion of :class:`TileBinop` / :class:`TileFMA` /
:class:`TileUnop` (``dace/libraries/tileops/nodes/*.py``) used to leave a
Tile-kind operand's own dtype uncast, relying on the C++ compiler's own
implicit conversion to resolve a mixed-type expression -- fine for every
ordinary numeric pair, broken for fp16. ``half_disambiguated``
(``_pure_codegen.py``) now routes a bare ``dace::float16`` value through one
explicit, lossless ``(float)`` hop wherever it would otherwise meet a
differently-typed sibling operand or an overloaded math function.

Covered:
  * structural (no GPU needed): the emitted tasklet C++ text carries the
    ``(float)(...)`` hop for a mismatched-dtype ``TileBinop`` and for
    ``TileUnop(op="sqrt")``, and does NOT re-cast operands that never leave
    ``dace::float16`` (native half arithmetic keeps its own precision);
  * GPU compile + bit-exact numeric correctness for the ``sqrt`` case --
    reproduces the exact "more than one instance of overloaded function
    'sqrt' matches" ambiguity pre-fix (confirmed against a standalone
    ``nvcc`` probe compiling the same expression: ``dace::float16``'s
    conversion set is the same defect class as ``__half`` itself) and stays
    bit-exact against a numpy fp32-then-fp16 oracle post-fix;
  * GPU compile + numeric correctness for a mismatched-dtype ``TileBinop``
    (``float16`` Tile + ``float64`` Tile), the same defect class applied to
    a plain infix operator rather than an elemental function call.
"""
import numpy as np
import pytest

import dace
from dace.libraries.tileops import TileBinop, TileUnop
from dace.libraries.tileops._pure_codegen import half_disambiguated


# --------------------------------------------------------------------------------------------------
# The disambiguation rule itself, in isolation.
# --------------------------------------------------------------------------------------------------
def test_half_disambiguated_hops_only_when_leaving_float16():
    """``half_disambiguated`` hops a float16 value meeting a different type,
    and leaves everything else (including float16 meeting float16) alone."""
    assert half_disambiguated("_a[0]", "dace::float16", "double") == "(float)(_a[0])"
    assert half_disambiguated("_a[0]", "dace::float16", "dace::float16") == "_a[0]"
    assert half_disambiguated("_a[0]", "double", "dace::float16") == "_a[0]"
    assert half_disambiguated("_a[0]", "double", "double") == "_a[0]"


# --------------------------------------------------------------------------------------------------
# Structural: the emitted tasklet text, no compiler needed.
# --------------------------------------------------------------------------------------------------
def _pure_tasklet_code(node, edges):
    """Build a one-state SDFG around ``node``, wire ``edges`` (list of
    ``(src_name, src_conn, dst_conn, dtype, shape)`` for inputs and
    ``(dst_conn, dtype, shape)`` for the single output), expand, and return
    the resulting tasklet's C++ source."""
    sdfg = dace.SDFG(f"probe_{node.label}")
    state = sdfg.add_state("main")
    state.add_node(node)
    for name, conn, dtype, shape in edges["inputs"]:
        sdfg.add_array(name, shape, dtype)
        state.add_edge(state.add_access(name), None, node, conn,
                       dace.Memlet(f"{name}[{','.join(f'0:{w}' for w in shape)}]"))
    out_name, out_conn, out_dtype, out_shape = edges["output"]
    sdfg.add_array(out_name, out_shape, out_dtype)
    state.add_edge(node, out_conn, state.add_access(out_name), None,
                   dace.Memlet(f"{out_name}[{','.join(f'0:{w}' for w in out_shape)}]"))
    sdfg.expand_library_nodes()
    sdfg.validate()
    tasklets = [n for st in sdfg.states() for n in st.nodes() if isinstance(n, dace.nodes.Tasklet)]
    assert len(tasklets) == 1, f"expected exactly one tasklet after expansion, got {len(tasklets)}"
    return tasklets[0].code.as_string


def test_binop_mismatched_tile_dtype_hops_the_half_operand():
    """``float16`` Tile + ``float64`` Tile: the float16 side is disambiguated,
    the float64 side is left bare (only half has the ambiguous-conversion defect)."""
    node = TileBinop(name="tb", widths=(8, ), op="+")
    code = _pure_tasklet_code(
        node, {
            "inputs": [("A", "_a", dace.float16, (8, )), ("B", "_b", dace.float64, (8, ))],
            "output": ("C", "_c", dace.float64, (8, )),
        })
    assert "(float)(_a[" in code, f"float16 Tile operand was not disambiguated: {code}"
    assert "(float)(_b[" not in code, f"float64 operand should never be hopped: {code}"


def test_binop_same_fp16_dtype_stays_native():
    """Both operands float16, output float16: no hop anywhere -- the multiply
    stays native ``__half`` arithmetic (this must not silently promote to
    float32 and change the computed value)."""
    node = TileBinop(name="tb_native", widths=(8, ), op="*")
    code = _pure_tasklet_code(
        node, {
            "inputs": [("A", "_a", dace.float16, (8, )), ("B", "_b", dace.float16, (8, ))],
            "output": ("C", "_c", dace.float16, (8, )),
        })
    assert "(float)(" not in code, f"same-dtype fp16*fp16 must not be widened: {code}"


def test_unop_sqrt_hops_fp16_operand_even_when_output_is_also_fp16():
    """``sqrt`` has no ``__half`` overload at all, so the hop is unconditional
    -- even a float16-in/float16-out sqrt must promote through float first."""
    node = TileUnop(name="tu", widths=(8, ), op="sqrt")
    code = _pure_tasklet_code(node, {
        "inputs": [("A", "_a", dace.float16, (8, ))],
        "output": ("C", "_c", dace.float16, (8, )),
    })
    assert "std::sqrt((float)(_a[" in code, f"sqrt operand was not disambiguated: {code}"


def test_unop_neg_does_not_hop():
    """``neg`` lowers to ``__half``'s own native unary ``operator-``, not an
    overloaded ``std::`` function -- it must stay untouched."""
    node = TileUnop(name="tu_neg", widths=(8, ), op="neg")
    code = _pure_tasklet_code(node, {
        "inputs": [("A", "_a", dace.float16, (8, ))],
        "output": ("C", "_c", dace.float16, (8, )),
    })
    assert "(float)(" not in code, f"neg must not be widened (native __half operator-): {code}"


# --------------------------------------------------------------------------------------------------
# GPU: the actual nvcc ambiguity, and the numeric gate.
# --------------------------------------------------------------------------------------------------
def _gpu_map_wrapped(node, in_edges, out_edge):
    """Wrap ``node`` in a trivial ``GPU_Device``-scheduled map with GPU_Global
    transients fed from / drained to host-visible arrays, mirroring how the
    vectorizer actually places a tile-op lib node inside a GPU kernel.

    :param in_edges: ``[(host_name, dev_name, conn, dtype, shape)]``.
    :param out_edge: ``(dev_name, host_name, conn, dtype, shape)``.
    """
    sdfg = dace.SDFG(f"gpu_{node.label}")
    state = sdfg.add_state("main")
    me, mx = state.add_map("outer", {"__k": "0:1"}, schedule=dace.ScheduleType.GPU_Device)
    state.add_node(node)
    state.add_edge(me, None, node, None, dace.Memlet())
    state.add_edge(node, None, mx, None, dace.Memlet())

    for host_name, dev_name, conn, dtype, shape in in_edges:
        full = ",".join(f"0:{w}" for w in shape)
        sdfg.add_array(host_name, shape, dtype)
        sdfg.add_array(dev_name, shape, dtype, storage=dace.StorageType.GPU_Global, transient=True)
        dev = state.add_access(dev_name)
        state.add_nedge(state.add_read(host_name), dev, dace.Memlet(f"{host_name}[{full}]"))
        in_c, out_c = f"IN_{conn}", f"OUT_{conn}"
        state.add_edge(dev, None, me, in_c, dace.Memlet(f"{dev_name}[{full}]"))
        state.add_edge(me, out_c, node, conn, dace.Memlet(f"{dev_name}[{full}]"))
        me.add_in_connector(in_c)
        me.add_out_connector(out_c)

    dev_name, host_name, conn, dtype, shape = out_edge
    full = ",".join(f"0:{w}" for w in shape)
    sdfg.add_array(host_name, shape, dtype)
    sdfg.add_array(dev_name, shape, dtype, storage=dace.StorageType.GPU_Global, transient=True)
    dev = state.add_access(dev_name)
    in_c, out_c = f"IN_{conn}", f"OUT_{conn}"
    state.add_edge(node, conn, mx, in_c, dace.Memlet(f"{dev_name}[{full}]"))
    state.add_edge(mx, out_c, dev, None, dace.Memlet(f"{dev_name}[{full}]"))
    mx.add_in_connector(in_c)
    mx.add_out_connector(out_c)
    state.add_nedge(dev, state.add_write(host_name), dace.Memlet(f"{host_name}[{full}]"))

    sdfg.expand_library_nodes()
    sdfg.validate()
    return sdfg


@pytest.mark.gpu
def test_gpu_sqrt_compiles_and_matches_numpy_oracle():
    """Reproduces bug3's exact nvcc diagnostic pre-fix (verified separately
    against a standalone ``nvcc`` probe on the bare expression) and, with the
    fix, compiles and is bit-exact against ``np.sqrt`` computed in fp32 then
    rounded back to fp16 -- the same promote-compute-demote the hop performs."""
    node = TileUnop(name="tu_gpu", widths=(8, ), op="sqrt")
    sdfg = _gpu_map_wrapped(node,
                            in_edges=[("A_host", "A", "_a", dace.float16, (8, ))],
                            out_edge=("C", "C_host", "_c", dace.float16, (8, )))
    csr = sdfg.compile()
    A = (np.random.default_rng(2).random(8).astype(np.float16) + 0.1).astype(np.float16)
    C = np.zeros(8, dtype=np.float16)
    csr(A_host=A, C_host=C)
    ref = np.sqrt(A.astype(np.float32)).astype(np.float16)
    assert np.array_equal(C.view(np.uint16), ref.view(np.uint16)), (C, ref)


@pytest.mark.gpu
def test_gpu_mismatched_binop_compiles_and_matches_numpy():
    """A ``float16`` Tile mixed directly with a ``float64`` Tile through the
    ``pure`` expansion's raw ``+`` -- the same defect class as the sqrt case,
    now at an infix operator instead of a function call."""
    node = TileBinop(name="tb_gpu", widths=(8, ), op="+")
    sdfg = _gpu_map_wrapped(node,
                            in_edges=[("A_host", "A", "_a", dace.float16, (8, )),
                                      ("B_host", "B", "_b", dace.float64, (8, ))],
                            out_edge=("C", "C_host", "_c", dace.float64, (8, )))
    csr = sdfg.compile()
    A = np.random.default_rng(0).random(8).astype(np.float16)
    B = np.random.default_rng(1).random(8)
    C = np.zeros(8)
    csr(A_host=A, B_host=B, C_host=C)
    ref = A.astype(np.float64) + B
    np.testing.assert_allclose(C, ref, rtol=0, atol=0)
