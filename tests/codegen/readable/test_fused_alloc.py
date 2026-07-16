# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Fused heap-array definition for the experimental "readable" code generator.

The legacy generator declares an allocated array and then assigns the allocation to it, as two
statements landing in two streams::

    double *tmp;
    tmp = new double DACE_ALIGN(64)[N];

The readable generator fuses them into a single definition carrying a restrict qualifier::

    double* __restrict__ tmp = new double DACE_ALIGN(64)[N];

Fusing is only a textual merge of two writes, so it is sound exactly when both land in the SAME
scope. DaCe deliberately separates them (a DECLARATION may be hoisted to an outer scope while the
ALLOCATION stays inner), so the generator falls back to the split form when either
``declare_array`` already declared the pointer in an enclosing scope (``declared``) or the
dispatcher handed out two different streams (the Persistent / External lifetimes, whose pointer
lives in the state struct). Those fallbacks are asserted here too -- a wrong fusion yields an
undeclared identifier or a shadowed redeclaration, neither of which a numerical check would catch.

A COMPILE-TIME-CONSTANT extent also stays split: the emitted element type carries
``DACE_ALIGN(64)``, and with a constant bound a fused declaration names the fixed array type
``double[1]``, which GCC rejects ("alignment of array elements is greater than element size") even
though the identical ``new`` is legal as a bare assignment.

``__restrict__`` is dropped only for a ``may_alias`` descriptor, matching the condition
``Array.as_arg`` already uses for kernel arguments.

On ``const``: a pointee-const definition (``const double* p = new double[N];``) is NOT emitted for
an allocated array, and this is deliberate -- a heap transient is filled at runtime THROUGH that
pointer, so ``const`` would make its own initializing write ill-formed ("assignment of read-only
location"). The generator's const machinery (``MarkConstInit`` -> ``_is_const_scalar`` /
``_is_const_len1_array``) instead handles read-only data by SKIPPING the allocation and fusing the
value into a ``const T x = expr;`` binding at the write site. ``test_const_init_heap_not_pointee_const``
pins that: const-initialized heap data still compiles and stays bit-exact.
"""
import re

import numpy as np

import dace
from dace.transformation.pass_pipeline import Pipeline
from dace.transformation.passes.mark_const_init import MarkConstInit
from tests.codegen.readable.conftest import (EXPERIMENTAL, LEGACY, assert_outputs_equivalent, generated_code,
                                             run_isolated, use_implementation)


def heap_transient_sdfg(name):
    """Symbolic-size CPU_Heap transient: ``tmp = a * 2`` then ``b = tmp + 1`` (two states).

    The symbolic extent ``N`` keeps it off the stack-array path, so it is a genuine
    ``new double[N]`` allocation whose declaration and allocation land in the same scope.
    """
    sdfg = dace.SDFG(name)
    N = dace.symbol('N')
    sdfg.add_array('a', [N], dace.float64)
    sdfg.add_array('b', [N], dace.float64)
    sdfg.add_transient('tmp', [N], dace.float64, storage=dace.StorageType.CPU_Heap)
    s1 = sdfg.add_state('w')
    r, w = s1.add_read('a'), s1.add_write('tmp')
    e, x = s1.add_map('m1', {'i': '0:N'})
    t = s1.add_tasklet('t1', {'inp'}, {'out'}, 'out = inp * 2.0')
    s1.add_memlet_path(r, e, t, dst_conn='inp', memlet=dace.Memlet('a[i]'))
    s1.add_memlet_path(t, x, w, src_conn='out', memlet=dace.Memlet('tmp[i]'))
    s2 = sdfg.add_state_after(s1, 'r')
    r2, w2 = s2.add_read('tmp'), s2.add_write('b')
    e2, x2 = s2.add_map('m2', {'i': '0:N'})
    t2 = s2.add_tasklet('t2', {'inp'}, {'out'}, 'out = inp + 1.0')
    s2.add_memlet_path(r2, e2, t2, dst_conn='inp', memlet=dace.Memlet('tmp[i]'))
    s2.add_memlet_path(t2, x2, w2, src_conn='out', memlet=dace.Memlet('b[i]'))
    sdfg.validate()
    return sdfg


def persistent_transient_sdfg(name):
    """Same dataflow with a Persistent-lifetime transient: the pointer lives in the state struct, so
    the dispatcher routes the declaration and the allocation to DIFFERENT streams (split path)."""
    sdfg = heap_transient_sdfg(name)
    sdfg.arrays['tmp'].lifetime = dace.AllocationLifetime.Persistent
    return sdfg


def may_alias_transient_sdfg(name):
    """Same dataflow with the transient flagged ``may_alias``: fused, but without ``__restrict__``."""
    sdfg = heap_transient_sdfg(name)
    sdfg.arrays['tmp'].may_alias = True
    return sdfg


def const_init_heap_sdfg(name):
    """Single-state ``const_runtime`` CPU_Heap array (``MarkConstInit`` sets ``const_init``):
    ``s = A[0] * 2`` written once, then read by ``B[i] = A[i] + s``.

    ``s`` is a CPU_Heap Array, so neither ``_is_const_scalar`` (Scalar only) nor
    ``_is_const_len1_array`` (Register only) claims it: it reaches the heap allocation path while
    being const-initialized, which is exactly the case a naive pointee-``const`` would miscompile.
    """
    N = dace.symbol('N')
    sdfg = dace.SDFG(name)
    sdfg.add_array('A', [N], dace.float64)
    sdfg.add_array('B', [N], dace.float64)
    sdfg.add_transient('s', [1], dace.float64, storage=dace.StorageType.CPU_Heap)
    st = sdfg.add_state('main')
    ra0 = st.add_access('A')
    ts = st.add_tasklet('setc', {'a'}, {'o'}, 'o = a * 2.0')
    acc_s = st.add_access('s')
    st.add_edge(ra0, None, ts, 'a', dace.Memlet('A[0]'))
    st.add_edge(ts, 'o', acc_s, None, dace.Memlet('s[0]'))
    ra, wb = st.add_access('A'), st.add_access('B')
    me, mx = st.add_map('m', dict(i='0:N'))
    t2 = st.add_tasklet('add', {'a', 'sc'}, {'o'}, 'o = a + sc')
    st.add_memlet_path(ra, me, t2, dst_conn='a', memlet=dace.Memlet('A[i]'))
    st.add_memlet_path(acc_s, me, t2, dst_conn='sc', memlet=dace.Memlet('s[0]'))
    st.add_memlet_path(t2, mx, wb, src_conn='o', memlet=dace.Memlet('B[i]'))
    sdfg.validate()
    return sdfg


def code_for(build, name, implementation):
    """Generated C++ for ``build(name)`` under ``implementation`` (codegen only, no compile)."""
    with use_implementation(implementation):
        return generated_code(build(name))


#: The fused definition: ``<type>* __restrict__ <name> = new <type> DACE_ALIGN(64)[<count>];``
FUSED = re.compile(r'double\*\s+__restrict__\s+tmp\s*=\s*new\s+double\s+DACE_ALIGN\(64\)\[')
#: The legacy split pair.
SPLIT_DECL = re.compile(r'double\s*\*\s*tmp\s*;')
SPLIT_ALLOC = re.compile(r'(?<![\w>])tmp\s*=\s*new\s+double\s+DACE_ALIGN\(64\)\[')


def test_fused_definition_with_restrict(require_experimental):
    """The readable generator emits ONE fused definition carrying ``__restrict__``, not the split pair."""
    code = code_for(heap_transient_sdfg, 'fused_readable', EXPERIMENTAL)
    assert FUSED.search(code), f'expected a fused restrict-qualified definition, got:\n{code}'
    assert not SPLIT_DECL.search(code), f'declaration was not fused away:\n{code}'
    # Exactly one statement mentions the allocation (the fused definition), i.e. no stray assignment.
    assert len(SPLIT_ALLOC.findall(code)) == 1


def test_legacy_still_splits(require_experimental):
    """The legacy generator keeps the split declaration + assignment (its output must not change)."""
    code = code_for(heap_transient_sdfg, 'fused_legacy', LEGACY)
    assert SPLIT_DECL.search(code), f'legacy declaration disappeared:\n{code}'
    assert SPLIT_ALLOC.search(code), f'legacy allocation disappeared:\n{code}'
    assert not FUSED.search(code), f'legacy must not emit the fused/restrict form:\n{code}'
    assert '__restrict__ tmp' not in code


def test_persistent_lifetime_stays_split(require_experimental):
    """A Persistent transient's pointer lives in the state struct: the dispatcher hands the
    declaration and the allocation two different streams, so it must NOT be fused into a local
    definition (which would shadow the member and leave it unallocated)."""
    code = code_for(persistent_transient_sdfg, 'fused_persistent', EXPERIMENTAL)
    assert re.search(r'__state->[\w]*tmp\s*=\s*new\s+double\s+DACE_ALIGN\(64\)\[', code), \
        f'expected the state-struct member to keep the split assignment:\n{code}'
    assert not FUSED.search(code), f'a state-struct member must not be fused into a local definition:\n{code}'


def test_may_alias_drops_restrict(require_experimental):
    """``may_alias`` marks data deliberately reachable through another pointer: still fused, but the
    no-alias promise must not be made (mirrors ``Array.as_arg``)."""
    code = code_for(may_alias_transient_sdfg, 'fused_may_alias', EXPERIMENTAL)
    assert re.search(r'double\*\s+tmp\s*=\s*new\s+double\s+DACE_ALIGN\(64\)\[', code), \
        f'expected a fused definition:\n{code}'
    assert '__restrict__ tmp' not in code, f'restrict must be dropped for a may_alias array:\n{code}'


def test_constant_extent_stays_split(require_experimental):
    """A COMPILE-TIME-CONSTANT extent keeps the split form.

    ``heap_alloc_stmt`` emits the element type carrying ``DACE_ALIGN(64)``, which makes ``new`` call
    the over-aligned ``operator new[]``. With a constant bound, a fused DECLARATION names the fixed
    array type ``double[1]`` and GCC rejects it ("alignment of array elements is greater than element
    size"); the same ``new`` is legal as a bare assignment. So this must stay split rather than
    de-align the allocation. ``const_init_heap_sdfg`` allocates ``s`` with a constant extent of 1.
    """
    code = code_for(const_init_heap_sdfg, 'fused_const_extent', EXPERIMENTAL)
    assert re.search(r'double\s*\*\s*s\s*;', code), f'expected the split declaration for a constant extent:\n{code}'
    assert not re.search(r'double\*\s+__restrict__\s+s\s*=\s*new', code), \
        f'a constant-extent heap array must not be fused (GCC rejects it):\n{code}'
    # The alignment attribute must survive on the split path.
    assert re.search(r's\s*=\s*new\s+double\s+DACE_ALIGN\(64\)\[', code), f'DACE_ALIGN was dropped:\n{code}'


def test_const_init_data_is_not_pointee_const(require_experimental):
    """Const-initialized data reaching the allocator is never emitted as pointee-``const``.

    ``MarkConstInit`` flags ``s`` (written once, then read-only), yet its initializing write is
    emitted THROUGH the pointer, so ``const double* s = new double[1];`` would not compile
    ("assignment of read-only location"). The generator's const machinery instead handles read-only
    data by skipping the allocation entirely and fusing the value into a ``const T x = expr;``
    binding at the write site. Guard against naively adding pointee-const at the allocation.
    """
    # Precondition: the pass really does classify this data as write-once/read-only, so the test
    # cannot pass vacuously if MarkConstInit ever stops marking it.
    marked = const_init_heap_sdfg('fused_const_init_desc')
    Pipeline([MarkConstInit()]).apply_pass(marked, {})
    assert marked.arrays['s'].const_init, 'expected MarkConstInit to flag s as const-initialized'

    code = code_for(const_init_heap_sdfg, 'fused_const_init_code', EXPERIMENTAL)
    assert not re.search(r'const\s+double\s*\*[^=]*=\s*new', code), \
        f'pointee-const would make the array\'s own initializing write ill-formed:\n{code}'
    # The write through the pointer is still emitted -- precisely why const is unsound here.
    assert re.search(r's\[[^\]]*\]\s*=', code), f'expected a write through the pointer:\n{code}'


def run(build, name, implementation):
    """Compile + run ``build(name)`` under ``implementation`` in a forked child; return the outputs."""

    def build_and_run():
        with use_implementation(implementation):
            sdfg = build(name)
            rng = np.random.default_rng(42)
            a = rng.random(64)
            b = np.zeros(64)
            sdfg(a=a, b=b, N=64)
            return {'b': b}

    return run_isolated(build_and_run)


def test_fused_definition_bit_exact(require_experimental):
    """The fused definition compiles and reproduces the legacy result bit-exactly."""
    legacy = run(heap_transient_sdfg, 'fused_run_legacy', LEGACY)
    experimental = run(heap_transient_sdfg, 'fused_run_readable', EXPERIMENTAL)
    assert_outputs_equivalent(legacy, experimental, 'cpu', label='fused_alloc')
    np.testing.assert_array_equal(experimental['b'], np.random.default_rng(42).random(64) * 2.0 + 1.0)


def run_const_init(name, implementation):
    """Compile + run the const-initialized heap SDFG; returns ``{'B': ...}``."""

    def build_and_run():
        with use_implementation(implementation):
            sdfg = const_init_heap_sdfg(name)
            rng = np.random.default_rng(7)
            A = rng.random(64)
            B = np.zeros(64)
            sdfg(A=A, B=B, N=64)
            return {'B': B}

    return run_isolated(build_and_run)


def test_const_init_heap_bit_exact(require_experimental):
    """Const-initialized heap data compiles under the readable generator and matches legacy."""
    legacy = run_const_init('const_run_legacy', LEGACY)
    experimental = run_const_init('const_run_readable', EXPERIMENTAL)
    assert_outputs_equivalent(legacy, experimental, 'cpu', label='const_init_heap')
