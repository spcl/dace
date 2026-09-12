# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The parallel FORM canonicalization reaches on the focus-40 kernels, pinned one kernel at a time.

A speedup table cannot say why a kernel is fast, and a value comparison cannot say whether anything
was parallelized at all -- a kernel left as one sequential loop reproduces its reference perfectly.
This file pins the third thing: the shape ``canonicalize(target='cpu')`` plus
:func:`~dace.transformation.passes.canonicalize.finalize.finalize_for_target` leaves behind. How many
maps carry the work, how many of them are ``CPU_Multicore``, how many sequential ``LoopRegion`` s
survive, which library nodes the recurrences lifted into, and whether those nodes lowered to their
PARALLEL or their SERIAL runtime entry point.

That last question is the one a pragma count answers wrongly. A library node's parallel form is a
CALL into a DaCe runtime header, not a pragma in the generated ``.cpp``: ``dace::reduce::sum``
carries its own ``#pragma omp parallel for reduction`` inside ``dace/runtime/include/dace/reduction.h``,
``dace::scan::inclusive_sum`` inside ``dace/runtime/include/dace/scan.hpp``, and
``dace::find_first_index`` inside ``dace/runtime/include/dace/detect.h``. Counting pragmas in the
emitted ``.cpp`` reads every one of those kernels as zero OpenMP while they are fully parallel. The
runtime's ``seq::`` namespace is what actually separates the two -- ``dace::reduce::seq::sum`` is the
serial entry point, ``dace::reduce::sum`` the parallel one -- so :attr:`Form.libcall` is asserted on
that namespace, over the POST-EXPANSION source read out of the build folder. ``generate_code`` runs
before expansion has picked an implementation and cannot answer the question at all.

Every kernel is asserted twice, as separate cases:

* **numerics** -- unconditionally, against the corpus's own numpy oracle. Canonicalization is
  value-preserving whatever it decides about parallelism, and wrongly parallelizing a recurrence is a
  far worse outcome than leaving one sequential.
* **form** -- against :data:`EXPECTED`, which states what each kernel's canonical shape SHOULD be and
  why. Where the recipe does not reach that shape the case is ``xfail(strict=True)`` naming the
  refusing matcher, so the day it does reach it the suite FAILS and this table has to be corrected
  rather than the new capability landing unnoticed. Kernels that are correctly sequential -- a real
  loop-carried dependence no analysis should ever break -- carry plain assertions and are
  deliberately NOT xfail, the same contract as
  ``smt_required_parallel_test.test_colliding_scatter_stays_sequential``.

Both corpora are this repo's own: the TSVC ``_d_single`` kernels in :mod:`tests.corpus.tsvc` and the
extension kernels in :mod:`tests.corpus.tsvc_2_5`, each with its numpy oracle alongside. Table keys
are corpus kernel names, and the ``_d_single`` suffix is what routes a kernel to its corpus.
"""
import dataclasses
import functools
import inspect
import os
import pathlib
import re

os.environ.setdefault("OMPI_MCA_pml", "ob1")
os.environ.setdefault("OMPI_MCA_btl", "self,vader")
os.environ.setdefault("UCX_VFS_ENABLE", "n")

import numpy as np
import pytest

import dace
from dace.codegen.compiled_sdfg import CompiledSDFG
from dace.frontend.python.parser import DaceProgram
from dace.sdfg import nodes as nd
from dace.sdfg.state import ConditionalBlock, LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target
from dace.transformation.passes.canonicalize.pipeline import canonicalize

from tests.corpus.tsvc import tsvc
from tests.corpus.tsvc.tsvc_numpy import REFERENCES
from tests.corpus.tsvc_2_5 import tsvc_2_5, tsvc_2_5_numpy

#: The recipe settings the rest of this directory canonicalizes the corpora at.
PEEL_LIMIT = 4
UNROLL_LIMIT = 4
TOL = 1e-9

#: Kernel-name suffix that routes a table key to the TSVC corpus rather than to TSVC-2.5.
TSVC_SUFFIX = "_d_single"

#: A runtime call OUTSIDE the ``seq::`` namespace is the parallel entry point, inside it the serial
#: one. ``find_first_index`` has no serial sibling to name, so it only ever matches parallel.
PARALLEL_LIBCALL = re.compile(r"dace::(?:reduce|scan)::(?!seq::)\w+|dace::find_first_index")
SERIAL_LIBCALL = re.compile(r"dace::(?:reduce|scan)::seq::\w+")


@dataclasses.dataclass(frozen=True, slots=True)
class Form:
    """The canonical shape one kernel must reach, and what makes that shape the correct one."""

    parallel_maps: int  #: maps scheduled ``CPU_Multicore``
    sequential_maps: int  #: maps scheduled ``Sequential`` -- an inner map under a parallel one
    sequential_loops: int  #: ``LoopRegion`` s canonicalization did not turn into parallel work
    libnodes: tuple[str, ...]  #: ``Type=implementation`` per library node, sorted
    libcall: str | None  #: ``'parallel'`` / ``'serial'``, or ``None`` when no runtime call is emitted
    why: str = ""  #: the dependence that makes this the right answer -- prose, not part of the compare


def shape(form: Form) -> tuple:
    """The comparable part of a :class:`Form` -- everything but the prose."""
    return (form.parallel_maps, form.sequential_maps, form.sequential_loops, form.libnodes, form.libcall)


#: What each kernel's canonicalized form SHOULD be. Entries are claims about the kernel's dependence
#: structure; the ones today's recipe does not reach are listed in :data:`XFAIL`.
EXPECTED: dict[str, Form] = {
    "s115_d_single":
    Form(
        1, 0, 1, (), None,
        "the outer j sweep carries a[j] into every later row (a[i] -= aa[j, i] * a[j]), so j is a real "
        "recurrence and must stay a loop; within one j the i updates touch disjoint elements"),
    "s119_d_single":
    Form(
        1, 0, 1, (), None, "aa[i, j] = aa[i - 1, j - 1] + bb[i, j] carries at distance (1, 1), so the i sweep stays "
        "sequential while a whole row of j is free"),
    "s1232_d_single":
    Form(
        1, 1, 0, (), None, "aa[i, j] = bb[i, j] + cc[i, j] over the i >= j * VLEN triangle carries nothing at all; the "
        "outer band takes the threads and the inner extent rides along as a Sequential map"),
    "s1244_d_single":
    Form(
        2, 2, 1, (), None,
        "d[i] = a[i] + a[i + 1] reads the a its own loop writes one AHEAD -- an anti dependence, not a "
        "recurrence -- so snapshotting a lets both statements share one parallel map. On the CPU that "
        "snapshot is the per-chunk seam rather than a copy of the whole array, which is the same "
        "residual loop plus two Sequential maps ext_war_unit shows"),
    "s13110_d_single":
    Form(
        0, 0, 0, ("ArgReduce=OpenMP", ), None,
        "a 2D running maximum carrying the value and BOTH its indices is an argmax, not a recurrence; "
        "the OpenMP ArgReduce expansion opens its own parallel region rather than calling the runtime"),
    "s152_d_single":
    Form(1, 0, 0, (), None,
         "two independent elementwise loops (b from d * e, then a from b * c) that fuse into one map"),
    "s2275_d_single":
    Form(
        2, 0, 0, (), None,
        "the inner 2D update and the per-row 1D statement are both elementwise, so distribution leaves "
        "two parallel maps and nothing carried"),
    "s231_d_single":
    Form(
        1, 1, 1, (), None, "aa[j, i] = aa[j - 1, i] + bb[j, i] carries DOWN a column, so the j sweep is a genuine "
        "recurrence; the columns i are independent and are what the threads split -- cut into per-thread "
        "bands, so the band the carry is given to rides along as a Sequential map (one barrier for "
        "the nest instead of one per trip)"),
    "s232_d_single":
    Form(
        1, 0, 1, (), None,
        "aa[j, i] = aa[j, i - 1] ** 2 + bb[j, i] carries ALONG a row, so the inner i stays sequential "
        "and the rows j are free -- the mirror image of s231"),
    "s233_d_single":
    Form(
        1, 1, 1, (), None, "two inner nests, one carried down columns (aa[j - 1, i]) and one carried across rows "
        "(bb[j, i - 1]); the dimension free in both parallelizes and one recurrence keeps its loop, and "
        "the band the carry is given to rides along as a Sequential map (one barrier for the nest "
        "instead of one per trip)"),
    "s235_d_single":
    Form(
        2, 1, 1, (), None,
        "a[i] += b[i] * c[i] is elementwise while aa[j, i] = aa[j - 1, i] + bb[j, i] * a[i] carries down "
        "j; distribution frees the a statement and the column recurrence keeps its loop, and the band "
        "the carry is given to rides along as a Sequential map (one barrier for the nest instead of "
        "one per trip)"),
    "s252_d_single":
    Form(
        1, 0, 0, (), None,
        "the carried scalar t only ever holds b[i - 1] * c[i - 1], so re-evaluating its producer one "
        "iteration back removes the carry and a[i] = b[i] * c[i] + b[i - 1] * c[i - 1] is a plain map"),
    "s255_d_single":
    Form(
        1, 0, 0, (), None,
        "x and y are a two-deep rotation of b, so rematerializing them from b[i - 1] and b[i - 2] turns "
        "the loop into a 3-point stencil with a peeled prologue"),
    "s275_d_single":
    Form(
        1, 1, 1, (), None, "the guard aa[0, i] > 0 is invariant within a column, so the outer i is free; the inner "
        "aa[j, i] = aa[j - 1, i] + ... is a real column recurrence and must stay a loop. A band owns whole "
        "columns, so BandCarriedLoops takes i for the threads and the narrowed column map rides inside the "
        "carry as the Sequential one -- s231's accepted shape behind a per-column guard, and the same band "
        "plus Sequential inner s1232 pins above. Measured, not argued: the banded build is bit-identical to "
        "the uncanonicalized one over 8 seeds at LEN_2D=1021, and to itself at 1, 2, 3, 5, 7 and 16 threads"),
    "s2710_d_single":
    Form(
        1, 0, 0, (), None,
        "nothing is carried -- only nested data- and config-dependent branches -- so the whole loop is "
        "one predicated parallel map"),
    "s311_d_single":
    Form(
        0, 0, 0, ("Reduce=OpenMP", ), "parallel",
        "a bare sum accumulation is a reduction and must become one Reduce lowered to dace::reduce::sum, "
        "whose OpenMP reduction lives in reduction.h and never appears in the emitted .cpp"),
    "s3110_d_single":
    Form(0, 0, 0, ("ArgReduce=OpenMP", ), None,
         "the same value-plus-index argmax as s13110, reached through a different chksum tail"),
    "s3111_d_single":
    Form(
        1, 0, 0, (), None,
        "a sum taken only where a[i] > 0 is a PREDICATED reduction: the guard belongs inside the map "
        "body with a conflict-resolved accumulator, not in a Reduce library node"),
    "s3112_d_single":
    Form(
        0, 0, 0, ("Scan=CPU", ), "parallel",
        "sum = sum + a[i]; b[i] = sum is the textbook inclusive prefix scan, so the canonical answer is "
        "one Scan lowered to dace::scan::inclusive_sum -- parallel inside scan.hpp, invisible in the .cpp"),
    "s316_d_single":
    Form(0, 0, 0, ("Reduce=OpenMP", ), "parallel",
         "a running minimum is a min reduction and must reach dace::reduce::min, not stay a compare chain"),
    "s318_d_single":
    Form(
        0, 0, 0, ("ArgReduce=OpenMP", ), None,
        "the strided |a[k]| search is an argmax over a gathered vector, and the gather needs no map: the "
        "ArgReduce reads a at stride inc through its own _in memlet and applies abs per element, so the "
        "LEN_1D-element staging buffer the map used to fill is never allocated"),
    "s319_d_single":
    Form(
        1, 0, 0, (), None,
        "a and b are elementwise and sum_val accumulates over what they just wrote; the accumulation "
        "rides the map as a write-conflict resolution, so nothing is left sequential. ONE map, not two: "
        "the two stores share an index space and depend on nothing of each other's, and once the "
        "accumulator reads the stored values instead of reloading a and b (ForwardStoreToLoad) the "
        "second map has no reason to exist -- c, d and e are streamed once"),
    "s323_d_single":
    Form(
        1, 0, 0, ("Scan=CPU", ), "parallel",
        "b[i] = a[i] + c[i] * e[i] with a[i] = b[i - 1] + c[i] * d[i] is b[i] = b[i - 1] + "
        "c[i] * (d[i] + e[i]) -- a first-order linear recurrence, i.e. an inclusive prefix sum -- after "
        "which a[i] = b[i - 1] + c[i] * d[i] is a plain map"),
    "s4112_d_single":
    Form(
        1, 0, 0, (), None,
        "a[i] += b[ip[i]] * 2 gathers b through ip but stores to a[i], so the writes are injective and "
        "the indirection blocks nothing"),
    "vag_d_single":
    Form(1, 0, 0, (), None, "a[i] = b[ip[i]] is a pure gather -- one read per output, no conflict"),
    "vpvts_d_single":
    Form(1, 0, 0, (), None, "a[i] += b[i] * S is elementwise with a symbolic scalar; S changes nothing"),
    "vtvtv_d_single":
    Form(1, 0, 0, (), None, "a[i] *= b[i] * c[i] is elementwise"),
    "ext_break_capture":
    Form(
        0, 0, 0, ("FindFirst=OpenMP", ), "parallel",
        "the loop has NO per-iteration body independent of the search, so a map here would be a "
        "regression, not progress: the whole loop is one FindFirst lowered to dace::find_first_index"),
    "ext_break_find_first":
    Form(
        1, 0, 0, ("FindFirst=OpenMP", ), "parallel",
        "TSVC s481 -- the break bound is data-dependent on d, so a FindFirst computes it and the body "
        "then runs as a map clipped to that bound"),
    "ext_break_post_body":
    Form(
        1, 0, 0, ("FindFirst=OpenMP", ), "parallel",
        "TSVC s482 -- the breaking iteration's write is retained, so the same FindFirst bound clips the "
        "body map INCLUSIVELY rather than exclusively"),
    "ext_war_unit":
    Form(
        2, 2, 1, (), None, "a[i] = a[i + 1] + b[i] is a pure anti dependence, which lifts once a is snapshotted; the "
        "residual loop and the two Sequential maps are that snapshot's chunked copy, not the kernel"),
    "fuse_diamond":
    Form(
        1, 0, 0, (), None,
        "one producer feeding two consumers that rejoin must collapse to a SINGLE map -- fusing it as a "
        "chain would either duplicate the producer or serialize the two consumers"),
    "fuse_move_ifs":
    Form(
        1, 0, 0, (), None,
        "sinking both guards to the innermost position gives the two nests one iteration space; what "
        "survives is predicated parallel work, not a sequential guard around it. ONE map, not two plus "
        "a Sequential inner: once cond[i] > 0 and K > 0 are both predicates rather than nests, the "
        "a-nest and the b-nest have the same (i, j) domain and fuse, so there is no second parallel "
        "region left to enter and nothing for the nested-parallelism rule to sequentialize. "
        "test_fuse_move_ifs_fuses_both_guarded_nests_into_one_predicated_map pins that shape by name"),
    "fuse_stencil_through_transient":
    Form(
        1, 0, 0, (), None,
        "the consumer reads tmp[i + 1], so vertical fusion has to widen the producer window before it "
        "can merge; done right the two maps become one and tmp disappears"),
    "quasi_affine_reduce_odd":
    Form(
        0, 0, 0, ("Reduce=OpenMP", ), "parallel",
        "a stride-2 sum from a non-zero base is still a sum reduction once the range is canonicalized; "
        "the odd base is the only thing standing between it and dace::reduce::sum"),
    "argmax_with_index":
    Form(
        0, 0, 0, ("ArgReduce=OpenMP", ), None,
        "a running maximum carrying both the value and its index is the two-accumulator ArgMaxLift "
        "shape, and must land as one ArgReduce rather than as a conditional recurrence"),
    "wavefront2d":
    Form(
        1, 0, 3, (), None, "dependence vectors (0, 1), (1, 0) and (1, 1) serialize both original loops; only the i + j "
        "anti-diagonal is free, so the skewed form sweeps diagonals sequentially and parallelizes across "
        "each front"),
    "wf_diff_skew":
    Form(
        1, 0, 1, (), None, "deps (1, 0) and (1, -1) make the legal front the i - j DIFFERENCE diagonal, not the "
        "anti-diagonal a north+west kernel would skew to"),
    "wf_north_west":
    Form(1, 0, 3, (), None, "north (1, 0) plus west (0, 1) deps leave the i + j anti-diagonal as the only free front"),
    "wf_triangular":
    Form(
        1, 0, 3, (), None,
        "the same anti-diagonal front as wf_north_west, clipped to the j >= i triangle -- the skew has to "
        "honour the triangular iteration space, not just the dependence vectors"),
}

#: Kernels whose :data:`EXPECTED` entry is the right answer but not today's answer, mapped to the
#: matcher that refuses them. ``strict=True``: a fix XPASSes and forces this table to be corrected.
XFAIL: dict[str, str] = {
    "s323_d_single":
    ("LoopToScan refuses it twice over: its v1 matcher takes the per-iteration delta as a clean array "
     "slice and declines a computed one (here c[i] * (d[i] + e[i]), reached through a), and it refuses "
     "any body with further carried writes to non-transient arrays (here a, read back at i - 1). "
     "Canonicalization leaves one sequential LoopRegion and no parallel work anywhere."),
}

#: The three break kernels share one reason, so it is written once.
BREAK_WITHDRAWN = ("no pass lifts an early exit any more. The lift that did matched on the break's SHAPE and "
                   "never costed the rewrite: it put a whole FindFirst pass over the predicate arrays in front "
                   "of the body map whether or not the body read them again, which on s482 is three extra "
                   "streams for a form measuring 0.70x of the sequential loop. Canonicalization leaves one "
                   "sequential LoopRegion; a lift that returns owes a cost model, and correcting this table is "
                   "how it announces itself.")
XFAIL.update({name: BREAK_WITHDRAWN for name in ("ext_break_capture", "ext_break_find_first", "ext_break_post_body")})


def ext_program(name: str) -> DaceProgram:
    """The TSVC-2.5 corpus kernel called ``name``."""
    return next(p for p in tsvc_2_5.collect() if p.f.__name__ == name)


def canonical_sdfg(name: str) -> dace.SDFG:
    """A FRESH SDFG for ``name``, canonicalized and finalized for the CPU.

    Built per kernel and never reused across kernels: canonicalize mutates in place and dace's pass
    state is process-global, so a graph another case has already been through is not the kernel this
    table describes.
    """
    if name.endswith(TSVC_SUFFIX):
        kernel = tsvc.collect(name=name)[0]
        sdfg = tsvc.to_sdfg(kernel, "focus40", simplify=True)
        canonicalize(sdfg, target="cpu", validate=True, peel_limit=PEEL_LIMIT)
    else:
        sdfg = ext_program(name).to_sdfg(simplify=True)
        canonicalize(sdfg, target="cpu", validate=True, peel_limit=PEEL_LIMIT, unroll_limit=UNROLL_LIMIT)
        # A hoisted config guard (fuse_move_ifs' K) can stay free but unregistered.
        for symbol in sorted({str(s) for s in sdfg.free_symbols} - set(sdfg.symbols)):
            sdfg.add_symbol(symbol, dace.int64)
    finalize_for_target(sdfg, "cpu")
    return sdfg


def emitted_source(sdfg: dace.SDFG) -> str:
    """The post-EXPANSION C++, read from the build folder rather than from ``generate_code``."""
    cpu = pathlib.Path(sdfg.build_folder) / "src" / "cpu"
    return "\n".join(path.read_text() for path in sorted(cpu.glob("*.cpp")))


def measure(sdfg: dace.SDFG, source: str) -> Form:
    """Read the canonical form off a finalized ``sdfg`` and its post-expansion ``source``."""
    schedules = [str(n.map.schedule) for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]
    libnodes = sorted({
        f"{type(n).__name__}={n.implementation}"
        for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.LibraryNode)
    })
    loops = sum(1 for sd in sdfg.all_sdfgs_recursive() for cfr in sd.all_control_flow_regions()
                if isinstance(cfr, LoopRegion) and cfr.loop_variable)
    serial = SERIAL_LIBCALL.search(source)
    parallel = PARALLEL_LIBCALL.search(source)
    return Form(parallel_maps=sum(1 for s in schedules if s.endswith("CPU_Multicore")),
                sequential_maps=sum(1 for s in schedules if s.endswith("Sequential")),
                sequential_loops=loops,
                libnodes=tuple(libnodes),
                libcall="serial" if serial else ("parallel" if parallel else None))


@dataclasses.dataclass(frozen=True, slots=True)
class Built:
    """One kernel's canonicalized form plus the compiled object that reproduces its values."""

    form: Form
    csdfg: CompiledSDFG
    symbols: tuple[str, ...]  #: free symbol names, read BEFORE compiling


@functools.lru_cache(maxsize=None, typed=True)
def built(name: str) -> Built:
    """Canonicalize, measure, compile -- once per kernel, shared by that kernel's two cases.

    The form is read off the SDFG BEFORE ``compile``: compiling is one-way, and the graph a
    ``CompiledSDFG`` still points at is not the object to inspect afterwards. Rebuilding for the
    numeric case would only pay compiler time; it would not change what either case asserts.
    """
    sdfg = canonical_sdfg(name)
    symbols = tuple(sorted(str(s) for s in sdfg.free_symbols))
    csdfg = sdfg.compile()
    return Built(form=measure(sdfg, emitted_source(sdfg)), csdfg=csdfg, symbols=symbols)


def reference_and_inputs(name: str, symbols: tuple[str, ...]) -> tuple[dict, dict, dict]:
    """``(arrays, call_kwargs, reference)`` for one kernel, from its own corpus's numpy oracle."""
    if name.endswith(TSVC_SUFFIX):
        kernel = tsvc.collect(name=name)[0]
        arrays, call_kwargs = tsvc.make_inputs(kernel)
        reference = {n: a.copy() for n, a in arrays.items()}
        REFERENCES[name](**reference, **call_kwargs)
        return arrays, call_kwargs, reference

    program = ext_program(name)
    arrays, scalars = tsvc_2_5.make_inputs(program)
    base = name[4:] if name.startswith("ext_") else name
    oracle = vars(tsvc_2_5_numpy)["ref_" + base]
    pool = {
        **{
            n: a.copy()
            for n, a in arrays.items()
        },
        **scalars,
        **{
            s.lower(): v
            for s, v in tsvc_2_5.SIZES.items()
        },
        "n": tsvc_2_5.SIZES["LEN_1D"],
    }
    oracle(**{p: pool[p] for p in inspect.signature(oracle).parameters})
    bindings = {s: v for s, v in tsvc_2_5.SIZES.items() if s in symbols}
    return arrays, {**scalars, **bindings}, {n: pool[n] for n in arrays}


def form_cases() -> list:
    """One param per kernel, xfailed where :data:`XFAIL` names the matcher that still refuses it."""
    return [
        pytest.param(name, marks=pytest.mark.xfail(strict=True, reason=XFAIL[name])) if name in XFAIL else name
        for name in EXPECTED
    ]


@pytest.mark.parametrize("name", form_cases())
def test_canonical_parallel_form(name: str):
    """The kernel reaches exactly the map / loop / library-node shape :data:`EXPECTED` claims for it."""
    want = EXPECTED[name]
    got = built(name).form
    assert shape(got) == shape(want), f"{name}: {want.why}\n  expected {shape(want)}\n  got      {shape(got)}"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_matches_numpy_reference(name: str):
    """The canonicalized kernel reproduces its numpy oracle -- asserted for every kernel, xfail or not.

    A structural check alone passes on a fast wrong answer, so parallelism is never pinned without
    this one beside it; and the kernels this file expects to stay sequential need it most, because
    the failure it guards against is a pass parallelizing them unsoundly.
    """
    case = built(name)
    arrays, call_kwargs, reference = reference_and_inputs(name, case.symbols)
    got = {n: a.copy() for n, a in arrays.items()}
    case.csdfg(**got, **call_kwargs)
    for array, value in arrays.items():
        # TSVC's integer arrays are gather indices, read-only by construction; TSVC-2.5's are
        # answers (ext_break_capture's out_index), so those are compared.
        if name.endswith(TSVC_SUFFIX) and np.issubdtype(value.dtype, np.integer):
            continue
        assert np.allclose(reference[array], got[array], rtol=TOL, atol=TOL, equal_nan=True), \
            f"{name}/{array}: canonicalize diverges from the numpy oracle"


def fuse_move_ifs_inputs(k: int) -> tuple[dict, dict, dict]:
    """``(arrays, call_kwargs, reference)`` for ``fuse_move_ifs`` at loop-invariant guard value ``k``."""
    arrays, _scalars = tsvc_2_5.make_inputs(ext_program("fuse_move_ifs"))
    reference = {name: array.copy() for name, array in arrays.items()}
    tsvc_2_5_numpy.ref_fuse_move_ifs(**reference, k=k)
    return arrays, {"LEN_2D": tsvc_2_5.SIZES["LEN_2D"], "K": k}, reference


def test_fuse_move_ifs_fuses_both_guarded_nests_into_one_predicated_map():
    """Both guards become predicates of ONE flat (i, j) map -- neither nest nor guard stays control flow.

    :data:`EXPECTED`'s ``(1, 0, 0)`` is a count, and a count cannot tell the fused nest apart from a
    lost one: dropping the ``K > 0`` nest entirely would also leave one parallel map and no
    sequential map. This pins the shape that makes the count the right one, so a regression to the
    pre-fusion ``(2, 1, 0)`` and a regression that loses a guard both go red here and not only in
    the table.
    """
    sdfg = canonical_sdfg("fuse_move_ifs")
    maps = [n.map for n, _ in sdfg.all_nodes_recursive() if isinstance(n, nd.MapEntry)]
    assert len(maps) == 1, f"the two guarded nests must share one iteration space, got {len(maps)} maps"
    fused = maps[0]
    assert len(fused.params) == 2, f"the fused map must span i and j together, got {fused.params}"
    assert str(fused.range) == "0:LEN_2D, 0:LEN_2D", f"the fused map must cover the whole square, got {fused.range}"

    staged = {
        lhs: rhs
        for sd in sdfg.all_sdfgs_recursive()
        for edge in sd.all_interstate_edges()
        for lhs, rhs in edge.data.assignments.items()
    }
    row_guards = [lhs for lhs, rhs in staged.items() if rhs.startswith("cond[")]
    assert len(row_guards) == 1, f"cond[i] must be staged exactly once inside the map body, got {row_guards}"
    guards = [{str(s)
               for s in cond.get_free_symbols()} for sd in sdfg.all_sdfgs_recursive()
              for cfr in sd.all_control_flow_regions(recursive=False) if isinstance(cfr, ConditionalBlock)
              for cond, _branch in cfr.branches if cond is not None]
    assert guards == [{"K", row_guards[0]}] * 3, \
        f"the three live guard combinations must each test BOTH sunk guards, got {guards}"


@pytest.mark.parametrize("k", (0, 3))
def test_fuse_move_ifs_sunk_guards_still_skip_the_work_they_guard(k: int):
    """Cells whose guard is false keep the value the caller pre-filled, at ``K`` false as well as true.

    Predication only preserves the kernel if a false predicate writes NOTHING, and the corpus binds
    ``K`` to 3, so the arm the sink newly turned into a predicate -- ``K <= 0``, which must leave
    every cell of ``b`` alone -- runs nowhere else in this file. Compared exactly rather than within
    :data:`TOL`: both bodies are one elementwise scale and one elementwise offset, so a preserved
    cell is bit-identical and an approximate compare would hide a cell that was written back with
    its own value plus rounding.
    """
    arrays, call_kwargs, reference = fuse_move_ifs_inputs(k)
    got = {name: array.copy() for name, array in arrays.items()}
    built("fuse_move_ifs").csdfg(**got, **call_kwargs)
    for array, value in reference.items():
        assert np.array_equal(value, got[array]), f"fuse_move_ifs/{array} at K={k}: predication changed the values"


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
