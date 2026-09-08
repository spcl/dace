# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Fix the two launch shapes canonicalization cannot pick between, and record the one not taken.

Canonicalization emits one map per parallel loop and says nothing about how that map is spread over
a device. The offload then gives every map the whole domain as its grid, and ``AddThreadBlockMaps``
cuts it into thread blocks. On MI300A (304 CUs, 64-lane wavefronts) that lands on two opposite
launch shapes, and this pass is what answers each.

**Too many blocks onto one address.** A reduction over ~2.6e8 elements lands on the device as ~2e6
blocks that each atomically update a SINGLE accumulator.
:class:`~dace.transformation.dataflow.gpu_grid_stride_tiling.GPUGridStridedTiling` replaces that with
a grid the device can hold resident and a strided loop inside it: still one atomic per block, but
836x fewer blocks, so 836x less contention on the one address they all want. That transformation
already existed and was wired into no pipeline; this pass is the wiring, plus the gate.

The gate is contention, and it was NOT the first guess. Folding the grid of a plain streaming map --
the ~400k-block elementwise case -- looked like the obvious win and MEASURED as nothing to gain and
something to lose. On llr-focus40 at XL, folding every eligible kernel regardless left the 21
non-accumulating ones between 0.99x and 1.19x, with no kernel outside noise on the fast side and
five materially slower (``fuse_stencil_through_transient`` 2.19 -> 2.62 ms, ``tsvc_2_s252``
2.16 -> 2.40, ``tsvc_2_s2710`` 2.66 -> 2.87, ``ext_war_unit`` 4.56 -> 4.81, ``tsvc_2_s1244``
3.18 -> 3.32). Those kernels are HBM-bound: block dispatch was never what they were waiting for, and
the strided loop only costs them the straight-line body the codegen had. The four that ended in a
one-address WCR went the other way by up to an order of magnitude (``tsvc_2_s311`` 18.85 -> 1.22 ms,
``quasi_affine_reduce_odd`` 18.83 -> 1.89, ``tsvc_2_s3111`` 9.75 -> 2.45, ``tsvc_2_s319``
2.60 -> 2.15). Contention separated the two groups with nothing in between, so contention is the
gate.

**Too few blocks.** The other family is one ``GPU_Device`` map inside a sequential loop, where the
launch count is the loop's trip count: 9.8k launches for ``tsvc_2_s233``, 22.8k for ``wf_diff_skew``,
each filling 39 to 90 blocks of a 304-CU device before a device-wide barrier. Grid-stride cannot
help a grid that is already smaller than the device -- there is nothing to fold -- so this pass does
not apply it there. What removes that cost is a persistent kernel with a grid-wide barrier, which is
a different transformation and is NOT taken here. It is recorded instead, as a
``specialization_hint``, so the reader of the canonical form is told the shape, its price, and the
rewrite that pays it.

Runs after ``AddThreadBlockMaps``, which is what creates the ``(GPU_Device, GPU_ThreadBlock)`` pair
the tiling matches, and which supplies the block extent this pass keeps rather than re-chooses.
"""
from typing import Any, Dict, List, Optional, Tuple

from dace import SDFG, dtypes, properties, symbolic
from dace.sdfg import nodes
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.analysis import loop_analysis

#: Blocks a grid-strided kernel keeps. MI300A holds 304 CUs x 8 blocks of 256 threads resident, so
#: 2432 is one full device with no tail. MEASURED on ``tsvc_2_s311`` at XL, which is the shape this
#: pass fires on: 608 -> 2.12 ms, 1216 -> 1.58, 2432 -> 1.28, 4864 -> 1.38, 9728 -> 1.31. One device
#: is the minimum of that curve; under-filling costs 1.65x and over-filling gives the contention
#: back.
DEVICE_GRID_BLOCKS = 2432

#: A grid known at compile time to be at most this many blocks is left alone: folding it would only
#: add index arithmetic to a kernel that already fits the device once. Symbolic grids are decided at
#: runtime instead, by the ``Min(DEVICE_GRID_BLOCKS, extent)`` the tiling emits.
SMALL_GRID_BLOCKS = DEVICE_GRID_BLOCKS

#: Launches below which the persistent-kernel note is noise. At ~4 us of launch latency a thousand
#: launches is ~4 ms, which is the point where the launches start to be visible next to the
#: 10-100 ms totals these kernels run in.
LAUNCH_HINT_FLOOR = 1000

#: What a launch costs on this class of device, in microseconds. Used only to price the hint.
LAUNCH_LATENCY_US = 4

#: The hint's opening words, matched to keep a re-run from recording the same finding twice.
PERSISTENT_KERNEL_LEAD = 'one kernel launch per trip of an enclosing sequential loop'

PERSISTENT_KERNEL = (PERSISTENT_KERNEL_LEAD + ': {launches} launches per call, each '
                     'filling {blocks} blocks of {threads} threads and ending in a device-wide barrier.\n'
                     'Alternative: one persistent kernel that holds the grid across the whole loop and replaces each '
                     'launch boundary with a grid-wide barrier.\n'
                     'GPU: the persistent form is the one that removes this cost -- at ~{latency} us per launch the '
                     'launches alone are ~{cost} us. MEASURED on this shape at the sizes it ships with: 39 to 90 '
                     'blocks per launch on a 304-CU MI300A, so each launch underfills the device as well as paying '
                     'for itself.\n'
                     'Not taken here: a grid-wide barrier needs a cooperative launch and a grid sized to what the '
                     'device can hold resident, neither of which this pass establishes. Recorded, not decided.\n'
                     'Both are correct. Measure before choosing.')


def record(carrier: Any, hint: str) -> None:
    """Append ``hint`` to ``carrier``'s ``specialization_hint``, once.

    Appended rather than assigned because a carrier usually already says what KIND of scope it is
    (``AnnotateLoopKinds``), and that is a different fact from this one, not a stale version of it.
    """
    existing = carrier.specialization_hint or ''
    if hint in existing:
        return
    carrier.specialization_hint = f'{existing}\n{hint}' if existing else hint


def enclosing_loops(sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry) -> List[LoopRegion]:
    """The sequential loops that re-enter ``entry``, outermost last."""
    from dace.transformation.helpers import get_parent_map_and_loop_scopes
    return [s for s in get_parent_map_and_loop_scopes(sdfg, entry, state) if isinstance(s, LoopRegion)]


def trip_count(loop: LoopRegion) -> Optional[Any]:
    """``loop``'s iteration count, or ``None`` when its bounds do not give one."""
    start = loop_analysis.get_init_assignment(loop)
    end = loop_analysis.get_loop_end(loop)
    stride = loop_analysis.get_loop_stride(loop)
    if start is None or end is None or stride is None:
        return None
    try:
        if int(str(symbolic.simplify(stride))) == 0:
            return None
    except (TypeError, ValueError):
        pass  # a symbolic stride cannot be zero-checked, and is not the degenerate case
    return symbolic.simplify(symbolic.int_floor(end - start, stride) + 1)


def as_int(expr: Any) -> Optional[int]:
    """``expr`` as a python int when it is a compile-time constant, else ``None``."""
    try:
        return int(symbolic.simplify(expr))
    except (TypeError, ValueError, AttributeError):
        return None


def thread_block_child(state: SDFGState, entry: nodes.MapEntry) -> Optional[nodes.MapEntry]:
    """The single ``GPU_ThreadBlock`` map directly inside ``entry``, or ``None``.

    ``None`` for anything else on purpose: two inner scopes, or an inner scope that is not a thread
    block, is a kernel someone else already shaped, and re-tiling it would overrule that decision.
    """
    scope_children = state.scope_children()[entry]
    inner = [n for n in scope_children if isinstance(n, nodes.MapEntry)]
    if len(inner) != 1 or inner[0].map.schedule != dtypes.ScheduleType.GPU_ThreadBlock:
        return None
    return inner[0]


def strip_mined_pair(state: SDFGState, entry: nodes.MapEntry) -> Optional[nodes.MapEntry]:
    """``entry``'s inner map iff the two are the 1-D pair ``AddThreadBlockMap`` strip-mines.

    That pair is recognised by the inner map STARTING at the outer parameter, which is what
    strip-mining produces and what makes the outer map's step the block width. A thread-block map
    that starts anywhere else came from a library expansion that chose its own block shape, and this
    pass does not overrule that choice.
    """
    if entry.map.schedule != dtypes.ScheduleType.GPU_Device or len(entry.map.params) != 1:
        return None
    inner = thread_block_child(state, entry)
    if inner is None or len(inner.map.params) != 1:
        return None
    return inner if str(inner.map.range[0][0]) == entry.map.params[0] else None


def block_threads(entry: nodes.MapEntry, inner: nodes.MapEntry) -> Optional[int]:
    """Threads per block of the strip-mined pair ``(entry, inner)``, or ``None`` when it is not one.

    The outer step counts index units, the inner step says how many of them one thread covers, so
    the thread count is their quotient. A remainder means the two maps are not a strip-mining of one
    another and the pair is left alone.
    """
    outer_step, inner_step = as_int(entry.map.range[0][2]), as_int(inner.map.range[0][2])
    if outer_step is None or inner_step is None or inner_step <= 0:
        return None
    return outer_step // inner_step if outer_step % inner_step == 0 else None


def contended_accumulator(state: SDFGState, entry: nodes.MapEntry) -> bool:
    """``True`` iff every block of this kernel updates ONE address with a conflict resolution.

    That is the whole gate. A ``wcr`` onto a subset of one element means the grid is not just wide,
    it is wide onto a single atomic target, and halving the grid halves the contention on it. A wcr
    that scatters over many addresses (``out[ix[i]] += ...``) contends with nothing in particular and
    is deliberately NOT matched: ``scatter_accum_dup`` has one and measured 1.01x, i.e. nothing.
    """
    for node in state.scope_subgraph(entry).nodes():
        if not isinstance(node, nodes.MapExit):
            continue
        for edge in state.in_edges(node) + state.out_edges(node):
            if edge.data.wcr and edge.data.subset is not None and as_int(edge.data.subset.num_elements()) == 1:
                return True
    return False


def grid_is_known_small(entry: nodes.MapEntry) -> bool:
    """``True`` iff ``entry``'s grid is a compile-time constant that already fits the device."""
    blocks = as_int(entry.map.range.num_elements())
    return blocks is not None and blocks <= SMALL_GRID_BLOCKS


@properties.make_properties
class GridStrideKernels(ppl.Pass):
    """Grid-stride the oversized kernels; hint the launch-bound ones."""

    CATEGORY: str = 'Device Specialization'

    def modifies(self) -> ppl.Modifies:
        return ppl.Modifies.Nodes | ppl.Modifies.Edges | ppl.Modifies.Scopes

    def should_reapply(self, _modified: ppl.Modifies) -> bool:
        return False

    def depends_on(self):
        return set()

    def apply_pass(self, sdfg: SDFG, _pipeline_results: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """Rewrite and annotate ``sdfg`` in place.

        :param sdfg: the offloaded, thread-block-tiled SDFG to specialize.
        :param _pipeline_results: unused.
        :returns: ``(kernels grid-strided, kernels hinted)``, or ``None`` when neither happened.
        """
        # Collected first: the tiling rewires scopes, and iterating a graph being rewritten skips
        # nodes.
        kernels = [(n, s) for n, s in sdfg.all_nodes_recursive()
                   if isinstance(n, nodes.MapEntry) and n.map.schedule == dtypes.ScheduleType.GPU_Device]
        strided = hinted = 0
        for entry, state in kernels:
            # Hint first: the tiling rewrites ``entry`` into a sequential map under a fresh device
            # map, and the launch shape being recorded is the one BEFORE that rewrite.
            hinted += self.hint_persistent(sdfg, state, entry)
            strided += self.grid_stride(sdfg, state, entry)
        return (strided, hinted) if strided or hinted else None

    def grid_stride(self, sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry) -> int:
        """Fold ``entry``'s grid onto a device-sized one, or leave it alone. ``1`` iff folded."""
        inner = strip_mined_pair(state, entry)
        if inner is None or grid_is_known_small(entry):
            return 0
        # The measured gate: only a kernel whose blocks all contend for one accumulator. See the
        # module docstring for the two groups this separates and the numbers behind it.
        if not contended_accumulator(state, entry):
            return 0
        # A kernel re-launched by a sequential loop gets ONE front per launch, which measured 39 to
        # 90 blocks on a 304-CU device across the llr-focus40 loop-carried kernels: there is no wide
        # grid to fold. Folding it anyway MEASURED 1.02x to 1.07x slower on the six that have this
        # shape, so this family is hinted below instead.
        if enclosing_loops(sdfg, state, entry):
            return 0
        from dace.transformation.dataflow.gpu_grid_stride_tiling import GPUGridStridedTiling
        # The block width in THREADS, which is what the tiling means by ``block_dim``: it emits a
        # thread-block map of ``(0, block_dim - 1, 1)``. Strip-mining a map of step ``s`` by ``t``
        # threads leaves the outer stepping ``s * t``, so the outer step alone over-reports the
        # block by a factor of ``s`` -- 2048 threads for the step-4 ``s31111`` at 512, past the
        # 1024 every device allows. Handed straight back so the rewrite keeps the thread block it
        # was given: block size is ``select_gpu_device_block_size``'s decision, and this pass only
        # changes the grid.
        block_dim = block_threads(entry, inner)
        if block_dim is None:
            return 0
        where = {'outer_map_entry': entry, 'inner_map_entry': inner}
        options = {'max_grid_dim': DEVICE_GRID_BLOCKS, 'block_dim': block_dim}
        # ``state.sdfg``, not the root: ``apply_to`` locates the state by scanning the SDFG it is
        # handed, and a kernel inside a nested SDFG (a lifted Reduce, say) is not in the root's
        # states -- it raised StopIteration on two of llr-focus40 before this was the owning SDFG.
        owner = state.sdfg
        if not GPUGridStridedTiling.can_be_applied_to(owner, options=options, **where):
            return 0
        GPUGridStridedTiling.apply_to(owner, options=options, save=False, **where)
        return 1

    def hint_persistent(self, sdfg: SDFG, state: SDFGState, entry: nodes.MapEntry) -> int:
        """Record the persistent-kernel rewrite ``entry`` would want. ``1`` iff recorded."""
        loops = enclosing_loops(sdfg, state, entry)
        if not loops or PERSISTENT_KERNEL_LEAD in (entry.specialization_hint or ''):
            return 0
        launches: Any = 1
        for loop in loops:
            count = trip_count(loop)
            if count is None:
                return 0  # nothing to say about a launch count that cannot be written down
            launches = symbolic.simplify(launches * count)
        known = as_int(launches)
        if known is not None and known < LAUNCH_HINT_FLOOR:
            return 0
        inner = thread_block_child(state, entry)
        hint = PERSISTENT_KERNEL.format(
            launches=launches,
            blocks=symbolic.simplify(entry.map.range.num_elements()),
            threads=symbolic.simplify(inner.map.range.num_elements()) if inner is not None else 'block-sized',
            latency=LAUNCH_LATENCY_US,
            cost=symbolic.simplify(launches * LAUNCH_LATENCY_US))
        record(entry, hint)
        # Also on the innermost enclosing loop, because that is where a reader meets it: the CUDA
        # target renders no map hints, so a hint left only on the kernel map is invisible in the
        # emitted form, while ``control_flow.py`` renders every loop's.
        record(loops[0], hint)
        return 1

    def report(self, pass_retval: Tuple[int, int]) -> str:
        strided, hinted = pass_retval
        return f'Grid-strided {strided} kernel(s); hinted {hinted} launch-bound kernel(s)'
