# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""How CPF spells the functions the DaCe runtime headers normally provide.

CPF (canonical parallel form) emits C++ -- or C23, which is the same semantics in a language
with no templates and no overloading -- that builds against a bare host compiler: no
``-I dace/runtime/include``, no ``libdace``. Every function the ordinary code generators reach for
therefore has to be re-expressed, and there are four ways to do it:

``STD_RENAMES``
    The C++ standard library has the same function under a different name. A pure rename.

``REWRITES``
    No standard function exists, but the operation is a short expression over its arguments. The
    call site is replaced by that expression, so nothing has to be declared.

``INLINE_DEFINITIONS``
    Neither of the above: the operation needs a real function (it is recursive, generic over
    signedness, or simply too long to inline at every use). CPF emits the definition once, at the
    top of the translation unit, and only for the helpers that translation unit actually calls.

``C_REWRITTEN_IN_NATIVE_CODE``
    C only: the C answer needs a type or a predicate only the CALL SITE has, so the call site is
    rewritten. The scans, the scan identities and the duplicate check need the element types the
    call site names, and each becomes a call to a function typed for them; the find-first takes a
    predicate, which is a C++ lambda and in C becomes a function of its own. Every entry here is a
    C++ helper whose C answer is a rewrite rather than a definition, which is what lets the anti-rot
    tests still demand an answer for each one.

This module is deliberately a LEAF: it imports nothing from ``dace`` but the leaves :mod:`dace.optionals`,
:mod:`dace.ordered` and :mod:`dace.tile_functions`. ``dace.symbolic`` (which
imports no code generator) and ``dace.codegen.cppunparse`` both consume it, and a shared table is
what keeps the two printers from drifting -- the same expression reaching C++ through a memlet
subset and through a tasklet body must produce the same text.

Names absent from all three tables are NOT silently passed through: see :func:`lowering_for`.
"""

import contextlib
import ctypes.util
import enum
import functools
import os
import re
from collections.abc import Callable
from typing import Any, NamedTuple

from dace import tile_functions
from dace.optionals import required
from dace.ordered import OrderedSet


class Dialect(enum.Enum):
    """Which C++ vocabulary a printer may emit.

    An enum rather than a boolean, for two reasons that both cost correctness otherwise.

    The printers below are memoized (``symstr`` and ``_sym2cpp`` are both ``lru_cache``d), so the
    dialect MUST reach the cache key: a mode read from configuration inside a printer would let a
    ``RUNTIME`` call warm the cache and a later ``STANDALONE`` call read that entry back, emitting
    ``dace::math::pow`` into a translation unit that never includes the header. Passing the dialect
    as an argument is what puts it in the key.

    And a boolean fourth argument sits next to ``cpp_mode``, which is also a boolean: transposing
    the two at one call site is a silent behaviour change that no type checker would see. A
    ``Dialect`` cannot be mistaken for ``cpp_mode``.
    """

    #: The DaCe runtime headers are available; emit ``dace::`` names as usual.
    RUNTIME = "runtime"
    #: No DaCe headers; emit only the C++ standard library and CPF's own inline definitions.
    STANDALONE = "standalone"
    #: No DaCe headers and no C++ either: one C23 translation unit. The C standard library is not
    #: type-generic and has no templates, so a helper is a set of typed ``static inline`` functions
    #: (see :data:`C_INLINE_DEFINITIONS`). The printer resolves the argument types and calls the one
    #: for their type by name; the unit defines no macro.
    STANDALONE_C = "standalone_c"
    #: No DaCe headers, one HIP translation unit holding both the host code and the kernels.
    #: C++ like :attr:`STANDALONE`, so every host-side helper and lowering it has applies here
    #: too; what it adds is the device vocabulary.
    #:
    #: The unit may include the ROCm toolkit's own headers -- ``hip_runtime.h`` and hipCUB --
    #: on the same footing as ``<math.h>`` and OpenMP on the host: they ship WITH the compiler
    #: that builds the unit, so requiring them adds no dependency a caller did not already have
    #: by choosing to compile for a GPU. What stays banned is ``dace/``, which does not.
    STANDALONE_HIP = "standalone_hip"
    #: :attr:`STANDALONE_HIP` for NVIDIA: the same unit through the same device path, built by nvcc.
    #: Only the spelling tables and the toolkit headers (``cuda_runtime.h``, CUB) differ.
    STANDALONE_CUDA = "standalone_cuda"


#: The dialect a printer uses when its caller names none. Set only through :func:`dialect_scope`.
#:
#: An AMBIENT default exists because the dialect has to reach hundreds of ``sym2cpp`` call sites --
#: every memlet subset, every allocation extent, every loop bound -- spread over the code
#: generators. Threading a parameter through all of them would be a large diff whose every line is
#: identical, and one missed site emits a ``dace::`` name into a standalone unit.
#:
#: This does NOT reintroduce the memoization hazard the :class:`Dialect` docstring describes. The
#: ambient value is resolved by the OUTERMOST wrapper (``sym2cpp``, ``symstr``) and passed to the
#: memoized inner function as an ordinary argument, so it is still part of the cache key. Nothing
#: inside a cached function reads it.
_active_dialect: Dialect = Dialect.RUNTIME


def active_dialect() -> Dialect:
    """The dialect currently in force (see :func:`dialect_scope`)."""
    return _active_dialect


@contextlib.contextmanager
def dialect_scope(dialect: Dialect):
    """Make ``dialect`` the ambient default for the duration of the block.

    Restores the previous value on the way out, exception or not -- a code generation that raises
    must not leave every later ``sym2cpp`` in the process emitting standalone text.
    """
    global _active_dialect
    previous = _active_dialect
    _active_dialect = dialect
    try:
        yield dialect
    finally:
        _active_dialect = previous


#: The dialects that emit a self-contained translation unit. Everything CPF refuses -- device
#: code, a state struct, an external buffer handshake -- it refuses for both of them, so the many
#: call sites that ask "is this an CPF rendering" ask through :func:`standalone`.
STANDALONE_DIALECTS = frozenset(
    {Dialect.STANDALONE, Dialect.STANDALONE_C, Dialect.STANDALONE_HIP, Dialect.STANDALONE_CUDA}
)

#: The standalone dialects that render DEVICE code. A rendering under one of these admits GPU
#: storages and schedules, and emits both the host code and the kernels into the one unit.
DEVICE_DIALECTS = frozenset({Dialect.STANDALONE_HIP, Dialect.STANDALONE_CUDA})

#: The ``compiler.cuda.backend`` each device dialect generates for. The generator prefixes runtime
#: calls with the backend (``cudaStreamSynchronize``), and the default backend is the machine's, so
#: a HIP unit rendered on an NVIDIA box called CUDA functions it never declares.
DEVICE_BACKENDS: dict[Dialect, str] = {Dialect.STANDALONE_HIP: "hip", Dialect.STANDALONE_CUDA: "cuda"}


def standalone() -> bool:
    """Whether the ambient dialect renders a self-contained unit, C++ or C."""
    return _active_dialect in STANDALONE_DIALECTS


def device() -> bool:
    """Whether the ambient dialect renders device code into the standalone unit."""
    return _active_dialect in DEVICE_DIALECTS


def standalone_c() -> bool:
    """Whether the ambient dialect is :attr:`Dialect.STANDALONE_C`.

    Separate from :func:`standalone` because the two answer different questions: "may I name a
    DaCe runtime symbol" (no, for both) versus "may I write C++" (no, only for this one).
    """
    return _active_dialect is Dialect.STANDALONE_C


#: Node GUID -> ``(origin GUID, description)`` of the library node it came from, for the rendering
#: in progress. Set only through :func:`provenance_scope`.
#:
#: A pure expansion replaces ``Gemm`` with a loop nest, and the loop nest does not say it was a
#: matrix product. CPF restores that: the description is recorded when the node is expanded and
#: written as a comment where the expansion's code is emitted. Keyed by GUID rather than by node
#: object because the code generator runs its own lowering (inlining, copy lifting) between the
#: expansion and the emission.
#:
#: Ambient for the same reason the dialect is: the emission points are inside the code generators,
#: several call layers below anything that knows an expansion happened. Nothing memoized reads it.
_provenance: dict[str, tuple[str, str]] = {}


def describe(guid: str) -> tuple[str, str] | None:
    """``(origin, description)`` for the library node that produced ``guid``, if CPF recorded one.

    ``origin`` is the GUID of the library node itself. The emitter dedupes on it rather than on the
    description text: two separate ``Gemm`` nodes in one program are two things worth commenting,
    while the forty tasklets of one expansion are one.
    """
    return _provenance.get(guid)


class HintDetail(enum.Enum):
    """How much of a ``specialization_hint`` a standalone rendering writes."""

    #: The whole hint: the class, the reason, the proof or its witness, the advice.
    PROOFS = "proofs"
    #: The class: ``parallel`` (``parallel reduction``, ``parallel scan``), or ``sequential`` or ``unknown`` with its
    #: reason. Nothing else is written as a comment.
    KINDS = "kinds"
    #: No hint at all: the code of :attr:`PROOFS`, without its loop comments (the comments-ablation variant).
    NONE = "none"


class LoopClass(enum.Enum):
    """The class a loop comment opens with. Every pass that writes a comment and every reader of one (the kinds
    rendering, the header, the contract legend) goes through this enum, so no pass matches another's text."""

    PARALLEL = "parallel"
    SEQUENTIAL = "sequential"
    UNKNOWN = "unknown"
    #: A loop a library call's lowering writes (a zero fill, a helper's pass): no program loop, so no verdict on the
    #: program and no tuning target. Its comment is ``library loop (<what>): ...``, not ``<class> -- <reason>``.
    LIBRARY = "library loop"

    @property
    def prefix(self) -> str:
        """What a hint's first line starts with: ``parallel -- ``."""
        return f"{self.value} -- "


class Reason(enum.Enum):
    """The opening words of a comment's reason, for the reasons the renderers tell apart from the proof texts: what
    the kinds rendering names, what the header lists apart or leaves out. A reason of any other opening is a proof
    text and has no member."""

    INDEPENDENT = "independent"
    REDUCTION = "reduction"
    SCAN = "scan"
    GUARDED = "guarded by if ("
    #: The in-order arm of a guarded loop, for the inputs its guard rejects.
    FALLBACK = "the else arm of if ("
    CHECKED = "the runtime check found "
    SMALL_EXTENT = "small extent"
    THREAD_TEAM = "thread team"
    NOT_A_LOOP = "not a loop"
    DIMENSION = "dimension "
    TILE = "tile loop"
    INNER_TILE = "inner tile of loop "
    IMPLEMENTATION = "implementation loop of "
    #: A GPU thread-block map that carries no program loop's line of its own: what its threads are.
    THREAD_BLOCK = "thread block"
    #: A block-tiled kernel's lane map: the lanes of the thread block that runs one iteration of the kernel loop.
    BLOCK_TILE = "block tile"
    WAVEFRONT_FRONT = "one wavefront step: "
    WAVEFRONT_COLUMN = "the tiles "
    # ``sequential``: an opening, or for a proven dependence the meaning of its kind, which follows the element
    PROVEN_FLOW = "an iteration reads what an earlier one wrote"
    PROVEN_ANTI = "an iteration overwrites what an earlier one still has to read"
    PROVEN_OUTPUT = "two iterations write the same element and the last write must win"
    CARRIED_COUNTER = "carries a value from "
    ORDERED_CALL = "ordered call: "
    EARLY_EXIT = "early exit ("
    WAVEFRONT_STEP = "wavefront step: "
    WAVEFRONT_TILE = "wavefront of tiles "
    WAVEFRONT_INTERIOR = "inside one wavefront tile "
    CHUNK_ORDER = "in order within a chunk"
    WHILE_LOOP = "while loop: "
    #: Followed by who pinned the loop: a pass (``LoopRegion.pinned_by``) or the user's schedule.
    PINNED = "kept in order by "
    # ``unknown``: one per refusal of the dependence test, then what kept the solver from an answer
    NON_UNIQUE_WRITE = "the solver could not show that each iteration writes a different element"
    SAME_ELEMENTS = "every iteration writes the same elements"
    DYNAMIC_WRITE = "a conditional write to "
    OVERLAPPING_WRITES = "the dependence test could not show that writes to "
    CARRIED_DEPENDENCE = "the loop reads "
    CARRIED_SYMBOL = "may carry a value from one iteration to the next"
    LIVE_OUT_SYMBOL = "is read after the loop"
    UNANALYZABLE_RANGE = "not analyzed for dependences: the loop bounds"
    UNSUPPORTED_BODY = "not analyzed for dependences: the body"
    ORDERED_SIDE_EFFECT = "the body has a side effect"
    REFUSED = "the dependence test refused the loop"
    COMPUTED_VALUE = "the only conflict the solver found needs "
    #: ``unknown``: a branch condition reads a container the loop writes.
    GUARD_READS_WRITTEN = "the guard "
    #: ``unknown``: the index of the read that may conflict is a symbol the loop assigns.
    ASSIGNED_INDEX = "the index "
    #: ``unknown``: the test saw the read that may conflict only as a range it could bound.
    RANGE_READ = "the test could only bound the read "
    #: ``unknown``: the solver used up its resource budget before it found a conflict or showed there is none.
    BUDGET_EXPIRED = "the solver ran out of its budget before it found a conflict or showed there is none"


#: What a pass or the renderer appends to the class line of a proven-parallel loop that has no thread pragma (or, on a
#: GPU, no kernel), followed by why: the loop stays ``parallel``, because it is; this says it is not run in parallel.
NOT_PARALLELIZED = "; not parallelized: "
#: What takes the place of :data:`NOT_PARALLELIZED` on a proven-parallel loop that has no thread pragma but an ``omp
#: simd`` one: its iterations run as vector lanes of one thread, so it is vectorized, followed by why it forks no team.
VECTORIZED_ONLY = "; vectorized (omp simd), not thread-parallel: "
#: What a GPU pass appends to the class line of a proven-parallel loop that renders as a plain ``for`` but whose
#: iterations the threads of a kernel share out by index (a grid-stride or lane-strided loop), followed by how: the
#: loop is run in parallel, so it takes no :data:`NOT_PARALLELIZED` note.
SPREAD_OVER_THREADS = "; spread over the GPU threads: "
#: How a :data:`SPREAD_OVER_THREADS` note shares the loop out (the words that open it) -> its header legend row, as
#: :data:`REASON_LEGEND`'s.
SPREAD_LEGEND: dict[str, tuple[str, str]] = {
    "lane-strided": (
        "lane t of one thread block takes iterations t, t + B, ...",
        "nothing; block size B is a tuning knob",
    ),
    "grid-stride": ("each resident block takes every G-th tile", "nothing; the grid G is the blocks resident at once"),
}
#: What a GPU kernel's comment appends to its class line: how the launch grid maps the loop's iterations to threads.
KERNEL_MAPPING = "; kernel: "
#: What a GPU thread-block map's comment appends to the class line of the program loop it runs.
THREAD_BLOCK_MAPPING = "; thread block: "


#: The members a reason holds after its opening: a proven dependence names its element first, a carried symbol its name.
INNER_REASONS = (
    Reason.PROVEN_FLOW,
    Reason.PROVEN_ANTI,
    Reason.PROVEN_OUTPUT,
    Reason.CARRIED_COUNTER,
    Reason.CARRIED_SYMBOL,
    Reason.LIVE_OUT_SYMBOL,
)


def reason_of(reason: str) -> Reason | None:
    """The :class:`Reason` ``reason`` opens with, else the first of :data:`INNER_REASONS` it holds, else ``None``."""
    return next((kind for kind in Reason if reason.startswith(kind.value)), None) or next(
        (kind for kind in INNER_REASONS if kind.value in reason), None
    )


#: Reason -> (what it means, what would make the loop parallel or what to do), one line each for the header legend.
REASON_LEGEND: dict[Reason, tuple[str, str]] = {
    Reason.INDEPENDENT: (
        "no iteration touches an element another one writes",
        "nothing to prove; tune schedule, tiles",
    ),
    Reason.REDUCTION: ("iterations combine into one value with an associative op", "nothing; mind rounding order"),
    Reason.SCAN: ("running combination, two blocked passes", "nothing; the passes are parallel"),
    Reason.GUARDED: ("parallel arm of a run-time check", "nothing; the check picks the arm"),
    Reason.FALLBACK: ("in-order arm for inputs the check rejects", "inputs that pass the check"),
    Reason.CHECKED: ("verdict of the run-time check on this input", "inputs that pass the check"),
    Reason.SMALL_EXTENT: ("unproven, too few iterations to share", "nothing; not worth threads"),
    Reason.THREAD_TEAM: ("one team; omp for loops inside split work", "nothing"),
    Reason.NOT_A_LOOP: ("a block, not a loop", "nothing"),
    Reason.DIMENSION: ("one loop of a parallel nest", "nothing; see the nest's line"),
    Reason.TILE: ("loop over tiles of a tiled loop", "nothing; tile size is a tuning knob"),
    Reason.INNER_TILE: ("loop inside one tile", "nothing; tile size is a tuning knob"),
    Reason.IMPLEMENTATION: ("loop a helper or library lowering wrote", "tune the call, not the loop"),
    Reason.THREAD_BLOCK: ("threads of one GPU thread block", "nothing; block size is a tuning knob"),
    Reason.BLOCK_TILE: ("lanes of the one block that runs a kernel iteration", "nothing; block size is a tuning knob"),
    Reason.WAVEFRONT_FRONT: ("points of one wavefront step", "nothing; tile the front"),
    Reason.WAVEFRONT_COLUMN: ("tiles of one wavefront step", "nothing; tile size is a tuning knob"),
    Reason.PROVEN_FLOW: (
        "flow (RAW): a later iteration reads an earlier write",
        "none; a scan or a wavefront if affine",
    ),
    Reason.PROVEN_ANTI: ("anti (WAR): a later write overwrites an earlier read", "copy the read data first"),
    Reason.PROVEN_OUTPUT: ("output (WAW): the last write must win", "keep only the last write"),
    Reason.CARRIED_COUNTER: ("a scalar hands its value to the next iteration", "a scan of its increments"),
    Reason.ORDERED_CALL: ("a call whose effects must stay in order", "a call without ordered effects"),
    Reason.EARLY_EXIT: ("later iterations must not run after the exit", "a find-first search"),
    Reason.WAVEFRONT_STEP: ("each step reads the previous step", "none; the front inside is parallel"),
    Reason.WAVEFRONT_TILE: ("each tile step reads the previous one", "none; the tile column is parallel"),
    Reason.WAVEFRONT_INTERIOR: ("original order inside one tile", "none; smaller tiles"),
    Reason.CHUNK_ORDER: ("order within a chunk; chunks in parallel", "nothing; chunk size is a tuning knob"),
    Reason.WHILE_LOOP: ("each test reads the previous trip", "a counted loop"),
    Reason.PINNED: ("a pass or the user's schedule kept it in order; it tests independent", "see who pinned it"),
    Reason.NON_UNIQUE_WRITE: ("two iterations may write one element", "distinct indices (a permutation)"),
    Reason.SAME_ELEMENTS: ("every iteration writes the same elements", "a private copy or a reduction"),
    Reason.DYNAMIC_WRITE: ("conditional write at an unfixed index", "distinct indices"),
    Reason.OVERLAPPING_WRITES: ("writes may overlap across iterations", "disjoint writes"),
    Reason.CARRIED_DEPENDENCE: ("a read may see an earlier iteration's write", "no earlier write to what it reads"),
    Reason.CARRIED_SYMBOL: ("a scalar may hand a value on", "assign it before each read"),
    Reason.LIVE_OUT_SYMBOL: ("a scalar is read after the loop", "keep the last value (lastprivate)"),
    Reason.UNANALYZABLE_RANGE: ("bounds or step not analyzable", "integer bounds fixed before the loop"),
    Reason.UNSUPPORTED_BODY: ("body holds a block the test cannot read", "rewrite it as an if"),
    Reason.ORDERED_SIDE_EFFECT: ("a side effect whose order counts", "an effect whose order does not matter"),
    Reason.REFUSED: ("the test refused the loop", "see the named array"),
    Reason.COMPUTED_VALUE: ("conflict needs a value computed before the loop", "a fact bounding that value"),
    Reason.GUARD_READS_WRITTEN: ("a guard reads data the loop writes", "no earlier write to what it reads"),
    Reason.ASSIGNED_INDEX: ("an index the loop assigns, not modeled", "an affine index"),
    Reason.RANGE_READ: ("a read known only as a bounded range", "an affine index"),
    Reason.BUDGET_EXPIRED: ("the solver gave up within its budget", "a simpler index or a fact"),
}


#: A hint's class prefix -> the one word :attr:`HintDetail.KINDS` writes; a reduction or a scan is a kind of parallel loop.
KIND_WORDS: tuple[tuple[str, str], ...] = (
    (LoopClass.PARALLEL.prefix + Reason.REDUCTION.value, "parallel reduction"),
    (LoopClass.PARALLEL.prefix + Reason.SCAN.value, "parallel scan"),
    *((cls.prefix, cls.value) for cls in LoopClass if cls is not LoopClass.LIBRARY),
)

#: Lines an older hint carried under its class line, before the rendering's header (``cpf.CONTRACT_LINES``) took
#: over saying what a class means; :func:`hint_comment` leaves them out of a hint stored with one.
SETTLED_PARALLEL = "settled: proven, no need to re-check; focus: schedule, tiling, vectorization, fusion, data layout"
OPEN_UNKNOWN = (
    "open: not proven either way; kept sequential to be safe, and may well be parallel -- a fact about the "
    "indices (a bound, a permutation) settles it"
)
HEADER_HINT_LINES = frozenset({SETTLED_PARALLEL, OPEN_UNKNOWN})

_hint_detail: HintDetail = HintDetail.PROOFS
#: Whether a standalone rendering ends each tasklet statement with ``// <label>`` (see :func:`tasklet_labels_scope`).
labels_in_force = False


def tasklet_labels() -> bool:
    """Whether the rendering in force names each statement's tasklet in a trailing comment."""
    return labels_in_force


@contextlib.contextmanager
def tasklet_labels_scope(on: bool):
    """Name (or not) each statement's tasklet in a trailing comment for the duration of the block."""
    global labels_in_force
    previous = labels_in_force
    labels_in_force = on
    try:
        yield on
    finally:
        labels_in_force = previous


#: Whether a device rendering leaves out the DaCe runtime scaffolding (see :func:`no_boilerplate_scope`).
no_boilerplate_in_force = False


def no_boilerplate() -> bool:
    """Whether the device rendering in force is one unit without DaCe scaffolding (:func:`no_boilerplate_scope`)."""
    return no_boilerplate_in_force and device()


@contextlib.contextmanager
def no_boilerplate_scope(on: bool):
    """Render a device form without the DaCe runtime scaffolding for the duration of the block.

    The unit then has no GPU context, no stream variables (every launch, copy and wait is on the default stream,
    ``nullptr``), no state struct, no ``__cpf_runkernel_*`` launchers (each kernel launches where the entry calls
    it) and no persistent buffers: a device array the entry allocates lives in the caller's workspace. The host
    entry and the kernels are one translation unit.
    """
    global no_boilerplate_in_force
    previous = no_boilerplate_in_force
    no_boilerplate_in_force = on
    try:
        yield on
    finally:
        no_boilerplate_in_force = previous


def hint_detail() -> HintDetail:
    """The hint detail currently in force (see :func:`hint_detail_scope`)."""
    return _hint_detail


@contextlib.contextmanager
def hint_detail_scope(detail: HintDetail):
    """Make ``detail`` the ambient hint detail for the duration of the block."""
    global _hint_detail
    previous = _hint_detail
    _hint_detail = detail
    try:
        yield detail
    finally:
        _hint_detail = previous


def kind_word(hint: str) -> str | None:
    """The class word of ``hint``'s first classed line, or ``None`` if no line names a class."""
    for line in str(hint).splitlines():
        for prefix, word in KIND_WORDS:
            if line.startswith(prefix):
                return word
    return None


#: The parts of a class line the kinds rendering leaves out: the example iterations and the alternative.
KIND_EXTRAS = re.compile(r" \(example[^)]*\)|; alt: .*$")


def kind_line(hint: str) -> str | None:
    """The kinds rendering of ``hint``: its class word, and for a ``sequential`` or ``unknown`` loop the reason (its
    class line without the example iterations and the alternative); ``None`` if no line names a class."""
    word = kind_word(hint)
    first = str(hint).split("\n", 1)[0]
    if word == LoopClass.PARALLEL.value and NOT_PARALLELIZED in first:
        return f"{word}: {first.split(NOT_PARALLELIZED, 1)[1].strip()}"
    if word not in (LoopClass.SEQUENTIAL.value, LoopClass.UNKNOWN.value):
        return word
    for line in str(hint).splitlines():
        if " -- " in line and kind_word(line) == word:
            reason = KIND_EXTRAS.sub("", line.split(" -- ", 1)[1]).strip()
            return f"{word}: {reason}" if reason else word
    return word


def hint_comment(hint: str | None, indent: str = "") -> str:
    """Render a ``specialization_hint`` as a comment block, or ``''`` when there is none.

    A no-op outside a standalone rendering: the hint is a note to whoever reads the maximally
    parallel form -- a specializing pass, or a person -- and the ordinary build has a target
    already, so emitting it there would be noise in code nobody reads.

    Under :attr:`HintDetail.KINDS` only the class is written (:func:`kind_line`: a ``sequential`` or ``unknown`` loop
    keeps its reason), and a hint naming no class writes nothing.
    Under :attr:`HintDetail.NONE` nothing is written.

    Multi-line hints stay multi-line. The alternatives a hint describes are per-device and do not
    read as one sentence: ``CPU: ... / GPU: ...`` is the shape, one line each.
    """
    if not hint or not standalone() or _hint_detail is HintDetail.NONE:
        return ""
    if _hint_detail is HintDetail.KINDS:
        line = kind_line(hint)
        return f"{indent}// {line}\n" if line else ""
    return "".join(
        f"{indent}// {line}\n"
        for line in str(hint).splitlines()
        if line.strip() and line.strip() not in HEADER_HINT_LINES
    )


@contextlib.contextmanager
def provenance_scope(provenance: dict[str, tuple[str, str]]):
    """Make ``provenance`` the ambient GUID -> description map for the duration of the block."""
    global _provenance
    previous = _provenance
    _provenance = provenance
    try:
        yield provenance
    finally:
        _provenance = previous


#: The heap transients a standalone rendering places in the caller's workspace instead of allocating, as data
#: name -> descriptor, and the workspace's name; empty outside :func:`workspace_carve_scope`.
carved_workspace_arrays: dict[str, Any] = {}
carved_workspace_name: str = ""

#: The workspace's size argument and the entry's name, for the size check of a :func:`no_boilerplate` entry
#: (:func:`workspace_offset_declaration`); empty outside :func:`workspace_carve_scope`.
carved_workspace_size_name: str = ""
carved_workspace_entry: str = ""

#: Each array placed in the workspace starts on this boundary, the alignment its own allocation had.
WORKSPACE_ALIGNMENT = 64


@contextlib.contextmanager
def workspace_carve_scope(workspace: str, carve: dict[str, Any], size: str = "", entry: str = ""):
    """Make ``carve`` the ambient workspace layout for the duration of the block (:func:`workspace_slot`); ``size``
    and ``entry`` name the workspace's size argument and the entry the size check reports (:func:`no_boilerplate`)."""
    global carved_workspace_arrays, carved_workspace_name, carved_workspace_size_name, carved_workspace_entry
    previous = carved_workspace_arrays, carved_workspace_name, carved_workspace_size_name, carved_workspace_entry
    carved_workspace_arrays, carved_workspace_name = carve, workspace
    carved_workspace_size_name, carved_workspace_entry = size, entry
    try:
        yield carve
    finally:
        carved_workspace_arrays, carved_workspace_name, carved_workspace_size_name, carved_workspace_entry = previous


def workspace_bound(carve: dict[str, Any]) -> Any:
    """An upper bound on the bytes the arrays of ``carve`` take in the workspace, as a symbolic expression over the
    entry's scalar arguments: each array's bytes plus one :data:`WORKSPACE_ALIGNMENT` of padding; ``0`` for none."""
    return sum(desc.total_size * desc.dtype.bytes + WORKSPACE_ALIGNMENT for desc in carve.values())


def workspace_bytes(carve: dict[str, Any]) -> str:
    """:func:`workspace_bound` as the text ``Rendering.workspace_bytes`` holds (``'0'`` for none)."""
    return str(workspace_bound(carve)) if carve else "0"


def carve_into_workspace(name: str, desc: Any) -> bool:
    """Add device array ``name`` to the workspace layout in force when a :func:`no_boilerplate` entry allocates it, so
    its allocation becomes a :func:`workspace_slot`; whether it is placed.

    The GPU generator decides some of its buffers only while generating code (``MakeTransientsPersistent``), after the
    rendering planned the workspace, so a device entry grows the layout as it allocates; the caller has checked that
    the entry allocates the array once per call. A name another descriptor holds already keeps its allocation, since
    :func:`workspace_slot` matches the name and the descriptor together.
    """
    if not no_boilerplate() or not carved_workspace_name or carved_workspace_arrays.get(name, desc) is not desc:
        return False
    carved_workspace_arrays[name] = desc
    return True


def workspace_offset_name(workspace: str) -> str:
    """The running byte offset into ``workspace`` the entry carves its arrays at."""
    return f"{workspace}_off"


def workspace_offset_declaration() -> str:
    """``size_t <workspace>_off = 0;`` at the top of an entry that carves arrays out of its workspace, else ``''``;
    a :func:`no_boilerplate` entry also checks there that the workspace holds them."""
    if not standalone() or not carved_workspace_arrays:
        return ""
    declaration = f"size_t {workspace_offset_name(carved_workspace_name)} = 0;\n"
    if not no_boilerplate() or not carved_workspace_size_name:
        return declaration
    # A no-boilerplate entry refuses a workspace smaller than the bound it reports (``Rendering.workspace_bytes``),
    # before any slot is taken, rather than writing past the caller's buffer.
    from dace.codegen.common import sym2cpp  # the code generators import this module

    needed, size = sym2cpp(workspace_bound(carved_workspace_arrays)), carved_workspace_size_name
    return declaration + (
        f"if ({size} < (int64_t)({needed})) {{\n"
        f'fprintf(stderr, "{carved_workspace_entry}: workspace needs %lld bytes, got %lld\\n", '
        f"(long long)({needed}), (long long){size});\n"
        "abort();\n"
        "}\n"
    )


def workspace_slot(alloc_name: str, name: str | None, desc: Any, ctype: str, count: str) -> str | None:
    """The statement pointing ``alloc_name`` at the next free slot of the workspace and moving the running offset
    past it, rounded up to :data:`WORKSPACE_ALIGNMENT`, for a heap transient the rendering carves out of the
    workspace; ``None`` for any other array.

    Matched on the descriptor itself, so a nested SDFG's array of the same name keeps its own allocation. The
    rendering allocates every carved array once, at the top of the entry (``cpf.render_unit``), so each slot is
    taken exactly once per call.
    """
    if not standalone() or name is None or carved_workspace_arrays.get(name) is not desc:
        return None
    offset = workspace_offset_name(carved_workspace_name)
    align = WORKSPACE_ALIGNMENT
    return (
        f"{alloc_name} = ({ctype} *)({carved_workspace_name} + {offset}); "
        f"{offset} += (sizeof({ctype}) * (size_t)({count}) + {align - 1}) / {align} * {align};\n"
    )


#: The entry's persistent transients a standalone rendering keeps in ``static`` pointers, which survive the call
#: and grow only when a call needs more: data name -> descriptor. Empty outside :func:`persistent_static_scope`.
_persistent_static: dict[str, Any] = {}


@contextlib.contextmanager
def persistent_static_scope(persistent: dict[str, Any]):
    """Make ``persistent`` the ambient set of static persistent buffers for the block (:func:`persistent_static`)."""
    global _persistent_static
    previous = _persistent_static
    _persistent_static = persistent
    try:
        yield persistent
    finally:
        _persistent_static = previous


def persistent_static(name: str | None, desc: Any) -> str | None:
    """The ``static`` pointer that holds a persistent transient across calls, else ``None``."""
    if not standalone() or name is None or _persistent_static.get(name) is not desc:
        return None
    return f"__cpf_persistent_{name}"


def persistent_static_desc(desc: Any) -> bool:
    """Whether ``desc`` lives in a persistent ``static`` pointer, so freeing it at the end of a call frees nothing."""
    return standalone() and any(held is desc for held in _persistent_static.values())


def carved_from_workspace(desc: Any) -> bool:
    """Whether ``desc`` lives in the workspace, so freeing it frees nothing."""
    return standalone() and any(carved is desc for carved in carved_workspace_arrays.values())


#: Runtime function -> the ``std`` function with identical semantics and arity.
#:
#: ``ROUND`` is here rather than in :data:`REWRITES` because the runtime's ``ROUND`` is literally
#: ``return round(value);`` -- both round half away from zero, so the rename is exact.
STD_RENAMES: dict[str, str] = {
    "Abs": "std::abs",
    "abs": "std::abs",
    "ceiling": "std::ceil",
    "ceil": "std::ceil",
    "floor": "std::floor",
    "ROUND": "std::round",
    "round": "std::round",
    "conj": "std::conj",
    "exp2": "std::exp2",
    "expm1": "std::expm1",
    "log1p": "std::log1p",
    "log2": "std::log2",
    "frexp": "std::frexp",
    "ldexp": "std::ldexp",
    "ilogb": "std::ilogb",
    "isfinite": "std::isfinite",
    "isinf": "std::isinf",
    "isnan": "std::isnan",
    "signbit": "std::signbit",
    "gcd": "std::gcd",
    "lcm": "std::lcm",
    "sin": "std::sin",
    "cos": "std::cos",
    "tan": "std::tan",
    "asin": "std::asin",
    "acos": "std::acos",
    "atan": "std::atan",
    "atan2": "std::atan2",
    "sinh": "std::sinh",
    "cosh": "std::cosh",
    "tanh": "std::tanh",
    "exp": "std::exp",
    "fabs": "std::fabs",
    "log": "std::log",
    "log10": "std::log10",
    "sqrt": "std::sqrt",
    "cbrt": "std::cbrt",
    "pow": "std::pow",
    "fma": "std::fma",
    "erf": "std::erf",
    "erfc": "std::erfc",
    "tgamma": "std::tgamma",
    "lgamma": "std::lgamma",
    "trunc": "std::trunc",
    "hypot": "std::hypot",
}

#: Runtime function -> ``(arity, format string over the printed arguments)``.
#:
#: Every replacement parenthesizes each argument and the whole result: these strings are spliced
#: into a larger expression, and an unparenthesized ``a - b`` would rebind against a neighbouring
#: operator. Each argument appears EXACTLY ONCE -- a template repeating ``{0}`` would duplicate
#: whatever expression the caller printed, and anything needing its argument twice, or needing to
#: name its argument's type, is an inline definition instead (see :data:`INLINE_DEFINITIONS`).
#:
#: ``reciprocal`` keeps the runtime's integer-division behaviour for an integer argument: the
#: runtime is ``T(1) / a``, which for ``T = int`` truncates exactly as ``1 / (a)`` does here.
#: ``Mod`` is SymPy's floored modulo, so it is ``py_mod``.
REWRITES: dict[str, tuple[int, str]] = {
    "reciprocal": (1, "(1 / ({0}))"),
    # A complex's components, of a real operand too (``.real()`` on a ``double`` does not compile);
    # ``std::real`` / ``std::imag`` have that overload.
    "re": (1, "(std::real({0}))"),
    "im": (1, "(std::imag({0}))"),
    "iround": (1, "(static_cast<int>(std::round({0})))"),
    "ITE": (3, "(({0}) ? ({1}) : ({2}))"),
    "IfExpr": (3, "(({0}) ? ({1}) : ({2}))"),
    "deg2rad": (1, "(({0}) * 0.017453292519943295)"),
    "rad2deg": (1, "(({0}) * 57.29577951308232)"),
    "left_shift": (2, "(({0}) << ({1}))"),
    "right_shift": (2, "(({0}) >> ({1}))"),
    "bitwise_and": (2, "(({0}) & ({1}))"),
    "bitwise_or": (2, "(({0}) | ({1}))"),
    "bitwise_xor": (2, "(({0}) ^ ({1}))"),
    "bitwise_invert": (1, "(~({0}))"),
    "int_floor": (2, "(({0}) / ({1}))"),
    # sympy's ``Mod`` is floored, so it is ``py_mod`` and never C's truncating ``%``.
    "Mod": (2, "py_mod({0}, {1})"),
    "np_float_pow": (2, "(std::pow(static_cast<double>({0}), static_cast<double>({1})))"),
}

#: Runtime function -> the C++ definition CPF emits for it.
#:
#: Emitted only when the translation unit calls the helper (see :func:`definitions_for`), so a
#: kernel that never rounds up never carries an ``int_ceil``. Each is ``static`` so several CPF
#: translation units can be compiled together, and templated so it works for whatever width the
#: index arithmetic settled on. A helper needs a definition rather than a rewrite when it names its
#: argument's type, uses an argument more than once, dispatches on integral-vs-floating, or writes
#: through out-parameters.
#:
#: ``py_mod``/``ftn_modulo`` floor (``py_mod(-1, 5) == 4``); ``c_mod``/``ftn_mod`` truncate (``-1``).
#:
#: ``sign`` and ``heaviside`` are here rather than in :data:`REWRITES` for a dtype reason. The
#: runtime's ``sign`` is ``T((T(0) < x) - (x < T(0)))``: the comparisons yield ``bool``, their
#: difference is ``int``, and the cast back to ``T`` is what keeps ``sign(2.5)`` a ``double``.
#: Written as a textual rewrite the cast has no ``T`` to name, so the result would decay to ``int``
#: and a following ``sign(x) / 2`` would silently become integer division.
#:
#: ``cpp_divmod``, ``py_divmod``, ``np_modf`` and ``np_frexp`` return ``void`` and write through
#: reference out-parameters -- they are statements, not expressions, so they only ever needed a
#: definition.
#:
#: Every definition that CAN be evaluated at compile time is ``constexpr`` (never ``consteval``,
#: which would forbid the runtime calls that are the normal case). ``np_modf`` and ``np_frexp`` are
#: not: each reaches a standard function that is not ``constexpr``
#: before C++23 for every instantiation. Marking those ``constexpr`` anyway is ill-formed with no
#: diagnostic required, and it LOOKS fine -- GCC folds ``std::floor`` as a builtin and accepts it,
#: while clang rejects the same code. Measured, not assumed; see the constexpr probes in
#: ``tests/codegen/cpf/test_lowering_table.py``.
INLINE_DEFINITIONS: dict[str, str] = {
    "cpf_max": "template <typename T>\n"
    "static constexpr inline T cpf_max(const T& value) {\n"
    "    return value;\n"
    "}\n"
    "template <typename T, typename... Ts>\n"
    "static constexpr inline typename std::common_type<T, Ts...>::type cpf_max(const T& a, const Ts&... rest) {\n"
    "    return (a < cpf_max(rest...)) ? cpf_max(rest...) : a;\n"
    "}",
    "cpf_min": "template <typename T>\n"
    "static constexpr inline T cpf_min(const T& value) {\n"
    "    return value;\n"
    "}\n"
    "template <typename T, typename... Ts>\n"
    "static constexpr inline typename std::common_type<T, Ts...>::type cpf_min(const T& a, const Ts&... rest) {\n"
    "    return (cpf_min(rest...) < a) ? cpf_min(rest...) : a;\n"
    "}",
    "sign": "template <typename T>\n"
    "static constexpr inline T sign(const T& value) {\n"
    "    return T((T(0) < value) - (value < T(0)));\n"
    "}",
    "sgn": "template <typename T>\n"
    "static constexpr inline T sgn(const T& value) {\n"
    "    return T((T(0) < value) - (value < T(0)));\n"
    "}",
    "sign_numpy_2": "template <typename T>\n"
    "static constexpr inline T sign_numpy_2(const T& value) {\n"
    "    return T((T(0) < value) - (value < T(0)));\n"
    "}\n"
    "template <typename T>\n"
    "static inline std::complex<T> sign_numpy_2(const std::complex<T>& value) {\n"
    "    return (value.real() != 0 && value.imag() != 0) ? value / std::abs(value) : std::complex<T>(0, 0);\n"
    "}",
    "heaviside": "template <typename T>\n"
    "static constexpr inline T heaviside(const T& value, const T& at_zero) {\n"
    "    return (value < T(0)) ? T(0) : ((value > T(0)) ? T(1) : at_zero);\n"
    "}\n"
    "template <typename T>\n"
    "static constexpr inline T heaviside(const T& value) {\n"
    "    return (value > T(0)) ? T(1) : T(0);\n"
    "}",
    # ``ifloor`` is the UNARY floor-to-integer the tasklet printer emits for a floor division
    # (``dace::math::ifloor(a / b)``), not the binary ``int_floor``. Integral input is already
    # floored, so the runtime returns it unchanged -- and that identity case is why this is a
    # definition and not a rewrite: ``(int)std::floor(x)`` on an int64 would truncate it to 32 bits.
    "ifloor": "template <typename T>\n"
    "static constexpr inline std::conditional_t<std::is_integral_v<T>, T, int> ifloor(const T& value) {\n"
    "    if constexpr (std::is_integral_v<T>) {\n"
    "        return value;\n"
    "    } else {\n"
    "        return static_cast<int>(std::floor(value));\n"
    "    }\n"
    "}",
    "min_identity": "template <typename T>\n"
    "static inline T min_identity() {\n"
    "    return std::numeric_limits<T>::has_infinity ? T(std::numeric_limits<T>::infinity())\n"
    "                                                : std::numeric_limits<T>::max();\n"
    "}",
    "max_identity": "template <typename T>\n"
    "static inline T max_identity() {\n"
    "    return std::numeric_limits<T>::has_infinity ? T(-std::numeric_limits<T>::infinity())\n"
    "                                                : std::numeric_limits<T>::lowest();\n"
    "}",
    # find-first
    # An early-exit loop lifts to a ``FindFirst`` library node whose expansion calls the runtime's
    # short-circuiting parallel search. CPF emits that search rather than unrolling it back into a
    # sequential scan, for the same reason it emits the inscan form of a prefix sum: the cancelling
    # parallel shape IS the rendering, and serializing it would answer a different question.
    "find_first_chunk": "#ifdef _OPENMP\n"
    "#include <omp.h>\n"
    "#endif\n"
    "static inline long long find_first_chunk(long long span, bool parallel) {\n"
    "    // Grows as sqrt(span): too big a chunk scans past the answer on one thread, too small a\n"
    "    // one pays dispatch on chunks the answer makes dead. The floor binds below ~64k elements.\n"
    "    constexpr double chunk_scale = 8.0;\n"
    "    constexpr long long chunks_per_thread = 4;\n"
    "    long long chunk = (long long)(chunk_scale * std::sqrt((double)span));\n"
    "    long long threads = 1;\n"
    "#ifdef _OPENMP\n"
    "    if (parallel) threads = (long long)omp_get_max_threads();\n"
    "#endif\n"
    "    long long ceiling = span / (chunks_per_thread * threads);\n"
    "    if (ceiling < 1) ceiling = 1;\n"
    "    if (chunk > ceiling) chunk = ceiling;\n"
    "    if (chunk < 1) chunk = 1;\n"
    "    return chunk;\n"
    "}",
    "find_first_index": "template <typename Pred>\n"
    "static inline long long find_first_index(long long begin, long long end, Pred pred, bool parallel) {\n"
    "    // The answer is a min-reduction and is exact; the hint is shared and races by design --\n"
    "    // every value it takes is a real firing index, so a lost update costs pruning, never\n"
    "    // correctness. Folding the two into one word is exactly that lost-update bug.\n"
    "    constexpr long long simd_block = 64;\n"
    "    if (begin >= end) return end;\n"
    "    const long long span = end - begin;\n"
    "    const long long chunk = find_first_chunk(span, parallel);\n"
    "    const long long nchunks = (span + chunk - 1) / chunk;\n"
    "    long long best = end;\n"
    "    long long hint = end;\n"
    "    #pragma omp parallel for schedule(dynamic, 1) if (parallel : parallel) reduction(min : best)\n"
    "    for (long long c = 0; c < nchunks; ++c) {\n"
    "        long long seen;\n"
    "        #pragma omp atomic read\n"
    "        seen = hint;\n"
    "        const long long lo = begin + c * chunk;\n"
    "        if (lo >= seen) continue;\n"
    "        long long hi = lo + chunk;\n"
    "        if (hi > end) hi = end;\n"
    "        if (hi > seen) hi = seen;\n"
    "        long long found = end;\n"
    "        for (long long b = lo; b < hi; b += simd_block) {\n"
    "            long long be = b + simd_block;\n"
    "            if (be > hi) be = hi;\n"
    "            long long block = end;\n"
    "            // A vectorized loop cannot break, so the block is the early-exit granularity.\n"
    "            #pragma omp simd reduction(min : block)\n"
    "            for (long long i = b; i < be; ++i) {\n"
    "                const long long v = pred(i) ? i : end;\n"
    "                block = v < block ? v : block;\n"
    "            }\n"
    "            if (block < end) { found = block; break; }\n"
    "        }\n"
    "        if (found < end) {\n"
    "            if (found < best) best = found;\n"
    "            long long cur;\n"
    "            #pragma omp atomic read\n"
    "            cur = hint;\n"
    "            if (found < cur) {\n"
    "                #pragma omp atomic write\n"
    "                hint = found;\n"
    "            }\n"
    "        }\n"
    "    }\n"
    "    return best;\n"
    "}",
    "detect_collision": "// Tagged-write + verify, the two-pass duplicate detection the scatter guard runs. Pass 1\n"
    "// writes owner[idx[i]] = i; pass 2 ORs owner[idx[i]] != i. A slot only one writer won reads\n"
    "// back a different i, so the OR trips. Pass 2 reads only slots pass 1 wrote, so the tag\n"
    "// buffer needs no initialization and none between calls. Values outside [0, capacity) are\n"
    "// skipped: the guarded scatter never writes those slots, so they can neither collide nor\n"
    "// mask a collision.\n"
    "#ifdef _OPENMP\n"
    "#include <omp.h>\n"
    "#endif\n"
    "template <typename T, typename TagT>\n"
    "static inline long long detect_collision(const T *idx, long long n, TagT *owner, long long capacity,\n"
    "                                         bool parallel) {\n"
    "#ifdef _OPENMP\n"
    "    if (parallel && omp_get_max_threads() == 1) return 1;  // one thread: in order is correct, skip the check\n"
    "#else\n"
    "    if (parallel) return 1;\n"
    "#endif\n"
    "    long long c = 0;\n"
    "    #pragma omp parallel for if (parallel : parallel)\n"
    "    for (long long i = 0; i < n; ++i) {\n"
    "        const long long v = (long long)idx[i];\n"
    "        if (v >= 0 && v < capacity) owner[v] = (TagT)i;\n"
    "    }\n"
    "    #pragma omp parallel for simd if (parallel : parallel) reduction(| : c)\n"
    "    for (long long i = 0; i < n; ++i) {\n"
    "        const long long v = (long long)idx[i];\n"
    "        if (v >= 0 && v < capacity) c |= ((long long)owner[v] != i) ? 1LL : 0LL;\n"
    "    }\n"
    "    return c;\n"
    "}\n"
    "// The same check sizing its own tag buffer from max(idx). A failed allocation reports a\n"
    "// collision, which is the answer that is never unsafe: the guard then takes its sequential\n"
    "// path instead of parallelizing a scatter nothing proved conflict-free.\n"
    "template <typename T>\n"
    "static inline long long detect_collision(const T *idx, long long n, bool parallel) {\n"
    "#ifdef _OPENMP\n"
    "    if (parallel && omp_get_max_threads() == 1) return 1;  // one thread: in order is correct, skip the check\n"
    "#else\n"
    "    if (parallel) return 1;\n"
    "#endif\n"
    "    long long mx = 0;\n"
    "    #pragma omp parallel for simd if (parallel : parallel) reduction(max : mx)\n"
    "    for (long long i = 0; i < n; ++i) {\n"
    "        const long long v = (long long)idx[i];\n"
    "        mx = v > mx ? v : mx;\n"
    "    }\n"
    "    long long *owner = (long long *)malloc((size_t)(mx + 1) * sizeof(long long));\n"
    "    if (owner == nullptr) return 1;\n"
    "    const long long c = detect_collision(idx, n, owner, mx + 1, parallel);\n"
    "    free(owner);\n"
    "    return c;\n"
    "}",
    "int_ceil": "template <typename T, typename U, typename R = decltype(std::declval<T>() / std::declval<U>())>\n"
    "static constexpr inline R int_ceil(const T& numerator, const U& denominator) {\n"
    "    return (numerator + denominator - 1) / denominator;\n"
    "}",
    "int_sqrt": "template <typename T>\n"
    "static constexpr inline T int_sqrt(const T& n) {\n"
    "    if (n < 2) return n < 0 ? T(-1) : n;\n"
    "    T x = n, y = (x + 1) / 2;\n"
    "    while (y < x) {\n"
    "        x = y;\n"
    "        y = (x + n / x) / 2;\n"
    "    }\n"
    "    return x;\n"
    "}",
    "int_floor_ni": "template <typename T, typename U, typename R = decltype(std::declval<T>() / std::declval<U>())>\n"
    "static constexpr inline R int_floor_ni(const T& numerator, const U& denominator) {\n"
    "    R quotient = numerator / denominator;\n"
    "    R remainder = numerator % denominator;\n"
    "    return quotient - ((remainder != 0) && ((remainder < 0) != (denominator < 0)));\n"
    "}",
    "py_floor": "template <typename T, typename U, typename R = decltype(std::declval<T>() / std::declval<U>())>\n"
    "static constexpr inline R py_floor(const T& numerator, const U& denominator) {\n"
    "    R quotient = 0;\n"
    "    R remainder = 0;\n"
    "    py_divmod(static_cast<R>(numerator), static_cast<R>(denominator), quotient, remainder);\n"
    "    return quotient;\n"
    "}",
    "py_mod": "template <typename T, typename U, typename R = decltype(std::declval<T>() / std::declval<U>())>\n"
    "static constexpr inline R py_mod(const T& numerator, const U& denominator) {\n"
    "    R quotient = 0;\n"
    "    R remainder = 0;\n"
    "    py_divmod(static_cast<R>(numerator), static_cast<R>(denominator), quotient, remainder);\n"
    "    return remainder;\n"
    "}",
    "c_mod": "template <typename T, typename U, typename R = std::conditional_t<std::is_integral_v<T> && std::is_integral_v<U>, decltype(std::declval<T>() / std::declval<U>()), decltype(std::fmod(std::declval<T>(), std::declval<U>()))>>\n"
    "static constexpr inline R c_mod(const T& numerator, const U& denominator) {\n"
    "    if constexpr (std::is_integral_v<T> && std::is_integral_v<U>) {\n"
    "        return numerator % denominator;\n"
    "    } else {\n"
    "        return std::fmod(numerator, denominator);\n"
    "    }\n"
    "}",
    "ftn_mod": "template <typename T, typename U, typename R = decltype(c_mod(std::declval<T>(), std::declval<U>()))>\n"
    "static constexpr inline R ftn_mod(const T& numerator, const U& denominator) {\n"
    "    return c_mod(numerator, denominator);\n"
    "}",
    "ftn_modulo": "template <typename T, typename U, typename R = decltype(py_mod(std::declval<T>(), std::declval<U>()))>\n"
    "static constexpr inline R ftn_modulo(const T& numerator, const U& denominator) {\n"
    "    return py_mod(numerator, denominator);\n"
    "}",
    "cpp_divmod": "template <typename T>\n"
    "static constexpr inline void cpp_divmod(const T& numerator, const T& denominator, T& quotient,\n"
    "                                        T& remainder) {\n"
    "    quotient = static_cast<T>(numerator / denominator);\n"
    "    remainder = static_cast<T>(numerator % denominator);\n"
    "}",
    "py_divmod": "template <typename T>\n"
    "static constexpr inline void py_divmod(const T& numerator, const T& denominator, T& quotient,\n"
    "                                       T& remainder) {\n"
    "    if constexpr (std::is_floating_point_v<T>) {\n"
    "        const T ratio = numerator / denominator;\n"
    "        remainder = std::fmod(numerator, denominator);\n"
    "        if (denominator == 0) {\n"
    "            quotient = ratio;\n"
    "            return;\n"
    "        }\n"
    "        quotient = (numerator - remainder) / denominator;\n"
    "        if (remainder == 0) {\n"
    "            remainder = std::copysign(static_cast<T>(0), denominator);\n"
    "        } else if ((denominator < 0) != (remainder < 0)) {\n"
    "            remainder += denominator;\n"
    "            quotient -= 1;\n"
    "        }\n"
    "        if (quotient == 0) {\n"
    "            quotient = std::copysign(static_cast<T>(0), ratio);\n"
    "        } else {\n"
    "            const T floored = std::floor(quotient);\n"
    "            quotient = quotient - floored > static_cast<T>(0.5) ? floored + 1 : floored;\n"
    "        }\n"
    "    } else if (denominator == 0) {\n"
    "        quotient = 0;\n"
    "        remainder = 0;\n"
    "    } else if (numerator == std::numeric_limits<T>::min() && denominator == static_cast<T>(-1)) {\n"
    "        quotient = numerator;\n"
    "        remainder = 0;\n"
    "    } else {\n"
    "        quotient = static_cast<T>(numerator / denominator);\n"
    "        remainder = static_cast<T>(numerator % denominator);\n"
    "        if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {\n"
    "            quotient = static_cast<T>(quotient - 1);\n"
    "            remainder = static_cast<T>(remainder + denominator);\n"
    "        }\n"
    "    }\n"
    "}",
    "np_modf": "template <typename T>\n"
    "static inline void np_modf(const T& value, T& integral, T& fractional) {\n"
    "    if constexpr (std::is_integral_v<T>) {\n"
    "        integral = value;\n"
    "        fractional = T(0);\n"
    "    } else {\n"
    "        fractional = std::modf(value, &integral);\n"
    "    }\n"
    "}",
    "np_frexp": "template <typename T>\n"
    "static inline void np_frexp(const T& value, T& mantissa, int& exponent) {\n"
    "    mantissa = std::frexp(value, &exponent);\n"
    "}",
    "ipow": "template <typename T, typename U>\n"
    "static constexpr inline T ipow(T base, U exponent) {\n"
    "    T result = 1;\n"
    "    while (exponent > 0) {\n"
    "        if (exponent & 1) { result *= base; }\n"
    "        base *= base;\n"
    "        exponent >>= 1;\n"
    "    }\n"
    "    return result;\n"
    "}",
    "logical_left_shift": "template <typename T, typename U>\n"
    "static constexpr inline T logical_left_shift(const T& value, const U& amount) {\n"
    "    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) << amount);\n"
    "}",
    "logical_right_shift": "template <typename T, typename U>\n"
    "static constexpr inline T logical_right_shift(const T& value, const U& amount) {\n"
    "    return static_cast<T>(static_cast<std::make_unsigned_t<T>>(value) >> amount);\n"
    "}",
}

#: Per scan operation: the OpenMP reduction identifier and how ``a`` folds in ``x`` in C++ and in C, with
#: ``{T}`` the accumulator type. ``min`` / ``max`` cast the input to ``{T}`` so the comparison happens where
#: the fold does; in C they keep ``a`` on a tie or a NaN, the rule of :data:`C_MINMAX_CONDITIONS`.
SCAN_OPERATIONS: dict[str, tuple[str, str, str]] = {
    "sum": ("+", "{a} + {x}", "{a} + {x}"),
    "product": ("*", "{a} * {x}", "{a} * {x}"),
    "min": ("min", "cpf_min({a}, static_cast<{T}>({x}))", "(({T}){x} < {a}) ? ({T}){x} : {a}"),
    "max": ("max", "cpf_max({a}, static_cast<{T}>({x}))", "({a} < ({T}){x}) ? ({T}){x} : {a}"),
}


def blocked_scan_body(
    kind: str,
    operation: str,
    ctype: str,
    c_dialect: bool,
    element: str = "f[{i}]",
    store: str = "o[{i}] = acc;",
    element_bytes: str | None = None,
) -> list[str]:
    """The body of one prefix scan over ``o[lo:hi]``: the tiled two-pass OpenMP scan of
    ``dace::scan::detail::blocked_scan``.

    One parallel region walks the range in tiles of ``team`` blocks of at most 512 KB each, so a block's second
    pass re-reads it from cache. Per tile, each thread folds its block (vectorized), one barrier publishes the
    totals (two banks alternate by tile, so no second barrier is needed), and each thread rescans its block from
    the combined offset. The rescan is a ``simd inscan`` loop on x86-64 and a scalar loop elsewhere, where the
    ``inscan`` loop measured slower. Inside a parallel region, or with one thread, it is the rescan alone.
    ``ctype`` is the accumulator (seed) type. ``element`` is the scanned value at position ``{i}`` and ``store``
    the statement run with the prefix ``acc`` at position ``{i}``: the defaults read ``f`` and write ``o``; a fused
    producer or consumer replaces them (:mod:`~dace.transformation.passes.cpu_specialization.fuse_scan_chains`).
    ``element_bytes`` is what one position reads from memory (the input's element size by default): a block holds
    512 KB of it, so a producer reading three arrays gets a third of the elements per block.

    """
    clause, cpp_step, c_step = SCAN_OPERATIONS[operation]
    step = c_step if c_dialect else cpp_step

    def fold(a: str, x: str) -> str:
        return step.format(a=a, x=x, T=ctype)

    directive = "inclusive" if kind == "incl" else "exclusive"

    def rescan(indent: str, first: str, last: str) -> list[str]:
        update = indent + "    acc = %s;" % fold("acc", element.format(i="i"))
        written = indent + "    " + store.format(i="i")
        phases = (update, written) if kind == "incl" else (written, update)
        return [
            "#if defined(__x86_64__) || defined(_M_X64)",
            indent + "#pragma omp simd reduction(inscan, %s:acc)" % clause,
            indent + "for (long i = %s; i < %s; ++i) {" % (first, last),
            phases[0],
            indent + "    #pragma omp scan %s(acc)" % directive,
            phases[1],
            indent + "}",
            "#else",
            indent + "for (long i = %s; i < %s; ++i) {" % (first, last),
            phases[0],
            phases[1],
            indent + "}",
            "#endif",
        ]

    first = element.format(i="b")
    cast = "(%s)%s" % (ctype, first) if c_dialect else "static_cast<%s>(%s)" % (ctype, first)
    alloc = (
        "(%s *)malloc(sizeof(%s) * (size_t)(2 * team))" % (ctype, ctype) if c_dialect else "new %s[2 * team]" % ctype
    )
    flags = "(char *)malloc((size_t)(2 * team))" if c_dialect else "new char[2 * team]"
    release = "free(totals); free(filled);" if c_dialect else "delete[] totals; delete[] filled;"
    per_element = element_bytes if element_bytes is not None else "sizeof(%s)" % ctype
    return [
        "    long team = 1;",
        "#ifdef _OPENMP",
        "    if (!omp_in_parallel()) team = (long)omp_get_max_threads();",
        "#endif",
        "    if (team > hi - lo) team = hi - lo;",
        "    if (team <= 1) {",
        "        %s acc = seed;" % ctype,
        *rescan("        ", "lo", "hi"),
        "        return;",
        "    }",
        "#ifdef _OPENMP",
        "    %s *totals = %s;" % (ctype, alloc),
        "    char *filled = %s;" % flags,
        "    const long span = (long)((1L << 19) / (%s)) > 0 ? (long)((1L << 19) / (%s)) : 1;"
        % (per_element, per_element),
        "    #pragma omp parallel num_threads(team)",
        "    {",
        "        const long size = (long)omp_get_num_threads();",
        "        const long me = (long)omp_get_thread_num();",
        "        const long even = (hi - lo + size - 1) / size;",
        "        const long per = span < even ? span : even;",
        "        const long tile = per * size;",
        "        %s carry = seed;" % ctype,
        "        long bank = 0;",
        "        for (long base = lo; base < hi; base += tile, bank ^= 1) {",
        "            const long end = base + tile < hi ? base + tile : hi;",
        "            const long b = base + me * per < end ? base + me * per : end;",
        "            const long e = b + per < end ? b + per : end;",
        "            filled[bank * size + me] = (char)(b < e);",
        "            if (b < e) {",
        "                %s total = %s;" % (ctype, cast),
        "                #pragma omp simd reduction(%s:total)" % clause,
        "                for (long i = b + 1; i < e; ++i) total = %s;" % fold("total", element.format(i="i")),
        "                totals[bank * size + me] = total;",
        "            }",
        "            #pragma omp barrier",
        "            for (long q = 0; q < size; ++q) {",
        "                if (q == me && b < e) {",
        "                    %s acc = carry;" % ctype,
        *rescan("                    ", "b", "e"),
        "                }",
        "                if (filled[bank * size + q]) carry = %s;" % fold("carry", "totals[bank * size + q]"),
        "            }",
        "        }",
        "    }",
        "    %s" % release,
        "#endif",
    ]


def cpp_scan_definition(kind: str, operation: str) -> str:
    """One prefix scan as a C++ template over ``<It, OutIt, T>``, the shape ``dace/scan.hpp`` gives it."""
    return "\n".join(
        [
            "#ifdef _OPENMP",
            "#include <omp.h>",
            "#endif",
            "template <typename It, typename OutIt, typename T>",
            "static inline void scan_%s_%s(It f, OutIt o, long lo, long hi, T seed) {" % (kind, operation),
            *blocked_scan_body(kind, operation, "T", c_dialect=False),
            "}",
        ]
    )


# prefix scans
# The DaCe runtime provides these in ``dace/scan.hpp``, one function per (op, inclusive) pair because an
# OpenMP reduction identifier cannot be a template parameter -- the operator has to be spelled into the
# clause. CPF reproduces them, blocking included: the ``inscan`` loop alone is one thread.
INLINE_DEFINITIONS.update(
    {
        f"scan_{kind}_{operation}": cpp_scan_definition(kind, operation)
        for kind in ("incl", "excl")
        for operation in SCAN_OPERATIONS
    }
)

#: Per ``dace::reduce`` entry point: the OpenMP reduction identifier and how ``a`` folds in ``x`` in C++ and in C,
#: as :data:`SCAN_OPERATIONS` spells the four a scan shares. The ``Reduce`` CPU lowering emits nothing else.
REDUCE_OPERATIONS: dict[str, tuple[str, str, str]] = {
    **SCAN_OPERATIONS,
    "bitwise_and": ("&", "{a} & {x}", "{a} & {x}"),
    "bitwise_or": ("|", "{a} | {x}", "{a} | {x}"),
    "bitwise_xor": ("^", "{a} ^ {x}", "{a} ^ {x}"),
    "logical_and": ("&&", "{a} && {x}", "{a} && {x}"),
    "logical_or": ("||", "{a} || {x}", "{a} || {x}"),
}


#: A complex sum or product reduces through a combiner it declares itself: OpenMP has no complex reduction, and C
#: refuses a ``declare reduction`` of ``+`` / ``*`` over its predeclared ``_Complex`` types. Identifier, combining
#: operator and identity, as the CPU code generator spells them.
COMPLEX_REDUCE_COMBINERS: dict[str, tuple[str, str, int]] = {
    "sum": ("cpf_cadd", "+=", 0),
    "product": ("cpf_cmul", "*=", 1),
}


def reduce_body(
    operation: str, ctype: str, parallel: bool, c_dialect: bool, complex_initial: str = "", simd: bool = False
) -> list[str]:
    """The body of one ``dace::reduce`` entry: fold ``count`` inputs ``stride`` apart into ``seed``, under an OpenMP
    reduction unless it is the ``seq`` entry the expansion calls from inside a parallel scope. ``complex_initial``
    spells the identity of a complex accumulator, which then reduces through a block-scope declared combiner.

    ``simd`` adds the ``simd`` modifier (the ``seq`` entry gets ``omp simd`` alone) and a unit-stride copy of the loop.
    A reduction clause lets the compiler reassociate the fold; without it a strict-FP build keeps the source's order
    and runs the sum scalar, slower than a spelling with split accumulators (E5: 2.6x on ``sum`` at one thread)."""
    clause, cpp_step, c_step = REDUCE_OPERATIONS[operation]
    step = c_step if c_dialect else cpp_step
    declare: list[str] = []
    if complex_initial:
        clause, combine, identity = COMPLEX_REDUCE_COMBINERS[operation]
        declare = [
            "    #pragma omp declare reduction(%s : %s : omp_out %s omp_in) initializer(omp_priv = %s)"
            % (clause, ctype, combine, complex_initial.format(identity=identity))
        ]
    modifier = "parallel for simd" if parallel and simd else "parallel for" if parallel else "simd" if simd else ""
    pragma = ["    #pragma omp %s reduction(%s:acc)" % (modifier, clause)] if modifier else []

    def loop(element: str, indent: str = "    ") -> list[str]:
        return [indent + line for line in pragma] + [
            indent + "    for (long i = 0; i < count; ++i) acc = %s;" % step.format(a="acc", x=element, T=ctype)
        ]

    if not simd:
        return [*declare, "    %s acc = seed;" % ctype, *loop("in[i * stride]", ""), "    return acc;"]
    return [
        *declare,
        "    %s acc = seed;" % ctype,
        "    if (stride == 1) {",
        *loop("in[i]"),
        "    } else {",
        *loop("in[i * stride]"),
        "    }",
        "    return acc;",
    ]


def reduce_name(operation: str, parallel: bool) -> str:
    return "reduce_%s%s" % ("" if parallel else "seq_", operation)


def cpp_reduce_definition(operation: str, parallel: bool) -> str:
    """One ``dace::reduce`` entry as a C++ template over the input type ``T`` and the seed type ``S``; a parallel sum
    or product adds the ``std::complex`` overload that declares its combiner."""
    name = reduce_name(operation, parallel)
    lines = [
        "template <typename T, typename S>",
        "static inline S %s(const T *in, long count, long stride, S seed) {" % name,
        *reduce_body(operation, "S", parallel, c_dialect=False),
        "}",
    ]
    if parallel and operation in COMPLEX_REDUCE_COMBINERS:
        lines += [
            "template <typename T, typename V>",
            "static inline std::complex<V> %s(const T *in, long count, long stride, std::complex<V> seed) {" % name,
            *reduce_body(
                operation, "std::complex<V>", parallel, c_dialect=False, complex_initial="std::complex<V>({identity})"
            ),
            "}",
        ]
    return "\n".join(lines)


# reductions
# The DaCe runtime provides these in ``dace/reduction.h``; CPF spells each one out with its OpenMP clause, so a
# ``Reduce`` renders through its parallel CPU lowering instead of ``pure``'s per-element critical section.
INLINE_DEFINITIONS.update(
    {
        reduce_name(operation, parallel): cpp_reduce_definition(operation, parallel)
        for parallel in (True, False)
        for operation in REDUCE_OPERATIONS
    }
)

# argmax / argmin
# ``dace/arg_reduction.h`` declares the (value, index) pair and its ``arg_max_pair`` / ``arg_min_pair`` OpenMP
# reductions; the ``ArgReduce`` CPU lowering names them. CPF carries the same text without the namespace.
ARG_VALUE_CTYPES: tuple[str, ...] = (
    "float",
    "double",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "std::complex<float>",
    "std::complex<double>",
)
ARG_INDEX_CTYPES: tuple[str, ...] = ("int32_t", "int64_t")


def arg_step_cpp(op: str, last: bool) -> str:
    """``arg_<op>[_last]_step``: a better value, or with ``last`` an equal one too, replaces the held pair."""
    better = "arg_greater(v, held.__ar_v)" if op == "max" else "arg_greater(held.__ar_v, v)"
    name = f"arg_{op}_last_step" if last else f"arg_{op}_step"
    take = f"{better} || v == held.__ar_v" if last else better
    return (
        "template <typename V, typename I>\n"
        f"static inline arg_pair<V, I> {name}(const arg_pair<V, I> &held, const V &v, I i) {{\n"
        f"    return {take} ? arg_pair<V, I>{{v, i}} : held;\n}}"
    )


ARG_REDUCTION_CPP: dict[str, str] = {
    "arg_pair": (
        "template <typename V, typename I>\n"
        "struct arg_pair {\n"
        "    V __ar_v;\n"
        "    I __ar_i;\n"
        "};\n"
        "template <typename V, typename = void>\n"
        "struct arg_is_complex : std::false_type {};\n"
        "template <typename V>\n"
        "struct arg_is_complex<V, std::void_t<decltype(std::declval<V>().imag())>> : std::true_type {};\n"
        "template <typename V>\n"
        "static inline bool arg_greater(const V &a, const V &b) {\n"
        "    if constexpr (arg_is_complex<V>::value) {\n"
        "        return a.real() > b.real() || (a.real() == b.real() && a.imag() > b.imag());\n"
        "    } else {\n"
        "        return a > b;\n"
        "    }\n"
        "}\n"
        "template <bool Max, bool Last, typename V, typename I>\n"
        "static inline arg_pair<V, I> arg_combine(const arg_pair<V, I> &out, const arg_pair<V, I> &in) {\n"
        "    if (!(out.__ar_v == out.__ar_v) || !(in.__ar_v == in.__ar_v))\n"
        "        return out.__ar_v == out.__ar_v ? in : out;\n"
        "    const bool better = Max ? arg_greater(in.__ar_v, out.__ar_v) : arg_greater(out.__ar_v, in.__ar_v);\n"
        "    const bool wins_tie = Last ? in.__ar_i > out.__ar_i : in.__ar_i < out.__ar_i;\n"
        "    return (better || (in.__ar_v == out.__ar_v && wins_tie)) ? in : out;\n"
        "}"
    ),
    "arg_pair_of": "template <typename V, typename I>\n"
    "static inline arg_pair<V, I> arg_pair_of(const V &v, I i) { return arg_pair<V, I>{v, i}; }",
    **{
        f"arg_{op}{'_last' if last else ''}_step": arg_step_cpp(op, last)
        for last in (False, True)
        for op in ("max", "min")
    },
    # One entry per merge, as per step: the renderer strips ``dace::`` from, and defines, only names it has an entry for.
    **{
        f"arg_{op}{'_last' if last else ''}_combine": "template <typename V, typename I>\n"
        f"static inline arg_pair<V, I> arg_{op}{'_last' if last else ''}_combine("
        "const arg_pair<V, I> &out, const arg_pair<V, I> &in) {\n"
        f"    return arg_combine<{str(op == 'max').lower()}, {str(last).lower()}>(out, in);\n"
        "}"
        for last in (False, True)
        for op in ("max", "min")
    },
}
INLINE_DEFINITIONS.update(ARG_REDUCTION_CPP)


def atomic_minmax_body(op: str, ctype: str) -> list[str]:
    """``*x = max(*x, v)`` (or min) as one atomic: OpenMP 5.1 ``atomic compare``, which NVHPC does not parse, so
    there the GNU compare-and-swap builtin loops instead. Same rule as ``cpf_max`` / ``cpf_min``: a NaN never wins."""
    better = "*x < v" if op == "max" else "v < *x"
    seen_better = "seen < v" if op == "max" else "v < seen"
    return [
        "#if defined(__NVCOMPILER)",
        f"    {ctype} seen;",
        "    __atomic_load(x, &seen, __ATOMIC_RELAXED);",
        f"    while ({seen_better} && !__atomic_compare_exchange(x, &seen, &v, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {{}}",
        "#else",
        "    #pragma omp atomic compare",
        f"    if ({better}) {{ *x = v; }}",
        "#endif",
    ]


#: A conflicting min / max accumulation (:meth:`~dace.codegen.targets.cpu.CPUCodeGen.standalone_atomic_wcr`).
INLINE_DEFINITIONS.update(
    {
        f"cpf_atomic_{op}": "\n".join(
            [
                "template <typename T, typename U>",
                f"static inline void cpf_atomic_{op}(T *x, U value) {{",
                "    T v = static_cast<T>(value);",
                *atomic_minmax_body(op, "T"),
                "}",
            ]
        )
        for op in ("max", "min")
    }
)

#: Definitions each definition calls. Emission is dependency-first (see :func:`definitions_for`).

#: The float math the standard library spells, run on a 16-bit float through ``float``. ``std::exp(__bf16)`` is
#: ambiguous between the ``float`` / ``double`` / ``long double`` overloads -- with Clang 22 in every standard,
#: with GCC before C++23 -- so a host unit that holds a ``__bf16`` or ``_Float16`` calls these instead
#: (:func:`rewrite_ctypes`). Any other argument type goes straight to the ``std::`` function.
LOW_PRECISION_MATH: tuple[str, ...] = (
    "abs",
    "acos",
    "asin",
    "atan",
    "atan2",
    "cbrt",
    "ceil",
    "cos",
    "cosh",
    "erf",
    "erfc",
    "exp",
    "exp2",
    "expm1",
    "fabs",
    "floor",
    "fma",
    "hypot",
    "isfinite",
    "isinf",
    "isnan",
    "ldexp",
    "lgamma",
    "log",
    "log10",
    "log1p",
    "log2",
    "pow",
    "round",
    "signbit",
    "sin",
    "sinh",
    "sqrt",
    "tan",
    "tanh",
    "tgamma",
    "trunc",
)
INLINE_DEFINITIONS["cpf_lowp_base"] = """\
//: A 16-bit float (``__bf16``, ``_Float16``) computes its math in ``float`` and stores back.
template <typename T>
inline constexpr bool cpf_lowp_narrow =
    std::is_same_v<std::remove_cv_t<T>, __bf16> || std::is_same_v<std::remove_cv_t<T>, _Float16>;
template <typename T>
using cpf_lowp_wide_t = std::conditional_t<cpf_lowp_narrow<T>, float, T>;
template <typename T>
static inline cpf_lowp_wide_t<T> cpf_lowp_widen(const T &value) {
    return static_cast<cpf_lowp_wide_t<T>>(value);
}
"""
INLINE_DEFINITIONS.update(
    {
        f"cpf_lowp_{name}": f"""\
//: ``std::{name}`` run in ``float`` on a 16-bit float, its floating result stored back in the operands' type.
template <typename... A>
using cpf_lowp_{name}_t = std::conditional_t<
    (cpf_lowp_narrow<A> || ...) &&
        std::is_floating_point_v<decltype(std::{name}(std::declval<cpf_lowp_wide_t<A>>()...))>,
    std::common_type_t<A...>, decltype(std::{name}(std::declval<cpf_lowp_wide_t<A>>()...))>;
template <typename... A>
static inline cpf_lowp_{name}_t<A...> cpf_lowp_{name}(const A &...a) {{
    return static_cast<cpf_lowp_{name}_t<A...>>(std::{name}(cpf_lowp_widen(a)...));
}}
"""
        for name in LOW_PRECISION_MATH
    }
)

#: ``std::<math>(`` in a host unit that holds a 16-bit float, respelled through :data:`LOW_PRECISION_MATH`'s wrappers.
LOW_PRECISION_MATH_CALL: "re.Pattern" = re.compile(
    r"(?<![\w:.])(?:::)?std::(" + "|".join(LOW_PRECISION_MATH) + r")\s*\("
)

DEFINITION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    **{f"cpf_lowp_{name}": ("cpf_lowp_base",) for name in LOW_PRECISION_MATH},
    "arg_pair_of": ("arg_pair",),
    **{
        f"arg_{op}{last}_{piece}": ("arg_pair",)
        for op in ("max", "min")
        for last in ("", "_last")
        for piece in ("step", "combine")
    },
    **{
        reduce_name(operation, parallel): (f"cpf_{operation}",)
        for operation in ("min", "max")
        for parallel in (True, False)
    },
    "scan_incl_min": ("min_identity", "cpf_min"),
    "scan_incl_max": ("max_identity", "cpf_max"),
    "scan_excl_min": ("min_identity", "cpf_min"),
    "scan_excl_max": ("max_identity", "cpf_max"),
    "find_first_index": ("find_first_chunk",),
    "py_floor": ("py_divmod",),
    "py_mod": ("py_divmod",),
    "ftn_mod": ("c_mod",),
    "ftn_modulo": ("py_mod",),
}

#: System headers each inline definition needs, beyond :data:`BASE_HEADERS`.
DEFINITION_HEADERS: dict[str, tuple[str, ...]] = {
    "cpf_lowp_base": ("<type_traits>",),
    **{f"cpf_lowp_{name}": ("<cmath>", "<type_traits>", "<utility>") for name in LOW_PRECISION_MATH},
    "arg_pair": ("<complex>", "<limits>", "<type_traits>", "<utility>"),
    **{reduce_name(operation, True): ("<complex>",) for operation in COMPLEX_REDUCE_COMBINERS},
    "ifloor": ("<type_traits>",),
    "cpf_narrow": ("<type_traits>",),
    "min_identity": ("<limits>",),
    "max_identity": ("<limits>",),
    "logical_left_shift": ("<type_traits>",),
    "logical_right_shift": ("<type_traits>",),
    "int_ceil": ("<utility>",),
    "int_floor_ni": ("<utility>",),
    "py_floor": ("<utility>",),
    "py_mod": ("<utility>",),
    "py_divmod": ("<limits>", "<type_traits>"),
    "c_mod": ("<type_traits>", "<utility>"),
    "ftn_mod": ("<utility>",),
    "ftn_modulo": ("<utility>",),
    "np_modf": ("<type_traits>",),
}

#: ``Max``/``Min`` are variadic in the runtime, which ``std::max``/``std::min`` are not: those are
#: binary or take an ``initializer_list``. The ORDER now matches (a later argument wins only by
#: comparing strictly better), so the difference is arity and mixed-type promotion, but CPF still
#: emits the runtime's own definition (see :data:`INLINE_DEFINITIONS`) so the two cannot drift.
VARIADIC_MINMAX: dict[str, str] = {"Max": "cpf_max", "Min": "cpf_min", "max": "cpf_max", "min": "cpf_min"}

#: Headers CPF always includes: the exact-width integer types and the maths every kernel may reach,
#: plus the two the readable generator's own allocations need -- ``<new>`` for the aligned
#: ``operator new[](std::align_val_t)`` it allocates heap transients with, and ``<type_traits>``
#: for the ``std::is_trivially_destructible`` static assertion it pairs with the matching delete.
#: ``<cstdlib>`` is for ``std::abort``, which canonicalization writes into the assumption-guard
#: tasklet (``if ((N < 0)) { std::abort(); }``) -- a body no printer sees, so nothing else would
#: pull the declaration in. ``<cassert>`` is the same case one level down: code generation guards
#: every map with a non-unit step by ``assert((step) > 0 && "...")``, which it writes directly into
#: the stream rather than through a printer, so no call-site table can discover it.
BASE_HEADERS: tuple[str, ...] = (
    "<cstdint>",
    "<cmath>",
    "<cstring>",
    "<cstdlib>",
    "<algorithm>",
    "<cassert>",
    "<complex>",
    "<numeric>",
    "<new>",
    "<type_traits>",
)

#: What a HIP unit adds to :data:`BASE_HEADERS`. Both ship with the ROCm toolkit, so a unit that
#: includes them needs nothing a caller compiling for an AMD GPU does not already have -- the same
#: standing OpenMP has on the host side, and the reason they are not what CPF exists to avoid.
#: hipCUB supplies the device scan, reduce and arg-reduce the library nodes expand into.
HIP_BASE_HEADERS: tuple[str, ...] = (
    "<hip/hip_runtime.h>",
    "<hip/hip_bf16.h>",
    "<hipcub/hipcub.hpp>",
    "<hip/hip_cooperative_groups.h>",
)

#: What a CUDA unit adds to :data:`BASE_HEADERS`, on the same footing: both ship with the CUDA toolkit.
#: Cooperative groups carry a megakernel's grid barrier (``Sync`` of the grid scope). The ``<cuda/std/...>``
#: headers (libcu++, part of CCCL) carry the ``__host__ __device__`` complex type and math the device unit
#: spells its kernels with (:func:`cuda_std_spell_out`).
CUDA_BASE_HEADERS: tuple[str, ...] = (
    "<cuda_runtime.h>",
    "<cuda_bf16.h>",
    "<cub/cub.cuh>",
    "<cooperative_groups.h>",
    "<cuda/std/complex>",
    "<cuda/std/cmath>",
    "<cuda/std/cstdlib>",
    "<cuda/std/limits>",
)

#: What ``dace/dace.h`` supplies that a DEVICE unit still needs, written out inline.
#:
#: These are not lowerings -- the generated device code is already correct C++ -- they are the
#: handful of declarations the header provides and the unit therefore has to provide for itself:
#: the backend-neutral ``gpu*`` types the code generator emits so one text serves CUDA and HIP, and
#: an error check. The header's MACROS are not reproduced: :func:`hip_spell_out` writes each one out
#: where the generated code uses it. Everything else the header would have brought (the reduction
#: functors, the copy templates, the runtime context) is lowered or replaced.
#:
#: ``cpf_gpu_check`` takes the call site from ``__builtin_FILE()`` / ``__builtin_LINE()`` default
#: arguments, which both hipcc and the host compilers evaluate at the caller.
#:
#: ``cpf_gpu_context`` is the replacement for ``dace::cuda::Context``: the generated frame reaches
#: it as ``__state->gpu_context->streams``, so keeping that SHAPE is what lets the device code
#: stand unaltered. One stream, because canon offloads onto the default stream
#: (``max_concurrent_streams = -1``); the array is what the emitted indexing expects.
#:
#: This is the part EVERY device unit needs, after its backend's :data:`DEVICE_TYPES`. What only
#: some need is in :data:`HIP_DEVICE_BLOCKS`, selected the same way the C helpers are -- from the
#: finished text. Calls are written as ``gpu*`` and spelled per backend (:func:`device_spell_out`).
DEVICE_TYPES: dict[Dialect, str] = {
    Dialect.STANDALONE_HIP: """\
using gpuStream_t = hipStream_t;
using gpuEvent_t = hipEvent_t;
using gpuError_t = hipError_t;
inline constexpr gpuError_t gpuSuccess = hipSuccess;
inline constexpr gpuError_t gpuErrorMemoryAllocation = hipErrorOutOfMemory;
""",
    Dialect.STANDALONE_CUDA: """\
using gpuStream_t = cudaStream_t;
using gpuEvent_t = cudaEvent_t;
using gpuError_t = cudaError_t;
inline constexpr gpuError_t gpuSuccess = cudaSuccess;
inline constexpr gpuError_t gpuErrorMemoryAllocation = cudaErrorMemoryAllocation;
""",
}

DEVICE_CORE: str = """\

//: A failed GPU call ends the program, naming the call site.
[[maybe_unused]] static inline void cpf_gpu_check(gpuError_t status, const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
    if (status != gpuSuccess) {
        fprintf(stderr, "%s:%d: GPU error %d (%s)\\n", file, line, (int)status, gpuGetErrorString(status));
        abort();
    }
}

//: The one stream canon offloads onto, in the shape the generated frame indexes.
struct cpf_gpu_context {
    gpuStream_t streams[1];
    gpuEvent_t events[1];
    gpuError_t lasterror;
};
"""

#: The arithmetic and comparison operators :func:`bfloat16_overloads` defines between a 16-bit vendor float and an
#: integer, with each one's result (``T`` is the float type).
BFLOAT16_INTEGER_OPERATORS: tuple[tuple[str, str], ...] = (
    *((op, "T") for op in ("+", "-", "*", "/")),
    *((op, "bool") for op in ("<", ">", "<=", ">=", "==", "!=")),
)


def bfloat16_overloads(vendor: str) -> str:
    """The device block of ``vendor``'s bfloat16 (``__hip_bfloat16``, ``__nv_bfloat16``): ``std::abs`` and its
    arithmetic and comparisons with an integer, each through ``float``.

    The type converts to every built-in arithmetic type, so ``std::abs(x)``, ``x / n`` and ``x < 0`` each match
    several built-in candidates at once and do not compile (ml40's resnet, GELU and group-norm kernels). A template
    over the integral types is an exact match and wins; ``float`` holds every bfloat16 value.
    """
    lines = [
        f"//: ``{vendor}`` converts to every built-in arithmetic type: these exact overloads go through ``float``.",
        "namespace std {",
        f"__host__ __device__ inline {vendor} abs({vendor} v) {{ return {vendor}(fabsf(static_cast<float>(v))); }}",
        "}  // namespace std",
    ]
    integral = "template <typename I, std::enable_if_t<std::is_integral_v<I>, int> = 0>"
    for op, result in BFLOAT16_INTEGER_OPERATORS:
        ret = vendor if result == "T" else result
        if result == "T":
            left = f"a {op} {vendor}(static_cast<float>(b))"
            right = f"{vendor}(static_cast<float>(a)) {op} b"
        else:
            left = right = f"static_cast<float>(a) {op} static_cast<float>(b)"
        lines += [
            integral,
            f"__host__ __device__ inline {ret} operator{op}({vendor} a, I b) {{ return {left}; }}",
            integral,
            f"__host__ __device__ inline {ret} operator{op}(I a, {vendor} b) {{ return {right}; }}",
        ]
    return "\n".join(lines) + "\n"


#: Device preamble blocks only SOME units need, keyed by the identifier that pulls each one in.
#:
#: Same principle as :func:`helpers_used`, and for the same reason: what a unit needs is a property
#: of the finished text, not of the SDFG. Emitting the lot unconditionally cost every GPU form the
#: 30-line atomic template and the cub aliases whether or not it reduced anything -- measured over
#: the 40-kernel corpus, the atomic block was dead in 33 of 40 and the cub aliases in 32, about 40
#: lines of a 200-line form. The form is read by an agent under a token budget, so text it has no
#: use for is not free.
HIP_DEVICE_BLOCKS: dict[str, str] = {
    "cpf_abort": """\
//: Ends the program when a run-time check fails: abort() on the host; a kernel has no abort, so it traps, and the
//: failed launch then ends the program on the host.
__host__ __device__ inline void cpf_abort() {
#if defined(__HIP_DEVICE_COMPILE__)
    __builtin_trap();
#elif defined(__CUDA_ARCH__)
    __trap();
#else
    abort();
#endif
}
""",
    "cpf_kernel_launch_check": """\
//: A failed kernel launch ends the program, naming the kernel and its launch geometry.
static inline void cpf_kernel_launch_check(gpuError_t err, const char *name, long long gx, long long gy, long long gz,
                                           long long bx, long long by, long long bz) {
    if (err != gpuSuccess) {
        fprintf(stderr, "%s launch failed (grid %lld,%lld,%lld block %lld,%lld,%lld): %s\\n", name, gx, gy, gz, bx, by,
                bz, gpuGetErrorString(err));
        abort();
    }
}
""",
    "cpf_request_dynamic_shared_memory": """\
//: A kernel that needs more dynamic shared memory than devices grant by default opts in before its launch; a device
//: that cannot grant it ends the program, naming the kernel and the request.
template <typename K>
static inline void cpf_request_dynamic_shared_memory(K kernel, const char *name, long long bytes) {
    const gpuError_t err = gpuFuncSetAttribute((const void *)kernel, gpuFuncAttributeMaxDynamicSharedMemorySize, (int)bytes);
    if (err != gpuSuccess) {
        fprintf(stderr, "%s requests %lld bytes of dynamic shared memory: %s\\n", name, bytes, gpuGetErrorString(err));
        abort();
    }
}
""",
    "cpf_narrow": """\
//: A value of any arithmetic type as a 16-bit vendor float (``__hip_bfloat16`` / ``__nv_bfloat16``). An integer
//: matches several of their converting constructors at once (int, short, float, double), so it goes through
//: ``float``, which holds every integer a 16-bit float can tell apart; a floating value converts directly.
template <typename T, typename V>
__host__ __device__ inline T cpf_narrow(const V &value) {
    if constexpr (std::is_integral_v<V>) {
        return T(static_cast<float>(value));
    } else {
        return T(value);
    }
}
""",
    "__hip_bfloat16": bfloat16_overloads("__hip_bfloat16"),
    "gpucub": """\
//: The backend cub, under the name the code generator writes -- what gpucub.cuh aliases. hipCUB
//: keeps the functor structs CCCL 3 dropped, and a unit built by hipcc is never built against CCCL.
namespace gpucub = hipcub;
""",
    "cpf_cub_multiplies": "struct cpf_cub_multiplies {\n"
    "    template <typename T>\n"
    "    __host__ __device__ T operator()(const T& a, const T& b) const { return a * b; }\n"
    "};\n",
    "cpf_cub_plus": "struct cpf_cub_plus {\n"
    "    template <typename T>\n"
    "    __host__ __device__ T operator()(const T& a, const T& b) const { return a + b; }\n"
    "};\n",
    "cpf_cub_minimum": "struct cpf_cub_minimum {\n"
    "    template <typename T>\n"
    "    __host__ __device__ T operator()(const T& a, const T& b) const { return a < b ? a : b; }\n"
    "};\n",
    "cpf_cub_maximum": "struct cpf_cub_maximum {\n"
    "    template <typename T>\n"
    "    __host__ __device__ T operator()(const T& a, const T& b) const { return a > b ? a : b; }\n"
    "};\n",
    "cpf_gpu_atomic": """\
//: A conflicting accumulation, applied atomically under any binary operator.
//:
//: HIP spells only a few operator/type pairs as an intrinsic (atomicAdd on the arithmetic types,
//: atomicMin/atomicMax on the integers), and an SDFG's write-conflict resolution is any of nine
//: reductions over any element type, so one CAS loop covers the set where a table of intrinsics
//: would leave holes. On a tree-reduced accumulator this runs ONCE PER BLOCK, after
//: gpucub::BlockReduce has folded the block's partials, so the loop is off the per-element path.
//:
//: A two-byte accumulator (bfloat16, half) has no atomic word of its own: the loop swaps the aligned
//: 32-bit word holding it and leaves the other half as it found it. Its value arrives widened, a
//: ``long`` or a ``float``, and reaches the 16-bit type through ``float``, the conversion it has.
//: nvcc 13 aborts constant-evaluating a size-changing ``__builtin_bit_cast`` even in a discarded
//: ``if constexpr`` branch (the two-byte one, for a double accumulator); a copy is the same cast.
template <typename To, typename From>
__device__ inline To cpf_bit_cast(const From &from) {
    static_assert(sizeof(To) == sizeof(From), "a bit cast keeps the size");
    To to;
    memcpy(&to, &from, sizeof(To));
    return to;
}
template <typename T, typename V, typename Op>
__device__ inline void cpf_gpu_atomic(T *address, V value, Op op) {
    if constexpr (sizeof(T) == 1) {
        // a one-byte accumulator (bool, char): the same swap of its aligned 32-bit word, one byte of it
        const uintptr_t at = reinterpret_cast<uintptr_t>(address);
        unsigned int *word = reinterpret_cast<unsigned int *>(at & ~uintptr_t(3));
        const unsigned int shift = static_cast<unsigned int>(at & 3) * 8u;
        const T operand = static_cast<T>(value);
        unsigned int old = *word;
        unsigned int assumed;
        do {
            assumed = old;
            const T updated = op(cpf_bit_cast<T>(static_cast<unsigned char>(assumed >> shift)), operand);
            const unsigned int bits = static_cast<unsigned int>(cpf_bit_cast<unsigned char>(updated)) << shift;
            old = atomicCAS(word, assumed, (assumed & ~(0xFFu << shift)) | bits);
        } while (assumed != old);
    } else if constexpr (sizeof(T) == 2) {
        const uintptr_t at = reinterpret_cast<uintptr_t>(address);
        unsigned int *word = reinterpret_cast<unsigned int *>(at & ~uintptr_t(3));
        const unsigned int shift = (at & 2) ? 16u : 0u;
        const T operand = static_cast<T>(static_cast<float>(value));
        unsigned int old = *word;
        unsigned int assumed;
        do {
            assumed = old;
            const T updated = op(cpf_bit_cast<T>(static_cast<unsigned short>(assumed >> shift)), operand);
            const unsigned int bits = static_cast<unsigned int>(cpf_bit_cast<unsigned short>(updated)) << shift;
            old = atomicCAS(word, assumed, (assumed & ~(0xFFFFu << shift)) | bits);
        } while (assumed != old);
    } else {
        using cpf_atomic_word =
            typename std::conditional<sizeof(T) == sizeof(unsigned int), unsigned int, unsigned long long>::type;
        static_assert(sizeof(T) == sizeof(cpf_atomic_word), "no atomic word as wide as the accumulator");
        cpf_atomic_word *word = reinterpret_cast<cpf_atomic_word *>(address);
        cpf_atomic_word old = *word;
        cpf_atomic_word assumed;
        do {
            assumed = old;
            T updated = op(cpf_bit_cast<T>(assumed), static_cast<T>(value));
            old = atomicCAS(word, assumed, cpf_bit_cast<cpf_atomic_word>(updated));
        } while (assumed != old);
    }
}
""",
    "cpf_gpu_atomic_add_complex": """\
//: A complex sum, applied atomically: the element is two words wide, which no compare-and-swap covers,
//: and a sum of complex numbers is the sum of their parts, so each part is folded on its own.
template <typename R, typename V>
__device__ inline void cpf_gpu_atomic_add_complex(std::complex<R> *address, V value) {
    const std::complex<R> v = static_cast<std::complex<R>>(value);
    R *parts = reinterpret_cast<R *>(address);  // std::complex<R> is laid out as R[2]
    atomicAdd(parts, v.real());
    atomicAdd(parts + 1, v.imag());
}
""",
}

#: The CUDA unit's blocks: the same ones, with cub under its CUDA name.
CUDA_DEVICE_BLOCKS: dict[str, str] = {
    **HIP_DEVICE_BLOCKS,
    "__nv_bfloat16": bfloat16_overloads("__nv_bfloat16"),
    "gpucub": "//: The backend cub, under the name the code generator writes -- what gpucub.cuh aliases.\n"
    "namespace gpucub = cub;\n",
}

DEVICE_BLOCKS: dict[Dialect, dict[str, str]] = {
    Dialect.STANDALONE_HIP: HIP_DEVICE_BLOCKS,
    Dialect.STANDALONE_CUDA: CUDA_DEVICE_BLOCKS,
}

#: Emission order for :data:`DEVICE_BLOCKS`. A dict preserves insertion order, but the blocks
#: are selected into a set, so the order a unit gets them in has to be stated rather than inherited
#: from however the set happened to iterate -- two runs of the same SDFG must render byte-identical.
DEVICE_BLOCK_ORDER: tuple[str, ...] = (*HIP_DEVICE_BLOCKS, "__nv_bfloat16")

#: What each block needs in turn.
DEVICE_BLOCK_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "cpf_gpu_atomic": ("gpucub",),
}

#: Backend-neutral ``gpu*`` calls the CUDA/HIP code generator emits, and the HIP function each one
#: is. ``dace/dace.h`` aliases these through its own compatibility header; a self-contained unit
#: calls the HIP function by its own name (:func:`hip_spell_out`).
#:
#: The generator can emit all of these, so a table missing one leaves an undeclared name in the
#: unit -- ``tsvc_2_s323`` failed on ``gpuMemcpyDeviceToHost`` that way.
GPU_ALIASES: dict[str, str] = {
    "gpuDeviceSynchronize": "hipDeviceSynchronize",
    "gpuEventSynchronize": "hipEventSynchronize",
    "gpuFree": "hipFree",
    "gpuFuncAttributeMaxDynamicSharedMemorySize": "hipFuncAttributeMaxDynamicSharedMemorySize",
    "gpuFuncSetAttribute": "hipFuncSetAttribute",
    "gpuFreeAsync": "hipFreeAsync",
    "gpuFreeHost": "hipHostFree",
    "gpuGetErrorString": "hipGetErrorString",
    "gpuGetLastError": "hipGetLastError",
    "gpuLaunchKernel": "hipLaunchKernel",
    "gpuMalloc": "hipMalloc",
    "gpuMallocAsync": "hipMallocAsync",
    "gpuMallocHost": "hipHostMalloc",
    "gpuMemcpy": "hipMemcpy",
    "gpuMemcpyAsync": "hipMemcpyAsync",
    "gpuMemcpyDeviceToDevice": "hipMemcpyDeviceToDevice",
    "gpuMemcpyDeviceToHost": "hipMemcpyDeviceToHost",
    "gpuMemcpyHostToDevice": "hipMemcpyHostToDevice",
    "gpuMemset": "hipMemset",
    "gpuMemsetAsync": "hipMemsetAsync",
    "gpuPeekAtLastError": "hipPeekAtLastError",
    "gpuStreamSynchronize": "hipStreamSynchronize",
}

#: Every DaCe spelling the generated device code carries, written out where it is used: the
#: attribute aliases as their attributes, the error checks as CPF's own functions, the cub operator
#: names as the functor each one names, and each ``gpu*`` call as its HIP function.
HIP_SPELLINGS: dict[str, str] = {
    "DACE_ABORT": "cpf_abort",
    "DACE_HDFI": "__host__ __device__ __forceinline__",
    "DACE_HFI": "__host__ __forceinline__",
    "DACE_DFI": "__device__ __forceinline__",
    "DACE_GPU_CHECK": "cpf_gpu_check",
    "DACE_KERNEL_LAUNCH_CHECK": "cpf_kernel_launch_check",
    "DACE_KERNEL_REQUEST_DYNAMIC_SHARED_MEMORY": "cpf_request_dynamic_shared_memory",
    "DACE_CUB_SUM_OP": "::gpucub::Sum()",
    "DACE_CUB_MIN_OP": "::gpucub::Min()",
    "DACE_CUB_MAX_OP": "::gpucub::Max()",
    "DACE_CUB_MUL_OP": "cpf_cub_multiplies()",
    # std::min / std::max are constexpr host functions: nvcc refuses them in a kernel without
    # --expt-relaxed-constexpr, which a judge does not pass (NestForge #93, bellman_ford's min-accumulation).
    # The cpf helpers are __host__ __device__ in a device unit and take the same arguments.
    "std::min": "cpf_min",
    "std::max": "cpf_max",
    **GPU_ALIASES,
}

#: :data:`HIP_SPELLINGS` for the CUDA unit: each ``gpu*`` call as its CUDA function, and the cub
#: operators as CPF's own functors, because CCCL 3 dropped the ``Sum``/``Min``/``Max`` structs.
CUDA_SPELLINGS: dict[str, str] = {
    **HIP_SPELLINGS,
    "DACE_CUB_SUM_OP": "cpf_cub_plus()",
    "DACE_CUB_MIN_OP": "cpf_cub_minimum()",
    "DACE_CUB_MAX_OP": "cpf_cub_maximum()",
    **{name: "cuda" + name[len("gpu") :] for name in GPU_ALIASES},
}

DEVICE_SPELLINGS: dict[Dialect, dict[str, str]] = {
    Dialect.STANDALONE_HIP: HIP_SPELLINGS,
    Dialect.STANDALONE_CUDA: CUDA_SPELLINGS,
}

#: One alternation per dialect over its spellings. No replacement contains a key, so one pass is exact.
DEVICE_SPELLING_PATTERNS: dict[Dialect, "re.Pattern"] = {
    dialect: re.compile(r"\b(" + "|".join(sorted(spellings, key=len, reverse=True)) + r")\b")
    for dialect, spellings in DEVICE_SPELLINGS.items()
}

#: ``DACE_EXPORTED`` names no attribute in a self-contained unit, so it goes with the blank after it.
HIP_EXPORTED_PATTERN: "re.Pattern" = re.compile(r"\bDACE_EXPORTED\b[ \t]*")

#: The ``__dace_`` prefix of the generator's own launchers and locals (``__dace_runkernel_*``,
#: ``__dace_current_stream``). ``__dace_init``/``__dace_exit`` keep it: they are a refused
#: handshake, and :data:`~dace.codegen.cpf.BANNED` has to still see one.
DACE_PREFIX_PATTERN: "re.Pattern" = re.compile(r"\b__dace_(?!init|exit)")


def device_spell_out(code: str, dialect: Dialect) -> str:
    """Write out each of ``dialect``'s spellings, drop ``DACE_EXPORTED`` and respell ``__dace_`` as ``__cpf_``.

    :param code: the emitted device unit, or its preamble.
    :param dialect: the device dialect rendering it (see :data:`DEVICE_SPELLINGS`).
    :returns: the text with every DaCe spelling replaced.
    """
    spellings = DEVICE_SPELLINGS[dialect]
    code = DACE_PREFIX_PATTERN.sub("__cpf_", HIP_EXPORTED_PATTERN.sub("", code))
    return DEVICE_SPELLING_PATTERNS[dialect].sub(lambda match: spellings[match.group(1)], code)


#: The standard-library names a CUDA kernel cannot call as ``std::``: ``std::complex`` and the functions that
#: take one are ``constexpr`` host functions (libstdc++, C++20), which nvcc refuses in device code without
#: ``--expt-relaxed-constexpr`` -- and a judge does not pass that flag (SC40: quatrex_rgf, qe_vexx_k,
#: ls3df_scf). libcu++'s ``cuda::std::`` has each one as ``__host__ __device__``, with the real overloads too,
#: so one call that a tasklet prints for a real or a complex argument resolves either way. Same layout as
#: ``std::complex`` (two components, real first), which is what DaCe's own CUDA build relies on with
#: ``thrust::complex``.
CUDA_STD_NAMES: tuple[str, ...] = (
    "complex",
    "conj",
    "real",
    "imag",
    "abs",
    "arg",
    "norm",
    "polar",
    "proj",
    "exp",
    "log",
    "log10",
    "sqrt",
    "pow",
    "sin",
    "cos",
    "tan",
    "asin",
    "acos",
    "atan",
    "sinh",
    "cosh",
    "tanh",
    "asinh",
    "acosh",
    "atanh",
    "numeric_limits",
)

#: A ``std::`` name of :data:`CUDA_STD_NAMES`, not already qualified further out (``cuda::std::``).
CUDA_STD_PATTERN: "re.Pattern" = re.compile(r"(?<![\w:])std::(" + "|".join(CUDA_STD_NAMES) + r")\b")


def cuda_std_spell_out(device_code: str, state_struct: str) -> str:
    """``device_code`` with each :data:`CUDA_STD_NAMES` name spelled ``cuda::std::``, but for the lines both
    units must declare alike: the launchers' signatures and the state struct.

    The host unit keeps ``std::complex``: its entry takes the caller's ``std::complex`` pointers, and it calls
    the launchers with them. A launcher passes its arguments to ``cudaLaunchKernel`` as ``void *``, so its
    ``std::complex`` parameters reach the kernel's ``cuda::std::complex`` ones unconverted, and its signature
    stays the one the host unit declares and links against.

    :param device_code: the finished device unit, preamble included.
    :param state_struct: the frame's state struct as the device unit repeats it (:func:`~dace.codegen.cpf.state_struct`).
    :returns: the unit with its kernels and device helpers spelled with libcu++.
    """
    head, struct, tail = device_code.partition(state_struct) if state_struct else (device_code, "", "")
    return cuda_std_spell_out_lines(head) + struct + cuda_std_spell_out_lines(tail)


def cuda_std_spell_out_lines(text: str) -> str:
    """:func:`cuda_std_spell_out` over ``text``, skipping each launcher's signature line (a declaration ends
    ``);``, the definition ``)`` with its body on the lines below)."""
    return "".join(
        line
        if "__cpf_runkernel_" in line and line.rstrip().endswith((")", ");"))
        else CUDA_STD_PATTERN.sub(r"cuda::std::\1", line)
        for line in text.splitlines(keepends=True)
    )


#: The local that points the generated body at the context's streams, and its release at the end of the scope.
STREAM_ARRAY_LINE: "re.Pattern" = re.compile(
    r"^[ \t]*(?:gpuStream_t\s*\*\s*gpu_streams\s*=\s*__state->gpu_context->streams|gpu_streams\s*=\s*nullptr);[ \t]*\n",
    re.MULTILINE,
)
#: A local naming one element of the stream array (``gpuStream_t __cpf_current_stream = gpu_streams[0];``).
STREAM_LOCAL_LINE: "re.Pattern" = re.compile(
    r"^[ \t]*gpuStream_t\s+(\w+)\s*=\s*gpu_streams\[0\];[ \t]*\n", re.MULTILINE
)
#: A stream wait directly before the entry's closing device synchronization, which already waits for it.
REDUNDANT_STREAM_WAIT: "re.Pattern" = re.compile(
    r"^[ \t]*cpf_gpu_check\(\w+StreamSynchronize\(nullptr\)\);[ \t]*\n(?=[ \t]*cpf_gpu_check\(\w+DeviceSynchronize\(\)\);)",
    re.MULTILINE,
)


def default_stream_spell_out(code: str) -> str:
    """``code`` with every GPU stream spelled ``nullptr``, the default stream, and the stream variables gone.

    The GPU code generator reaches its streams through ``gpu_streams``, a local pointing at the state's context,
    and through locals naming one element of it. Without boilerplate (:func:`no_boilerplate`) there is one stream,
    the default one (``compiler.cuda.max_concurrent_streams = -1``), so each of those names is ``nullptr``.

    :raises RuntimeError: if a stream other than the first is named, or ``gpu_streams`` survives: the code then
                          relies on a stream this rendering does not have.
    """
    code = STREAM_ARRAY_LINE.sub("", code)
    locals_ = STREAM_LOCAL_LINE.findall(code)
    code = STREAM_LOCAL_LINE.sub("", code)
    code = re.sub(r"\bgpu_streams\[0\]", "nullptr", code)
    for name in OrderedSet(locals_):
        code = re.sub(rf"\b{re.escape(name)}\b", "nullptr", code)
    if re.search(r"\bgpu_streams\b", code):
        raise RuntimeError(
            "a rendering without boilerplate has only the default stream, but the code names gpu_streams"
        )
    return code


def drop_redundant_stream_wait(code: str) -> str:
    """``code`` without a wait on the default stream directly before the device synchronization that ends the entry
    without boilerplate (:func:`no_boilerplate`), which waits for it too. Run on the compacted unit: before
    :func:`~dace.codegen.cpf.compact_blocks` the closing braces of the generator's blocks sit between the two."""
    return REDUNDANT_STREAM_WAIT.sub("", code)


#: The name one line of :data:`DEVICE_TYPES` defines.
DEVICE_ALIAS: "re.Pattern" = re.compile(r"(?:using|inline constexpr gpuError_t) (\w+)")


def device_preamble(code: str, dialect: Dialect) -> str:
    """``dialect``'s :data:`DEVICE_TYPES` and :data:`DEVICE_CORE`, plus the blocks ``code`` reaches for.

    :param code: the emitted translation unit after :func:`device_spell_out`, WITHOUT its preamble --
                 so a block's own definition never counts as a use of it.
    :param dialect: the device dialect rendering it.
    :returns: the device preamble, core first and blocks in :data:`DEVICE_BLOCK_ORDER`, spelled for the backend.
    """
    blocks = DEVICE_BLOCKS[dialect]
    needed: OrderedSet[str] = OrderedSet()
    pending = [name for name in blocks if re.search(rf"\b{re.escape(name)}\b", code)]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(dep for dep in DEVICE_BLOCK_DEPENDENCIES.get(name, ()) if dep not in needed)
    types, core = DEVICE_TYPES[dialect], DEVICE_CORE
    # The flag itself, not :func:`no_boilerplate`: the preamble is written after the dialect's scope has closed.
    if no_boilerplate_in_force:
        # No context; of the backend aliases, only those the unit or the check names.
        core = core.partition("\n//: The one stream")[0] + "\n"
        used = code + core + "".join(blocks[name] for name in needed)
        types = "".join(
            line + "\n" for line in types.splitlines() if re.search(rf"\b{DEVICE_ALIAS.match(line).group(1)}\b", used)
        )
    return device_spell_out(
        "\n".join([types + core] + [blocks[name] for name in DEVICE_BLOCK_ORDER if name in needed]),
        dialect,
    )


def device_entry_prologue(state_struct: str, init_code: str = "", exit_code: str = "", has_state: bool = True) -> str:
    """The device setup CPF's single entry function opens with.

    An ordinary build does this in ``__dace_init_<name>`` and undoes it in ``__dace_exit_<name>``,
    both taking the state pointer the caller kept between invocations. CPF has one entry point and
    no handshake, so the context is a LOCAL: created here, destroyed by the scope guard on the way
    out, whichever way the function leaves.

    The stream is the default one, which is what canon offloads onto
    (``compiler.cuda.max_concurrent_streams = -1``), so there is nothing to create and nothing to
    destroy -- only the device to select and the work to drain.

    The state's own setup and teardown -- a vendor FFT plan the state caches, which a tasklet declares as
    ``code_init`` / ``code_exit`` -- runs here too, since the state lives exactly as long as the call: the setup
    after the state is built, the teardown on the way out AFTER the drain, so nothing the device still runs loses
    its plan. (Guards are destroyed in reverse order, so the teardown's guard is declared first.)

    :param state_struct: the mangled state struct name the generated body dereferences.
    :param init_code: the state's setup, the code ``__dace_init`` would run.
    :param exit_code: the state's teardown, the code ``__dace_exit`` would run.
    :param has_state: whether the state struct has any field; without boilerplate (:func:`no_boilerplate`) an empty
                      one is not emitted, so there is no state to declare.
    :returns: the prologue lines, indented one level.
    """
    teardown = ""
    if exit_code.strip():
        teardown = f"""\
    struct __cpf_exit {{
        {state_struct} *__state;
        ~__cpf_exit() {{
{exit_code}
        }}
    }} __cpf_exit_guard{{__state}};
"""
    if no_boilerplate():
        # No context and no drain guard: every stream is ``nullptr`` and the footer synchronizes the device.
        state = (
            f"    {state_struct} __cpf_state{{}};\n    {state_struct} *__state = &__cpf_state;\n" if has_state else ""
        )
        return f"{state}{init_code}{teardown}"
    return f"""\
    // The device handshake, which a single entry point has nowhere else to put.
    cpf_gpu_context __cpf_context{{}};
    __cpf_context.streams[0] = nullptr;
    {state_struct} __cpf_state{{&__cpf_context}};
    {state_struct} *__state = &__cpf_state;
{init_code}
{teardown}    struct __cpf_drain {{
        ~__cpf_drain() {{ cpf_gpu_check(gpuDeviceSynchronize()); }}
    }} __cpf_drain_guard;
"""


#: Runtime functions CPF deliberately does NOT lower, and why. Reaching one is a refusal, not a
#: pass-through: the name is declared by a DaCe header CPF does not include, so passing it through
#: would produce a translation unit that does not build.
UNSUPPORTED: dict[str, str] = {}

#: Math functions a RUNTIME-dialect printer must qualify as ``dace::math::``, rather than leave
#: bare for unqualified lookup to resolve.
#:
#: A bare ``sqrt(x)`` binds to ``std::sqrt``, whose ``float`` / ``double`` / ``long double``
#: overloads are all equally good for a 16-bit float: ``dace::float16`` IS CUDA's ``half``, and its
#: many non-explicit conversion operators each reach a different overload through a different
#: conversion, so nvcc calls the call ambiguous and rejects the translation unit. Each name here
#: carries a non-template ``dace::float16`` / ``dace::bfloat16`` overload in ``dace/math.h``, an
#: exact type match that wins outright -- qualifying the call is what reaches it. A math name with
#: no such overload (``sin``, ``cos``, ...) is deliberately absent: qualifying it would move the
#: same ambiguity one frame down, into ``dace::math``'s own template body.
#:
#: Shared because the SAME expression reaches C++ through two printers -- a tasklet body through
#: ``cppunparse``, a memlet subset or interstate assignment through ``dace.symbolic`` -- and a name
#: qualified by only one of them builds in one place and is ambiguous in the other. This table has
#: nothing to say about the standalone dialects, which resolve these names through
#: :data:`STD_RENAMES` before any of this applies.
RUNTIME_QUALIFIED_MATH: dict[str, str] = {
    "fma": "dace::math::fma",
    "sqrt": "dace::math::sqrt",
    "exp": "dace::math::exp",
    "log": "dace::math::log",
}

#: Every runtime function this module knows about, in any lane.
KNOWN: set[str] = set(STD_RENAMES) | set(REWRITES) | set(INLINE_DEFINITIONS) | set(VARIADIC_MINMAX) | set(UNSUPPORTED)

#: ``dace::``-namespaced C++ type -> the standalone spelling. Most DaCe ctypes are already plain
#: (``float64`` is ``double``, ``int32`` is ``int32_t``), so only these few ever leak.
CTYPE_RENAMES: dict[str, str] = {
    "dace::bool_": "bool",
    "dace::uint": "uint32_t",
    "dace::uint8": "uint8_t",
    "dace::uint16": "uint16_t",
    "dace::uint32": "uint32_t",
    "dace::uint64": "uint64_t",
    "dace::int8": "int8_t",
    "dace::int16": "int16_t",
    "dace::int32": "int32_t",
    "dace::int64": "int64_t",
    "dace::float32": "float",
    "dace::float64": "double",
    "dace::complex64": "std::complex<float>",
    "dace::complex128": "std::complex<double>",
}

#: The 16-bit floats a HOST unit spells natively: the compiler's own ``__bf16`` and ``_Float16`` -- arithmetic,
#: conversions and OpenMP reductions in C and C++ with GCC 13 and later and Clang 18 and later (checked with GCC
#: 13 / 14 / 16 and Clang 18 / 22 on x86-64). Each is two bytes in the bit layout of HB's ``__npb_bf16`` /
#: ``uint16_t`` storage, so an entry taking ``__bf16 *`` takes what the caller hands over as ``uint16_t *``.
HOST_LOW_PRECISION_CTYPES: dict[str, str] = {"dace::bfloat16": "__bf16", "dace::float16": "_Float16"}

#: Types with no portable standalone spelling. ``float16``/``bfloat16``/fp8 exist in DaCe only as
#: the CUDA and ROCm vendor types, or as an emulation the runtime headers carry -- neither is
#: reachable from a translation unit that includes nothing but the standard library.
UNSUPPORTED_CTYPES: dict[str, str] = {
    "dace::float16": "the C dialect has no half type; render the C++ dialect (_Float16)",
    "dace::bfloat16": "the C dialect has no bfloat16; render the C++ dialect (__bf16), or a device dialect",
    "dace::float8_e4m3fn": "no portable C++ fp8 type",
    "dace::float8_e5m2": "no portable C++ fp8 type",
}


def refuse_unspellable_containers(sdfg, dialect: Dialect | None = None) -> None:
    """Refuse an SDFG holding a container whose type ``dialect`` has no spelling for, before anything is printed.

    Left to the printers, the first one to reach such a container reports something else -- the C dialect picks its
    typed math by type name and says it found none for ``exp`` -- where this names the type and the way out.

    :raises NotImplementedError: naming the type and why (:data:`UNSUPPORTED_CTYPES`).
    """
    renames = tables_for(dialect).ctype_renames
    for _, name, desc in sdfg.arrays_recursive():
        ctype = desc.dtype.ctype
        if ctype in UNSUPPORTED_CTYPES and ctype not in renames:
            raise NotImplementedError(f"CPF cannot emit the type {ctype!r} of {name!r}: {UNSUPPORTED_CTYPES[ctype]}.")


def ctype_for(ctype: str, dialect: Dialect | None = None) -> str:
    """The standalone spelling of a type name.

    :param ctype: the type as the ordinary generators spell it.
    :param dialect: which standalone dialect to spell it for; ambient when omitted.
    :returns: the standalone spelling, or ``ctype`` unchanged when it is already plain.
    :raises NotImplementedError: if the type has no standalone spelling at all.
    """
    renames = tables_for(dialect).ctype_renames
    if ctype in UNSUPPORTED_CTYPES and ctype not in renames:
        raise NotImplementedError(f"CPF cannot emit the type {ctype!r}: {UNSUPPORTED_CTYPES[ctype]}.")
    return renames.get(ctype, ctype)


def variadic_minmax(
    name: str,
    arguments: tuple[str, ...],
    dialect: Dialect | None = None,
    types: tuple[str | None, ...] | None = None,
) -> str | None:
    """Spell a variadic ``Max``/``Min`` for ``dialect``.

    The runtime's ``Max`` takes any number of arguments, and so does the C++ dialect's own
    ``cpf_max`` template, so that one is called with the arguments as they stand. C has neither a
    template nor overloading, so the C dialect NESTS the typed binary helper instead -- left to
    right, which is the association the recursive template has too (see :func:`c_minmax_call`).

    :param name: the runtime function name.
    :param arguments: already-printed argument expressions.
    :param dialect: which standalone dialect to spell it for; ambient when omitted.
    :param types: the dace type name of each argument. Read only by the C dialect.
    :returns: the expression, or ``None`` if ``name`` is not a min/max.
    """
    resolved = dialect if dialect is not None else _active_dialect
    target = tables_for(resolved).variadic_minmax.get(name)
    if target is None:
        return None
    if len(arguments) == 1:
        return "(%s)" % arguments[0]
    if resolved is Dialect.STANDALONE_C:
        return c_minmax_call(target, arguments, types)
    if len(arguments) == 2:
        return "%s(%s, %s)" % (target, arguments[0], arguments[1])
    return "%s(%s)" % (target, ", ".join(arguments))


def needs_definition(name: str, dialect: Dialect | None = None) -> bool:
    """Whether CPF calls ``name`` unchanged and emits a definition for it."""
    return name in tables_for(dialect).inline_definitions


def lowering_for(
    name: str,
    arguments: tuple[str, ...],
    dialect: Dialect | None = None,
    types: tuple[str | None, ...] | None = None,
) -> str | None:
    """The CPF spelling of a call to ``name`` with ``arguments`` already printed.

    :param name: the runtime function name as the ordinary generators would emit it.
    :param arguments: already-printed argument expressions.
    :param types: the dace type name of each argument where the printer resolved it. The C dialect
                  names a typed helper after them; the C++ dialect never reads them.
    :returns: the C++ expression, or ``None`` if ``name`` needs no rewriting -- either it is not a
              runtime function at all, or it is one CPF emits a definition for and calls unchanged
              (:func:`needs_definition` separates those two).
    :raises ValueError: if ``name`` is a known rewrite but the argument count does not match, which
                        means the caller and this table disagree about the function's shape.
    :raises NotImplementedError: if ``name`` is a runtime function CPF cannot express (see
                                 :data:`UNSUPPORTED`).
    """
    # numpy's conjugate of a real is that real; ``std::conj`` of one is a ``std::complex``.
    if (
        name in ("conj", "conjugate")
        and types is not None
        and len(types) == 1
        and types[0] is not None
        and not types[0].startswith("complex")
    ):
        return "(%s)" % arguments[0]
    tables = tables_for(dialect)
    if name in tables.unsupported:
        raise NotImplementedError(f"CPF cannot lower {name!r}: {tables.unsupported[name]}.")
    if (dialect if dialect is not None else _active_dialect) is Dialect.STANDALONE_C:
        if name in C_TYPED_MATH:
            return "%s(%s)" % (c_math_function(name, types), ", ".join(arguments))
        if name == "iround":
            return "((int)%s(%s))" % (c_math_function("round", types), ", ".join(arguments))
        # ``Mod`` is sympy's floored modulus: the same typed helper as ``py_mod``.
        spec = c_helper_spec("py_mod" if name == "Mod" else name, len(arguments))
        if spec is not None:
            return c_helper_call(spec, arguments, types)
    variadic = variadic_minmax(name, arguments, dialect, types)
    if variadic is not None:
        return variadic
    if name in tables.rewrites:
        arity, template = tables.rewrites[name]
        if len(arguments) != arity:
            raise ValueError(f"CPF lowering of {name!r} expects {arity} arguments, got {len(arguments)}")
        return template.format(*arguments)
    if name in tables.std_renames:
        return "%s(%s)" % (tables.std_renames[name], ", ".join(arguments))
    return None


#: A ``dace::``-qualified name in hand-written C++ (a native tasklet body, a library expansion's
#: code), with the trailing identifier captured. Matches a type and a function alike -- the
#: distinction is made by which table the identifier is found in.
_QUALIFIED_NAME = re.compile(r"(?:::)?\bdace::(?:[A-Za-z_]\w*::)*([A-Za-z_]\w*)\b")


def rewrite_ctypes(code: str, dialect: Dialect | None = None) -> str:
    """Spell every DaCe ctype in ``code`` the standalone way.

    Type names reach the emitted text from places no expression printer sees -- the entry
    signature, transient declarations, casts -- so this runs over the finished unit as well as over
    hand-written tasklet bodies. Purely a rename, from the same table
    (:data:`CTYPE_RENAMES` / :data:`C_CTYPE_RENAMES`) both callers share, so a container's type and
    a tasklet's cast cannot be spelled differently.

    The rename itself is one alternation over every ``ctype_renames`` key
    (:attr:`~Tables.ctype_pattern`, built once per dialect at import time) rather than one
    ``re.sub`` per entry: every entry is a distinct literal name that cannot appear in another
    entry's REPLACEMENT text (none of ``bool``, ``uint32_t``, ``std::complex<float>``, ... contains
    a ``dace::``-qualified name), so the entries never interacted and folding them into one pass
    changes nothing they could produce. The refusal check stays its own small loop, in TABLE
    order, so two unsupported types in one unit are still reported in the same order as before.

    :param code: emitted text, of any size.
    :param dialect: which standalone dialect emitted it; ambient when omitted.
    :returns: the text with DaCe type names replaced.
    :raises NotImplementedError: if the text names a type with no standalone spelling.
    """
    tables = tables_for(dialect)
    if "::" not in code:
        return code
    for qualified, reason in UNSUPPORTED_CTYPES.items():
        if qualified not in tables.ctype_renames and re.search(r"(?:::)?\b%s\b" % re.escape(qualified), code):
            raise NotImplementedError(f"CPF cannot emit the type {qualified!r}: {reason}")
    code = tables.ctype_pattern.sub(lambda match: tables.ctype_renames[match.group(1)], code)
    resolved = dialect if dialect is not None else _active_dialect
    if resolved in DEVICE_DIALECTS:
        code = VENDOR_FLOAT_CAST.sub(lambda match: f"cpf_narrow<{match.group(1)}>(", code)
    elif resolved is Dialect.STANDALONE and re.search(r"\b(?:__bf16|_Float16)\b", code):
        code = LOW_PRECISION_MATH_CALL.sub(lambda match: f"cpf_lowp_{match.group(1)}(", code)
    return code


#: A functional cast to a 16-bit vendor float, ``__hip_bfloat16(n)``: spelled through :data:`HIP_DEVICE_INLINE_DEFINITIONS`'
#: ``cpf_narrow``, because an integer operand is ambiguous between the type's converting constructors. Not a
#: default construction (``T()``), and not a template argument or a qualified name.
VENDOR_FLOAT_CAST: "re.Pattern" = re.compile(r"(?<![\w:.<])(__hip_bfloat16|__nv_bfloat16)\s*\((?!\s*\))")


class NativeSite:
    """What the C rewrite of one hand-written body knows about the place it is emitted.

    ``names`` maps each identifier in scope -- container, connector, symbol -- to ``(dace type,
    pointer)``, which types the helpers the body calls. ``label`` is unique per site and names the
    functions a find-first becomes; ``functions`` collects those for the caller to emit at file scope.
    """

    __slots__ = ("names", "label", "functions")

    def __init__(self, names: dict[str, tuple[str, bool]], label: str) -> None:
        self.names: dict[str, tuple[str, bool]] = names
        self.label: str = label
        self.functions: list[str] = []


#: The ``Scan`` CPU lowering's ``dace::scan`` entry points, respelled by :func:`respell_scan_entry_points`.
SCAN_ENTRY_CALL: "re.Pattern" = re.compile(
    r"(?:::)?dace::scan::(inclusive|exclusive|strided_inclusive)_(sum|product|min|max)\s*\("
)
AFFINE_SCAN_ENTRY_CALL: "re.Pattern" = re.compile(r"(?:::)?dace::scan::inclusive_affine(_strided)?\s*\(")
#: The ``Reduce`` CPU lowering's ``dace::reduce`` entry points, respelled by :func:`respell_reduce_entry_points`.
REDUCE_ENTRY_CALL: "re.Pattern" = re.compile(r"(?:::)?\bdace::reduce::(seq::)?(%s)\s*\(" % "|".join(REDUCE_OPERATIONS))
#: The copy node's contiguous CPU lowering, ``dace::CopyImpl<T, 1>(src, dst, count)``.
COPY_IMPL_CALL: "re.Pattern" = re.compile(r"(?:::)?dace::CopyImpl\s*<\s*([^,<>]+?)\s*,\s*1\s*>\s*\(")

#: How the running value ``{a}`` takes in the next element ``{b}``, per scan operation, as the runtime's
#: ``std::min`` / ``std::max`` order them, so C and C++ read the same expression.
STRIDED_SCAN_COMBINE: dict[str, str] = {
    "sum": "{a} + {b}",
    "product": "{a} * {b}",
    "min": "({b} < {a} ? {b} : {a})",
    "max": "({a} < {b} ? {b} : {a})",
}

#: Elements per chunk of the blocked strided scan, rounded down to whole rows of the carry distance.
STRIDED_SCAN_CHUNK = 4096
#: Chunks per thread per round, so a round's chunks are still cached when pass 2 rereads them.
STRIDED_SCAN_CHUNKS_PER_THREAD = 8


def parenthesized(expression: str) -> str:
    """``expression`` inside exactly one pair of enclosing parentheses."""
    expression = expression.strip()
    return expression if expression.startswith("(") and c_encloses(expression, 0) else "(%s)" % expression


def residue_class_loop(n: str, s: str, head: str, step: str, parallel: bool) -> str:
    """One pass per residue class ``cpf_r`` of the index mod ``s``: ``head`` writes element ``cpf_r``,
    ``step`` element ``cpf_j`` from ``cpf_j - cpf_s``. The classes are independent, so the class loop is
    the parallel one."""
    return (
        "{ const long cpf_n = (long)%s; const long cpf_s = (long)%s;\n"
        "  if (cpf_s <= 0) std::abort();\n"
        "%s"
        "  for (long cpf_r = 0; cpf_r < cpf_s; ++cpf_r) {\n"
        "      if (cpf_r >= cpf_n) continue;\n"
        "      %s;\n"
        "      for (long cpf_j = cpf_r + cpf_s; cpf_j < cpf_n; cpf_j += cpf_s) %s;\n"
        "  }\n"
        "}" % (parenthesized(n), parenthesized(s), "  #pragma omp parallel for\n" if parallel else "", head, step)
    )


#: The rescan arms of a strided scan by carry distance ``K``: ``(least K, the runtime finding, what the arm runs)``.
#: A loop of the recurrence through memory waits on store-to-load forwarding at every step, which made ``K == 2`` 2.5x
#: slower than the same loop with the literal 2, whose chains the compiler keeps in registers.
STRIDED_RESCAN_ARMS: tuple[tuple[int, str, str], ...] = (
    (2, "K == 2", "two independent chains, each a prefix scan at the literal distance 2"),
    (3, "3 <= K <= 8", "K independent chains, each a prefix scan carried in a register"),
    (9, "K > 8", "K independent chains, each a prefix scan; a vector holds consecutive elements of different chains"),
)
#: The most chains the register arm carries.
STRIDED_REGISTER_CHAINS = 8


def strided_rescan_arm(
    least: int, o: str, i: str, s: str, span: tuple[str, str], fold: Callable[[str, str], str], acc_type: str
) -> list[str]:
    """``o[j] = o[j - s] OP i[j]`` over ``span`` (whose first element is at least ``s``) in the arm from ``least``."""
    first, last = span
    if least == 2:
        return [
            "for (long cpf_i = %s; cpf_i < %s; ++cpf_i) %s[cpf_i] = %s;"
            % (first, last, o, fold("%s[cpf_i - 2]" % o, "%s[cpf_i]" % i))
        ]
    if least > STRIDED_REGISTER_CHAINS:
        return [
            "for (long cpf_i = %s; cpf_i < %s; ++cpf_i) %s[cpf_i] = %s;"
            % (first, last, o, fold("%s[cpf_i - %s]" % (o, s), "%s[cpf_i]" % i))
        ]
    return [
        "if (%s < %s) {" % (first, last),
        "    %s cpf_acc[%d];" % (acc_type, STRIDED_REGISTER_CHAINS),
        "    for (long cpf_k = 0; cpf_k < %s; ++cpf_k) cpf_acc[cpf_k] = %s[%s - %s + cpf_k];" % (s, o, first, s),
        "    long cpf_i = %s;" % first,
        "    for (; cpf_i + %s <= %s; cpf_i += %s) {" % (s, last, s),
        "        for (long cpf_k = 0; cpf_k < %s; ++cpf_k) {" % s,
        "            cpf_acc[cpf_k] = %s;" % fold("cpf_acc[cpf_k]", "%s[cpf_i + cpf_k]" % i),
        "            %s[cpf_i + cpf_k] = cpf_acc[cpf_k];" % o,
        "        }",
        "    }",
        "    for (long cpf_k = 0; cpf_i + cpf_k < %s; ++cpf_k) %s[cpf_i + cpf_k] = %s;"
        % (last, o, fold("cpf_acc[cpf_k]", "%s[cpf_i + cpf_k]" % i)),
        "}",
    ]


def strided_rescan(
    o: str,
    i: str,
    s: str,
    name: str,
    span: tuple[str, str],
    fold: Callable[[str, str], str],
    acc_type: str,
    kind: str,
    distance: int | None,
) -> list[str]:
    """The recurrence ``o[j] = o[j - s] OP i[j]`` over ``span``, ``s >= 2``: the arm for a constant ``distance``, or a
    runtime test of ``s`` over :data:`STRIDED_RESCAN_ARMS`, each arm labeled ``kind`` with what the test found about
    the distance the program spells ``name``."""
    if distance is not None:
        arm = max(arm for arm in STRIDED_RESCAN_ARMS if arm[0] <= distance)
        least, what = arm[0], arm[2]
        label = "// %s -- scan, the carry distance is %d: %s" % (kind, distance, what.replace("K", str(distance), 1))
        return [label, *strided_rescan_arm(least, o, i, s, span, fold, acc_type)]
    tests = ("if (%s == 2) {" % s, "} else if (%s <= %d) {" % (s, STRIDED_REGISTER_CHAINS), "} else {")
    lines = []
    for test, (least, found, what) in zip(tests, STRIDED_RESCAN_ARMS):
        label = "// %s -- scan, the runtime check found %s: %s" % (
            kind,
            found.replace("K", name),
            what.replace("K", name, 1),
        )
        lines += [
            test,
            "    " + label,
            *("    " + line for line in strided_rescan_arm(least, o, i, s, span, fold, acc_type)),
        ]
    return [*lines, "}"]


def blocked_strided_scan(n: str, s: str, source: str, target: str, operation: str, seeds: str | None) -> str:
    """The inclusive scan ``target[j] = target[j - s] OP source[j]`` for a carry distance ``K = s``.

    ``K == 1`` is the unit-stride prefix scan, tested once per call. Above it the scan is blocked over contiguous
    chunks. A chunk is whole rows of ``s`` elements, so every residue class runs through it intact. Pass 1 folds each
    chunk but the last into its own last row, a combine carries those totals from chunk to chunk, and pass 2 rescans
    each chunk from the previous chunk's last row; the last rows are final after the combine, so pass 2 leaves them
    alone. Rounds of chunks keep a chunk cached between its two passes. One thread runs the recurrence once, as one
    chunk. Pass 2 takes the arm of :func:`strided_rescan` for ``K``: a constant ``K`` its one arm, a symbolic one a test
    per chunk, which is once per call on one thread. Class ``k`` starts at ``seeds[k] OP source[k]``, or at
    ``source[k]`` with no seeds.
    """

    def fold(a: str, b: str) -> str:
        return STRIDED_SCAN_COMBINE[operation].format(a=a, b=b)

    o, i, k = parenthesized(target), parenthesized(source), parenthesized(s)
    head = fold("%s[cpf_j]" % parenthesized(seeds), "%s[cpf_j]" % i) if seeds is not None else "%s[cpf_j]" % i
    first_row = fold("%s[cpf_k]" % parenthesized(seeds), "%s[cpf_k]" % i) if seeds is not None else "%s[cpf_k]" % i
    name = k[1:-1].strip()
    distance = int(name) if re.fullmatch(r"\d+", name) and int(name) >= 2 else None
    rescan = strided_rescan(
        o, i, "cpf_s", name, ("cpf_j", "cpf_e"), fold, "__typeof__(%s[0])" % o, "parallel", distance
    )
    blocked = [
        "  const long cpf_s = (long)%s;" % k,
        "  // threads in the team; 1 without OpenMP",
        "  long cpf_team = 0;",
        "  #pragma omp parallel reduction(+ : cpf_team)",
        "  cpf_team = 1;",
        "  const long cpf_len = (%d / cpf_s > 1 ? %d / cpf_s : 1) * cpf_s;" % (STRIDED_SCAN_CHUNK, STRIDED_SCAN_CHUNK),
        "  const long cpf_chunks = cpf_team > 1 ? (cpf_n + cpf_len - 1) / cpf_len : 1;",
        "  const long cpf_round = %d * cpf_team;" % STRIDED_SCAN_CHUNKS_PER_THREAD,
        "  #pragma omp parallel if (cpf_chunks > 1)",
        "  for (long cpf_t = 0; cpf_t < cpf_chunks; cpf_t += cpf_round) {",
        "      const long cpf_te = cpf_t + cpf_round < cpf_chunks ? cpf_t + cpf_round : cpf_chunks;",
        "      const long cpf_fe = cpf_te < cpf_chunks - 1 ? cpf_te : cpf_chunks - 1;",
        "      // pass 1: each chunk's per-class totals, into its last row",
        "      #pragma omp for schedule(static)",
        "      for (long cpf_c = cpf_t; cpf_c < cpf_fe; ++cpf_c) {",
        "          const long cpf_b = cpf_c * cpf_len, cpf_l = cpf_b + cpf_len - cpf_s;",
        "          for (long cpf_k = 0; cpf_k < cpf_s; ++cpf_k) {",
        "              %s[cpf_l + cpf_k] = cpf_c == 0 ? %s : %s[cpf_b + cpf_k];" % (o, first_row, i),
        "          }",
        "          for (long cpf_r = cpf_b + cpf_s; cpf_r <= cpf_l; cpf_r += cpf_s) {",
        "              for (long cpf_k = 0; cpf_k < cpf_s; ++cpf_k) {",
        "                  %s[cpf_l + cpf_k] = %s;" % (o, fold("%s[cpf_l + cpf_k]" % o, "%s[cpf_r + cpf_k]" % i)),
        "              }",
        "          }",
        "      }",
        "      // combine: carry the totals from chunk to chunk, one class per iteration",
        "      #pragma omp for schedule(static)",
        "      for (long cpf_k = 0; cpf_k < cpf_s; ++cpf_k) {",
        "          for (long cpf_c = cpf_t > 0 ? cpf_t : 1; cpf_c < cpf_fe; ++cpf_c) {",
        "              const long cpf_l = (cpf_c + 1) * cpf_len - cpf_s + cpf_k;",
        "              %s[cpf_l] = %s;" % (o, fold("%s[cpf_l - cpf_len]" % o, "%s[cpf_l]" % o)),
        "          }",
        "      }",
        "      // pass 2: rescan each chunk from the previous chunk's last row",
        "      #pragma omp for schedule(static)",
        "      for (long cpf_c = cpf_t; cpf_c < cpf_te; ++cpf_c) {",
        "          long cpf_j = cpf_c * cpf_len;",
        "          const long cpf_e = cpf_c == cpf_chunks - 1 ? cpf_n : cpf_j + cpf_len - cpf_s;",
        "          if (cpf_c == 0) {",
        "              for (; cpf_j < cpf_s && cpf_j < cpf_e; ++cpf_j) %s[cpf_j] = %s;" % (o, head),
        "          }",
        *("          " + line for line in rescan),
        "      }",
        "  }",
    ]
    lines = ["{ const long cpf_n = (long)%s;" % parenthesized(n)]
    if distance is not None:
        return "\n".join([*lines, *blocked, "}"])
    return "\n".join(
        [
            *lines,
            "  if (%s <= 0) std::abort();" % k,
            "  if (%s == 1) {" % k,
            "  // parallel -- scan, the runtime check found %s == 1: one chain, the unit-stride prefix scan" % name,
            "  if (cpf_n > 0) {",
            "      %s[0] = %s;" % (o, head.replace("cpf_j", "0")),
            "      ::dace::scan::detail::scan_incl_%s(%s, %s, 1L, cpf_n, %s[0]);" % (operation, i, o, o),
            "  }",
            "  } else {",
            *blocked,
            "  }",
            "}",
        ]
    )


def respell_scan_entry_points(code: str) -> str:
    """Spell the ``Scan`` CPU lowering's ``dace::scan`` calls in the vocabulary CPF already defines.

    A unit-stride ``inclusive_*`` / ``exclusive_*`` becomes CPF's ``scan_incl_*`` / ``scan_excl_*``, the
    runtime's blocked scan with its ``inscan`` block pass, the strided entry point becomes
    :func:`blocked_strided_scan`, and the affine entry points become their residue-class loops, so both
    dialects render them with no new definition.

    :param code: the body as the expansion wrote it.
    :returns: the body with every ``dace::scan`` entry point respelled.
    """

    def scan(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        kind, operation = match.group(1), match.group(2)
        if kind == "strided_inclusive":
            source, target, n, s, *seeds = arguments
            return blocked_strided_scan(n, s, source, target, operation, seeds[0] if seeds else None)
        first, last, target, seed = arguments
        n = last[len(first) :].strip().lstrip("+").strip() if last.startswith(first) else "%s - %s" % (last, first)
        return "::dace::scan::detail::scan_%s_%s(%s, %s, 0L, static_cast<long>%s, %s)" % (
            "incl" if kind == "inclusive" else "excl",
            operation,
            first,
            target,
            parenthesized(n),
            seed,
        )

    def affine(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        if match.group(1):
            coef, source, target, n, s, seeds = arguments
            seed = "0" if "zero_seeds" in seeds else "%s[cpf_r]" % seeds
        else:
            (coef, source, target, n, seed), s = arguments, "1"
        head = "%s[cpf_r] = %s[cpf_r] * (%s) + %s[cpf_r]" % (target, coef, seed, source)
        step = "%s[cpf_j] = %s[cpf_j] * %s[cpf_j - cpf_s] + %s[cpf_j]" % (target, coef, target, source)
        return residue_class_loop(n, s, head, step, parallel=bool(match.group(1)))

    code = c_rewrite_calls(code, SCAN_ENTRY_CALL, scan)
    return c_rewrite_calls(code, AFFINE_SCAN_ENTRY_CALL, affine)


def respell_reduce_entry_points(code: str) -> str:
    """Spell each ``dace::reduce::[seq::]<op>(in, count, stride, seed)`` as CPF's ``reduce_[seq_]<op>``.

    :param code: the body as the expansion wrote it.
    :returns: the body with every ``dace::reduce`` entry point respelled.
    """
    return REDUCE_ENTRY_CALL.sub(lambda m: "%s(" % reduce_name(m.group(2), m.group(1) is None), code)


def respell_copy_impl(code: str) -> str:
    """Spell the copy node's ``dace::CopyImpl<T, 1>(src, dst, count)`` as the ``std::memcpy`` it performs.

    The runtime template only differs from a byte copy for a type that is not trivially copyable, and every
    element type a CPF unit declares is a scalar.

    :param code: the body as the expansion wrote it.
    :returns: the body with every contiguous ``CopyImpl`` as a ``std::memcpy``.
    """

    def copy(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        if len(arguments) != 3:
            return None
        source, target, count = arguments
        return "std::memcpy(%s, %s, sizeof(%s) * %s)" % (target, source, match.group(1), parenthesized(count))

    return c_rewrite_calls(code, COPY_IMPL_CALL, copy)


#: A body's ``throw std::runtime_error("...");`` (FFTW3's unplannable layout), message and all.
C_THROW: "re.Pattern" = re.compile(r'\bthrow\s+(?:::)?std::runtime_error\s*\(\s*("(?:[^"\\\n]|\\.)*")\s*\)\s*;')


def c_throws(code: str) -> str:
    """A C body's ``throw std::runtime_error("msg");`` as ``abort()``, the message kept as a comment: C has no
    exceptions, and an error a library reports this way ends the program in C++ too unless the caller catches it.
    """
    return C_THROW.sub(lambda match: "{ /* %s */ abort(); }" % match.group(1).replace("*/", "* /"), code)


#: A function-scope static initialized by a call (FFTW3's ``static const int __threads_ready = fftw_init_threads();``).
C_STATIC_CALL: "re.Pattern" = re.compile(r"\bstatic\s+const\s+(int|long|bool)\s+(\w+)\s*=\s*(\w+\s*\([^;()]*\))\s*;")


def c_static_calls(code: str) -> str:
    """A C body's static initialized by a call, as a zero static assigned on first pass: C requires a constant
    initializer. A call that may return zero runs again on the next pass, which for FFTW's ``init_threads`` (and
    any idempotent setup) is harmless; the expansion makes the call inside a ``critical`` section."""
    return C_STATIC_CALL.sub(
        lambda match: (
            "static %s %s = 0; if (!%s) %s = %s;"
            % (match.group(1), match.group(2), match.group(2), match.group(2), match.group(3))
        ),
        code,
    )


#: A class line a library expansion writes into its own code, such as the label of a dispatch arm.
NATIVE_CLASS_LINE: "re.Pattern" = re.compile(r"^([ \t]*)// ((?:parallel|sequential|unknown) -- [^\n]*)\n", re.MULTILINE)


def native_class_lines(code: str) -> str:
    """``code`` with each class line an expansion wrote rendered at the hint detail in force, as a loop's hint is."""
    return NATIVE_CLASS_LINE.sub(lambda match: hint_comment(match.group(2), match.group(1)), code)


def rewrite_native_code(code: str, dialect: Dialect | None = None, site: NativeSite | None = None) -> str:
    """Rewrite the ``dace::`` names in a hand-written C++ body to their standalone spellings.

    Native tasklet bodies never reach the expression printers -- they are emitted verbatim -- so
    this is the only point at which a library expansion's own C++ can be re-spelled. Which is
    needed for the real cases: the ``Scan`` expansion calls ``::dace::scan::detail::scan_incl_sum``
    and the ``FindFirst`` expansion calls ``dace::find_first_index``, and CPF emits both functions
    itself rather than serializing a prefix sum or a cancelling search into a sequential loop.

    In C the same pass also rewrites the call shapes whose C function depends on the call site --
    the scans, the sort and the scatter guard's duplicate check are named for the element types
    ``site`` gives their arrays, the scan identities become constants, and the find-first over a
    lambda predicate becomes a function of its own (:func:`c_scan`, :func:`c_sort`,
    :func:`c_detect_collision`, :func:`c_scan_identities`, :func:`c_find_first`) -- and re-spells
    the ``std::`` names a body wrote directly (:func:`c_native_renames`).

    Textual by necessity, and deliberately conservative: only the qualified name is rewritten, only
    when the identifier is one CPF knows, and never with knowledge of the arguments. A
    ``dace::`` name with no standalone spelling is LEFT ALONE, so it reaches
    ``dace.codegen.cpf.verify`` and is reported against the construct that emitted it -- a silent
    partial rewrite would be worse than none.

    :param code: the C++ body as the expansion wrote it.
    :param dialect: which standalone dialect to rewrite for; ambient when omitted.
    :returns: the body with the names CPF can spell rewritten.
    :raises NotImplementedError: if the body names a type or function this dialect cannot express.
    """
    tables = tables_for(dialect)
    c_dialect = (dialect if dialect is not None else _active_dialect) is Dialect.STANDALONE_C
    code = POLICY_SORT_CALL.sub("std::sort(", code)
    code = rewrite_ctypes(respell_reduce_entry_points(respell_scan_entry_points(respell_copy_impl(code))), dialect)
    code = native_class_lines(code)

    if c_dialect:
        # A C++ standard header (``<algorithm>``) names nothing in C; the unit's C headers come from its preamble.
        code = C_CXX_INCLUDE.sub("", code)
        # The scan reads its seed's type from the identity call, so it runs before the identities.
        code = c_sort(c_detect_collision(c_reduce(c_scan(c_arg_pairs(code), site), site), site), site)
        code = c_native_renames(c_copy(c_numeric_limits(c_scan_identities(c_static_calls(c_throws(code))))))

    def replace(match: "re.Match") -> str:
        name = match.group(1)
        if name in tables.unsupported:
            raise NotImplementedError(f"CPF cannot lower {name!r}: {tables.unsupported[name]}.")
        if name in tables.inline_definitions:
            return name  # CPF emits this one's definition at the top of the unit
        if name in tables.std_renames:
            return tables.std_renames[name]
        return match.group(0)  # unknown: left for verify() to report

    code = _QUALIFIED_NAME.sub(replace, code)
    if c_dialect:
        # Last, so the predicate the search function takes is C already.
        code = c_find_first(c_cast_native_code(c_native_ipow(code, site)), site)
    return code


def helpers_used(code: str, dialect: Dialect | None = None) -> set[str]:
    """Which inline-definition helpers ``code`` calls.

    Recovered from the finished text rather than accumulated while printing, for two reasons. The
    symbolic printer is reached through memoized entry points, so per-printer state does not
    survive a cache hit and would under-report. And a translation unit is written by several
    emitters -- memlet subsets, tasklet bodies, loop bounds -- so scanning the result is the only
    place that sees all of them at once.

    :param code: the emitted translation unit, or any fragment of it.
    :param dialect: which standalone dialect emitted it; ambient when omitted.
    :returns: the helper names called, for :func:`definitions_for`.
    """
    return {match.group(1) for match in tables_for(dialect).helper_call.finditer(code)}


def required_definitions(names: set[str], dialect: Dialect | None = None) -> set[str]:
    """Close ``names`` over :data:`DEFINITION_DEPENDENCIES`.

    ``py_mod`` calls ``py_floor``, which calls ``int_floor_ni``: a unit that mentions only the
    first still has to carry all three.

    :param names: every function name the emitted code calls.
    :param dialect: which standalone dialect emitted them; ambient when omitted.
    :returns: every definition the unit needs, callers and callees alike.
    """
    tables = tables_for(dialect)
    needed: set[str] = set()
    pending = [name for name in names if definition_of(tables, name) is not None]
    while pending:
        name = pending.pop()
        if name in needed:
            continue
        needed.add(name)
        pending.extend(
            dependency for dependency in required(definition_of(tables, name))[1] if dependency not in needed
        )
    return needed


def definition_of(tables: "Tables", name: str) -> tuple[str, tuple[str, ...]] | None:
    """``(definition, dependencies)`` of a helper ``tables`` defines by name or as a typed instantiation,
    or ``None`` when it defines no such helper."""
    if name in tables.inline_definitions:
        return tables.inline_definitions[name], tables.definition_dependencies.get(name, ())
    return tables.instances(name)


def definitions_for(names: set[str], dialect: Dialect | None = None) -> tuple[str, ...]:
    """The inline definitions a translation unit calling ``names`` has to carry, callees first.

    C++ needs a function declared before it is called, so the order is a topological one over
    :data:`DEFINITION_DEPENDENCIES` rather than alphabetical -- emitting ``py_mod`` before
    ``py_floor`` would not compile. Ties break on the name, so the same input always produces
    byte-identical output.

    :param names: every function name the emitted code calls.
    :param dialect: which standalone dialect emitted them; ambient when omitted.
    :returns: the definitions to place at the top of the translation unit.
    :raises ValueError: if the dependencies contain a cycle, which no valid ordering satisfies.
    """
    tables = tables_for(dialect)
    needed = required_definitions(names, dialect)
    definitions = {name: definition_of(tables, name) for name in needed}
    emitted: list[str] = []
    placed: set[str] = set()
    while len(placed) < len(needed):
        ready = sorted(
            name
            for name in needed - placed
            if all(dependency in placed for dependency in required(definitions[name])[1] if dependency in needed)
        )
        if not ready:
            raise ValueError(f"CPF inline definitions have a dependency cycle among {sorted(needed - placed)}")
        for name in ready:
            emitted.append(required(definitions[name])[0])
            placed.add(name)
    return tuple(emitted)


def headers_for(names: set[str], dialect: Dialect | None = None) -> tuple[str, ...]:
    """The system headers a translation unit calling ``names`` has to include, in a stable order.

    Closed over the dependencies too: a unit calling only ``py_mod`` still ends up with
    ``py_floor``'s body, and that is what needs ``<type_traits>``.

    :param names: every function name the emitted code calls.
    :param dialect: which standalone dialect emitted them; ambient when omitted.
    :returns: the include list, base headers first.
    """
    tables = tables_for(dialect)
    extra: set[str] = set()
    for name in required_definitions(names, dialect) | set(names):
        extra.update(tables.definition_headers.get(name, ()))
    return tables.base_headers + tuple(sorted(extra - set(tables.base_headers)))


# The C dialect
#
# C23 has no templates, no function overloading and no ``constexpr`` on functions, and its maths
# library is not type-generic: ``sqrt(x)`` on a ``float`` promotes to ``double`` and rounds twice,
# which ``std::sqrt(float)`` does not. ``<tgmath.h>`` would fix the second problem and create a
# worse one -- its macros are named ``exp``, ``pow``, ``log``, ``round``, which is exactly the set
# of names a scientific SDFG gives its containers.
#
# So the printer resolves the argument types and names the function for them -- ``sqrtf`` for a
# ``float``, ``cpf_max_int64`` for two ``int64_t``. A helper that owns a loop is one typed
# ``static inline`` function per type combination a call site instantiates (see :func:`c_instance`).
# The unit defines no macro.

#: dace type name -> the C type a helper the printers call BY NAME is instantiated at. The printer
#: resolves its argument types and names the helper for the type C's own conversions give them, so
#: the call is a plain function call and no dispatch macro is emitted.
C_HELPER_TYPES: dict[str, str] = {
    "int32": "int32_t",
    "int64": "int64_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
}

#: Integer types narrower than ``int``, which C promotes to ``int`` before any arithmetic.
C_INTEGER_PROMOTIONS: dict[str, str] = {
    "bool": "int32",
    "bool_": "int32",
    "int8": "int32",
    "int16": "int32",
    "uint8": "int32",
    "uint16": "int32",
}

#: ``(conversion rank, signed)`` of each promoted integer type on an LP64 target.
C_INTEGER_RANKS: dict[str, tuple[int, bool]] = {
    "int32": (1, True),
    "uint32": (1, False),
    "int64": (2, True),
    "uint64": (2, False),
}

#: ``(rank of the real type, complex)`` of each floating type.
C_FLOATING_RANKS: dict[str, tuple[int, bool]] = {
    "float32": (1, False),
    "float64": (2, False),
    "complex64": (1, True),
    "complex128": (2, True),
}
C_FLOATING_BY_RANK: dict[tuple[int, bool], str] = {rank: name for name, rank in C_FLOATING_RANKS.items()}


def c_arithmetic(name: str) -> bool:
    """Whether ``name`` is a dace type :func:`c_common_type` can convert."""
    return name in C_INTEGER_PROMOTIONS or name in C_INTEGER_RANKS or name in C_FLOATING_RANKS


def c_symbol_type(name: str) -> str:
    """The type the C dialect picks a helper by for a SYMBOL of dace type ``name``.

    Every signed integer is taken at ``int64``. A symbol's recorded width is not the width the unit
    declares it with: one parsed from text carries ``DEFAULT_SYMBOL_TYPE``, and a nested function
    takes as ``int64_t`` a symbol its caller declares ``int``. A helper instantiated at the recorded
    width could narrow an ``int64_t`` value; widening a signed integer changes no value a narrower
    helper returns without overflowing.
    """
    return "int64" if name in C_INTEGER_PROMOTIONS or name in ("int32", "int64") else name


def c_integer_conversion(first: str, second: str) -> str:
    """The usual arithmetic conversion of two promoted integer types (C11 6.3.1.8) on LP64."""
    (first_rank, first_signed), (second_rank, second_signed) = C_INTEGER_RANKS[first], C_INTEGER_RANKS[second]
    if first_signed == second_signed:
        return first if first_rank >= second_rank else second
    signed, unsigned = (first, second) if first_signed else (second, first)
    # Only a strictly wider signed type represents every value of the unsigned one.
    return signed if C_INTEGER_RANKS[signed][0] > C_INTEGER_RANKS[unsigned][0] else unsigned


def c_common_type(types: tuple[str, ...]) -> str:
    """The type of ``(a0) + (a1) + ...`` in C for operands of the dace types ``types``.

    This is the type a ``_Generic`` selection on the sum of the arguments picks at compile time,
    computed at print time so the printer can name the typed helper directly.

    :param types: dace type names (``'int64'``, ``'float32'``), at least one.
    :returns: the dace type name of the result.
    :raises NotImplementedError: for a type with no C arithmetic spelling here.
    """
    promoted = [C_INTEGER_PROMOTIONS.get(name, name) for name in types]
    unknown = sorted({name for name in promoted if name not in C_INTEGER_RANKS and name not in C_FLOATING_RANKS})
    if unknown or not promoted:
        raise NotImplementedError(f"CPF cannot type C arithmetic over {list(types)}: {unknown} has no C spelling here")
    floating = [C_FLOATING_RANKS[name] for name in promoted if name in C_FLOATING_RANKS]
    if floating:
        return C_FLOATING_BY_RANK[(max(rank for rank, _ in floating), any(is_complex for _, is_complex in floating))]
    result = promoted[0]
    for other in promoted[1:]:
        result = c_integer_conversion(result, other)
    return result


def c_minmax_call(stem: str, arguments: tuple[str, ...], types: tuple[str | None, ...] | None) -> str:
    """A C ``Max``/``Min`` over two or more arguments, as nested calls to the typed binary helper.

    Nested left to right, and each call is instantiated at the type C's conversions give its two
    operands, the type C's usual arithmetic conversions give ``(a) + (b)``.

    :param stem: ``'cpf_max'`` or ``'cpf_min'``.
    :param arguments: already-printed argument expressions.
    :param types: the dace type name of each argument, as the printer resolved it.
    :returns: the nested call.
    :raises NotImplementedError: if an argument type is unknown or has no ordering.
    """
    if types is None or len(types) != len(arguments) or any(dtype is None for dtype in types):
        raise NotImplementedError(
            f"CPF cannot pick the C {stem} helper for ({', '.join(arguments)}): the printer "
            f"resolved the argument types as {types}"
        )
    nested, nested_type = arguments[0], types[0]
    for argument, argument_type in zip(arguments[1:], types[1:]):
        nested_type = c_common_type((nested_type, argument_type))
        if nested_type not in C_HELPER_TYPES:
            raise NotImplementedError(f"CPF cannot order {nested_type} values for {stem}: the type has no ordering")
        nested = "%s_%s(%s, %s)" % (stem, nested_type, nested, argument)
    return nested


#: ``(runtime name, C base name, family, arity)`` for every :data:`STD_RENAMES` entry that has a C
#: counterpart. The family names the set of functions the argument type picks between:
#:
#: ``real``
#:     ``<base>f`` for ``float``, ``<base>l`` for ``long double``, ``<base>`` otherwise -- which is
#:     also what an integer argument gets, matching ``std::sqrt(int) -> double``.
#: ``abs``
#:     the integer, floating and complex absolute values, which C spells with five different names.
#: ``complex``
#:     ``conjf`` / ``conj`` / ``conjl``.
#: ``elementary``
#:     ``real``, plus the ``<complex.h>`` counterpart (``cexp``, ``cpow``) for a complex argument, which
#:     the ``real`` association would silently convert to its real part.
#:
#: Arity 1 is picked by the PROMOTED argument type, so a ``short`` or an ``int8_t`` takes the ``int``
#: function. Arity 2 and 3 are picked by the type C's conversions give the arguments together, which
#: is the type the call converts them to anyway -- except ``frexp`` and ``ldexp``, whose second
#: argument is an ``int`` exponent, so the first argument alone picks (``first2`` arity 2).
C_MATH_SPEC: tuple[tuple[str, str, str, object], ...] = (
    ("Abs", "abs", "abs", 1),
    ("abs", "abs", "abs", 1),
    ("ceiling", "ceil", "real", 1),
    ("ceil", "ceil", "real", 1),
    ("floor", "floor", "real", 1),
    ("ROUND", "round", "real", 1),
    ("round", "round", "real", 1),
    ("conj", "conj", "complex", 1),
    ("exp2", "exp2", "real", 1),
    ("expm1", "expm1", "real", 1),
    ("log1p", "log1p", "real", 1),
    ("log2", "log2", "real", 1),
    ("frexp", "frexp", "real", "first2"),
    ("ldexp", "ldexp", "real", "first2"),
    ("ilogb", "ilogb", "real", 1),
    ("sin", "sin", "elementary", 1),
    ("cos", "cos", "elementary", 1),
    ("tan", "tan", "elementary", 1),
    ("asin", "asin", "elementary", 1),
    ("acos", "acos", "elementary", 1),
    ("atan", "atan", "elementary", 1),
    ("atan2", "atan2", "real", 2),
    ("sinh", "sinh", "elementary", 1),
    ("cosh", "cosh", "elementary", 1),
    ("tanh", "tanh", "elementary", 1),
    ("exp", "exp", "elementary", 1),
    ("fabs", "fabs", "real", 1),
    ("log", "log", "elementary", 1),
    ("log10", "log10", "real", 1),
    ("sqrt", "sqrt", "elementary", 1),
    ("cbrt", "cbrt", "real", 1),
    ("pow", "pow", "elementary", 2),
    ("fma", "fma", "real", 3),
    ("erf", "erf", "real", 1),
    ("erfc", "erfc", "real", 1),
    ("tgamma", "tgamma", "real", 1),
    ("lgamma", "lgamma", "real", 1),
    ("trunc", "trunc", "real", 1),
    ("hypot", "hypot", "real", 2),
    # Called unqualified in C++, so no STD_RENAMES entry; C must type it inside a Max (cloudsc_monolith).
    ("copysign", "copysign", "real", 2),
)

#: Runtime maths C already spells type-generically, as a ``<math.h>`` MACRO. Wrapping these in an
#: ``cpf_`` dispatch would be wrong as well as pointless: there is no ``isnanf`` to dispatch TO.
C_TYPE_GENERIC_MATH: dict[str, str] = {
    "isfinite": "isfinite",
    "isinf": "isinf",
    "isnan": "isnan",
    "signbit": "signbit",
}

#: The ``<math.h>`` / ``<complex.h>`` function each family names for the type that picks it, with
#: ``{base}`` the function's base name. The printer names the function itself, so a call reaches the
#: function a type-generic call on its argument type would select.
C_MATH_FUNCTIONS: dict[str, dict[str, str]] = {
    "real": {"float32": "{base}f"},
    "abs": {
        "int32": "abs",
        "int64": "labs",
        "float32": "fabsf",
        "float64": "fabs",
        "complex64": "cabsf",
        "complex128": "cabs",
    },
    "elementary": {"float32": "{base}f", "complex64": "c{base}f", "complex128": "c{base}"},
    "complex": {"complex64": "{base}f"},
    # A real argument has no imaginary part to read, and C's ``creal``/``cimag`` accept one, so the
    # double function keeps working for a complex-valued expression that folded to a real type.
    "complex_component": {"complex64": "{base}f"},
}

#: The function a family names for every other type: the ``double`` one, which an integer argument
#: converts to, and ``fabs`` for an unsigned integer's absolute value.
C_MATH_DEFAULTS: dict[str, str] = {
    "real": "{base}",
    "abs": "fabs",
    "elementary": "{base}",
    "complex": "{base}",
    "complex_component": "{base}",
}

#: Runtime function -> its C spelling, for the names C spells the same for every argument type.
C_STD_RENAMES: dict[str, str] = dict(C_TYPE_GENERIC_MATH)

#: Runtime function -> ``(C base name, family, arity)`` for every maths call C names by argument
#: type. ``re`` / ``im`` are the component accessors :data:`REWRITES` spells as member calls in C++.
C_TYPED_MATH: dict[str, tuple[str, str, object]] = {
    runtime: (base, family, arity) for runtime, base, family, arity in C_MATH_SPEC
}
C_TYPED_MATH.update({"re": ("creal", "complex_component", 1), "im": ("cimag", "complex_component", 1)})


def c_math_dispatch(name: str, types: tuple[str | None, ...] | None) -> tuple[str, str, str]:
    """``(C base name, family, the type that picks the function)`` for a call to ``name``.

    :raises NotImplementedError: if an argument type is unknown or has no C arithmetic spelling.
    """
    base, family, arity = C_TYPED_MATH[name]
    count = 2 if arity == "first2" else arity
    if types is not None and "complex128" in types:
        # The widest type C has: whatever an untyped operand is, the call computes in it
        # (contour_integral's ``np.power(z, slab_per_bc / 2 - n)``).
        types = tuple("complex128" if dtype is None else dtype for dtype in types)
    if types is None or len(types) != count or any(dtype is None for dtype in types):
        raise NotImplementedError(
            f"CPF cannot pick the C function for {name}: the printer resolved the argument types as {types}"
        )
    return base, family, c_common_type(types[:1] if count == 1 or arity == "first2" else types)


def c_math_function(name: str, types: tuple[str | None, ...] | None) -> str:
    """The C function a call to ``name`` over arguments of ``types`` names: ``sqrtf``, ``cexp``, ``labs``."""
    base, family, picking = c_math_dispatch(name, types)
    return C_MATH_FUNCTIONS[family].get(picking, C_MATH_DEFAULTS[family]).replace("{base}", base)


def c_math_result_type(name: str, types: tuple[str | None, ...] | None) -> str:
    """The dace type name of what the function :func:`c_math_function` names returns."""
    base, family, picking = c_math_dispatch(name, types)
    if base == "ilogb":
        return "int32"
    if family == "abs":
        return {"int32": "int32", "int64": "int64", "float32": "float32", "complex64": "float32"}.get(
            picking, "float64"
        )
    if family == "elementary" and picking in ("float32", "complex64", "complex128"):
        return picking
    if family == "complex":
        return "complex64" if picking == "complex64" else "complex128"
    if family == "complex_component":
        return "float32" if picking == "complex64" else "float64"
    return "float32" if picking == "float32" else "float64"


def pow_exponent_type(base_type: str | None, default: str) -> str:
    """The floating type a ``p/q`` exponent divides in: ``float32`` for a single-precision power, else ``float64``."""
    if base_type is None:
        return default
    return "float32" if c_math_result_type("pow", (base_type, base_type)) in ("float32", "complex64") else "float64"


#: ``Max``/``Min`` in C. Not the ``<stdlib.h>`` integer ``max``, which does not exist: CPF emits its
#: own typed pair (see :data:`C_TYPED_MINMAX_DEFINITIONS`).
C_VARIADIC_MINMAX: dict[str, str] = {"Max": "cpf_max", "Min": "cpf_min", "max": "cpf_max", "min": "cpf_min"}

#: :data:`REWRITES` in C. The component accessors and ``iround`` call a maths function whose C name
#: depends on the argument type, so :func:`lowering_for` answers them; ``np_float_pow`` converts both
#: operands to ``double``, which names ``pow`` outright.
C_REWRITES: dict[str, tuple[int, str]] = {
    name: rewrite for name, rewrite in REWRITES.items() if name not in C_TYPED_MATH and name != "iround"
}
C_REWRITES["np_float_pow"] = (2, "(pow((double)({0}), (double)({1})))")

#: The scalar types a cast can name once it has been through a SYMBOLIC expression. Sympy carries
#: an unknown function by name, so ``static_cast<int64_t>(i)`` comes back out as ``int64_t(i)`` --
#: a C++ functional cast, which C does not have. The expression printers are the only place that
#: spelling can be answered: the text is built from the sympy tree, never from a hand-written body
#: that :func:`c_cast_native_code` could reach.
#:
#: A closed list of the types with a C spelling. The ``dace::`` widths are deliberately absent, so
#: a cast to one is left for ``dace.codegen.cpf.verify`` to report against the construct that wrote
#: it rather than silently cast to something else.
C_SYMBOLIC_CAST_TYPES: tuple[str, ...] = (
    "bool",
    "char",
    "short",
    "int",
    "long",
    "long long",
    "unsigned int",
    "unsigned long",
    "unsigned long long",
    "float",
    "double",
    "long double",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "size_t",
)
C_REWRITES.update({name: (1, "((%s)({0}))" % name) for name in C_SYMBOLIC_CAST_TYPES})

#: ``dace::``-namespaced C++ type -> its C spelling. Only the two complex types differ from the C++
#: dialect's table: C spells a complex ``float _Complex`` rather than ``std::complex<float>``.
C_CTYPE_RENAMES: dict[str, str] = dict(CTYPE_RENAMES)
C_CTYPE_RENAMES.update(
    {
        "dace::complex64": "float _Complex",
        "dace::complex128": "double _Complex",
        "std::complex<float>": "float _Complex",
        "std::complex<double>": "double _Complex",
    }
)

#: ``b`` wins only by comparing STRICTLY better, so a tie -- and a comparison false because an
#: operand is NaN -- keeps ``a``. Same rule as the runtime's ``max``/``min``, which is what these
#: stand in for.
C_MINMAX_CONDITIONS: tuple[tuple[str, str], ...] = (("cpf_max", "a < b"), ("cpf_min", "b < a"))

#: ``cpf_max_<type>`` / ``cpf_min_<type>``: what a printed ``Max``/``Min`` calls by name (see
#: :func:`c_minmax_call`).
C_TYPED_MINMAX_DEFINITIONS: dict[str, str] = {
    "%s_%s" % (stem, name): "static inline %s %s_%s(%s a, %s b) { return (%s) ? b : a; }"
    % (ctype, stem, name, ctype, ctype, condition)
    for stem, condition in C_MINMAX_CONDITIONS
    for name, ctype in C_HELPER_TYPES.items()
}

#: ``dace::`` complex type -> the C function that builds one of its values from two components. C has
#: no functional cast, and ``re + im * I`` evaluates, so a NaN or infinite component would propagate.
C_COMPLEX_BUILDERS: dict[str, str] = {"dace::complex64": "cpf_complex64", "dace::complex128": "cpf_complex128"}

#: The C form of every :data:`INLINE_DEFINITIONS` entry a printer calls by name, plus the two
#: ``<numeric>`` functions C has no counterpart for at all (``gcd`` / ``lcm``, which are a rename in
#: C++ and a definition here). The helpers a native body calls with a loop of their own are typed
#: per instantiation instead (:func:`c_instance`). An out-parameter helper takes the address of the
#: lvalue the call names (:func:`c_helper_call`).
C_INLINE_DEFINITIONS: dict[str, str] = dict(C_TYPED_MINMAX_DEFINITIONS)
# C11 6.2.5p13: a complex has the representation of a two-element array of its real type.
C_INLINE_DEFINITIONS.update(
    {
        builder: (
            "static inline %s _Complex %s(%s re, %s im) { union { %s _Complex value; %s parts[2]; } z = "
            "{.parts = {re, im}}; return z.value; }" % ((real, builder) + (real,) * 4)
        )
        for real, builder in (("float", "cpf_complex64"), ("double", "cpf_complex128"))
    }
)

#: The dace types a typed helper is instantiated at, grouped the way a helper's bodies differ.
C_SIGNED_DTYPES: tuple[str, ...] = ("int32", "int64")
C_FLOATING_DTYPES: tuple[str, ...] = ("float32", "float64")
C_ARITHMETIC_DTYPES: tuple[str, ...] = C_SIGNED_DTYPES + C_FLOATING_DTYPES
C_COMPLEX_DTYPES: tuple[str, ...] = ("complex64", "complex128")

#: The C spelling of each complex dace type a typed helper is instantiated at.
C_COMPLEX_HELPER_TYPES: dict[str, str] = {"complex64": "float _Complex", "complex128": "double _Complex"}

#: The unsigned C type of each signed dace type's width, which a logical shift shifts through.
C_UNSIGNED_OF: dict[str, str] = {"int32": "uint32_t", "int64": "uint64_t"}

#: The ``<math.h>`` / ``<complex.h>`` suffix of each dace type whose body calls a maths function.
C_DTYPE_LIBM_SUFFIXES: dict[str, str] = {"float32": "f", "complex64": "f"}

#: A real value's sign in its own type. The comparisons are ``int``; the cast back is what keeps the
#: sign of a ``double`` a ``double``, so a following ``/ 2`` does not become integer division.
C_SIGN_BODY = "return ({T})((({T})0 < value) - (value < ({T})0));"

#: ``//`` and ``%`` are the two halves of ``py_divmod``, so the three cannot disagree about a sign,
#: a zero divisor or an infinity. ``%s`` names the half returned.
C_DIVMOD_HALF_BODY = (
    "{T} quotient;\n{T} remainder;\ncpf_py_divmod_{t}(numerator, denominator, &quotient, &remainder);\nreturn %s;"
)

#: Runtime helper -> ``(parameters, (types, return type, body) groups)``.
#:
#: A call picks its helper by the type C's conversions give the arguments of the ``{T}`` parameters.
#: ``ipow``'s exponent and a shift's amount are counts and an out-parameter receives a result, so
#: none of those picks; a ``{T} *`` parameter is passed the address of the lvalue the call names. A
#: helper exists only for the types its groups list, so an argument of any other type is refused at
#: print time. In a body ``{T}`` is the C type, ``{t}`` the dace type a sibling helper is named for,
#: ``{f}`` the ``<math.h>`` suffix and ``{U}`` the unsigned type of the same width.
C_TYPED_HELPER_SPECS: dict[str, tuple[tuple[tuple[str, str], ...], tuple[tuple[tuple[str, ...], str, str], ...]]] = {
    "int_ceil": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        ((C_SIGNED_DTYPES, "{T}", "return (numerator + denominator - 1) / denominator;"),),
    ),
    # Floored on the argument type itself, never a promoted one: the correction reads the sign of the
    # remainder, which a floating or unsigned instantiation would not have.
    "int_sqrt": (
        (("{T}", "n"),),
        (
            (
                C_SIGNED_DTYPES,
                "{T}",
                "if (n < 2) return n < 0 ? -1 : n;\n"
                "{T} x = n, y = (x + 1) / 2;\n"
                "while (y < x) {\n"
                "    x = y;\n"
                "    y = (x + n / x) / 2;\n"
                "}\n"
                "return x;",
            ),
        ),
    ),
    "int_floor_ni": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        (
            (
                C_SIGNED_DTYPES,
                "{T}",
                "{T} quotient = numerator / denominator;\n"
                "{T} remainder = numerator % denominator;\n"
                "return quotient - ((remainder != 0) && ((remainder < 0) != (denominator < 0)));",
            ),
        ),
    ),
    "py_floor": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        ((C_ARITHMETIC_DTYPES, "{T}", C_DIVMOD_HALF_BODY % "quotient"),),
    ),
    "py_mod": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        ((C_ARITHMETIC_DTYPES, "{T}", C_DIVMOD_HALF_BODY % "remainder"),),
    ),
    "c_mod": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        (
            (C_SIGNED_DTYPES, "{T}", "return numerator % denominator;"),
            (C_FLOATING_DTYPES, "{T}", "return fmod{f}(numerator, denominator);"),
        ),
    ),
    "ftn_mod": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        ((C_ARITHMETIC_DTYPES, "{T}", "return cpf_c_mod_{t}(numerator, denominator);"),),
    ),
    "ftn_modulo": (
        (("{T}", "numerator"), ("{T}", "denominator")),
        ((C_ARITHMETIC_DTYPES, "{T}", "return cpf_py_mod_{t}(numerator, denominator);"),),
    ),
    "ipow": (
        (("{T}", "base"), ("long long", "exponent")),
        (
            (
                C_ARITHMETIC_DTYPES,
                "{T}",
                "{T} result = 1;\nwhile (exponent > 0) {\n    if (exponent & 1) { result *= base; }\n"
                "    base *= base;\n    exponent >>= 1;\n}\nreturn result;",
            ),
        ),
    ),
    "logical_left_shift": (
        (("{T}", "value"), ("int", "amount")),
        ((C_SIGNED_DTYPES, "{T}", "return ({T})(({U})(value) << amount);"),),
    ),
    "logical_right_shift": (
        (("{T}", "value"), ("int", "amount")),
        ((C_SIGNED_DTYPES, "{T}", "return ({T})(({U})(value) >> amount);"),),
    ),
    "gcd": (
        (("{T}", "a"), ("{T}", "b")),
        (
            (
                C_SIGNED_DTYPES,
                "{T}",
                "{T} x = a < 0 ? -a : a;\n{T} y = b < 0 ? -b : b;\nwhile (y != 0) {\n"
                "    {T} t = x % y;\n    x = y;\n    y = t;\n}\nreturn x;",
            ),
        ),
    ),
    "lcm": (
        (("{T}", "a"), ("{T}", "b")),
        (
            (
                C_SIGNED_DTYPES,
                "{T}",
                "{T} divisor = cpf_gcd_{t}(a, b);\n{T} product;\nif (divisor == 0) { return 0; }\n"
                "product = (a / divisor) * b;\nreturn product < 0 ? -product : product;",
            ),
        ),
    ),
    "sign": ((("{T}", "value"),), ((C_ARITHMETIC_DTYPES, "{T}", C_SIGN_BODY),)),
    "sgn": ((("{T}", "value"),), ((C_ARITHMETIC_DTYPES, "{T}", C_SIGN_BODY),)),
    "sign_numpy_2": (
        (("{T}", "value"),),
        (
            (C_ARITHMETIC_DTYPES, "{T}", C_SIGN_BODY),
            (
                C_COMPLEX_DTYPES,
                "{T}",
                "return (creal{f}(value) != 0 && cimag{f}(value) != 0) ? value / cabs{f}(value) : 0;",
            ),
        ),
    ),
    # Integral input is already floored, so it comes back unchanged -- narrowing an int64 through
    # ``(int)floor(...)`` would truncate it to 32 bits.
    "ifloor": (
        (("{T}", "value"),),
        ((C_SIGNED_DTYPES, "{T}", "return value;"), (C_FLOATING_DTYPES, "int", "return (int)floor{f}(value);")),
    ),
    "cpp_divmod": (
        (("{T}", "numerator"), ("{T}", "denominator"), ("{T} *", "quotient"), ("{T} *", "remainder")),
        (
            (
                C_SIGNED_DTYPES,
                "void",
                "*quotient = ({T})(numerator / denominator);\n*remainder = ({T})(numerator % denominator);",
            ),
        ),
    ),
    # ``x / -1`` is ``-x`` with remainder 0, negated through the unsigned type so ``MIN / -1`` wraps
    # to ``MIN`` instead of trapping; ``x / 0`` is 0 remainder 0. Both are numpy's answers.
    "py_divmod": (
        (("{T}", "numerator"), ("{T}", "denominator"), ("{T} *", "quotient"), ("{T} *", "remainder")),
        (
            (
                C_SIGNED_DTYPES,
                "void",
                "if (denominator == 0) {\n"
                "    *quotient = 0;\n"
                "    *remainder = 0;\n"
                "} else if (denominator == -1) {\n"
                "    *quotient = ({T})(({U})0 - ({U})numerator);\n"
                "    *remainder = 0;\n"
                "} else {\n"
                "    *quotient = numerator / denominator;\n"
                "    *remainder = numerator % denominator;\n"
                "    if (*remainder != 0 && ((*remainder < 0) != (denominator < 0))) {\n"
                "        *quotient -= 1;\n"
                "        *remainder += denominator;\n"
                "    }\n"
                "}",
            ),
            (
                C_FLOATING_DTYPES,
                "void",
                "{T} ratio = numerator / denominator;\n"
                "*remainder = fmod{f}(numerator, denominator);\n"
                "if (denominator == 0) {\n"
                "    *quotient = ratio;\n"
                "    return;\n"
                "}\n"
                "*quotient = (numerator - *remainder) / denominator;\n"
                "if (*remainder == 0) {\n"
                "    *remainder = copysign{f}(({T})0, denominator);\n"
                "} else if ((denominator < 0) != (*remainder < 0)) {\n"
                "    *remainder += denominator;\n"
                "    *quotient -= 1;\n"
                "}\n"
                "if (*quotient == 0) {\n"
                "    *quotient = copysign{f}(({T})0, ratio);\n"
                "} else {\n"
                "    {T} floored = floor{f}(*quotient);\n"
                "    *quotient = *quotient - floored > ({T})0.5 ? floored + 1 : floored;\n"
                "}",
            ),
        ),
    ),
    "np_modf": (
        (("{T}", "value"), ("{T} *", "integral"), ("{T} *", "fractional")),
        (
            (C_SIGNED_DTYPES, "void", "*integral = value;\n*fractional = 0;"),
            (C_FLOATING_DTYPES, "void", "*fractional = modf{f}(value, integral);"),
        ),
    ),
    "np_frexp": (
        (("{T}", "value"), ("{T} *", "mantissa"), ("int *", "exponent")),
        ((C_FLOATING_DTYPES, "void", "*mantissa = frexp{f}(value, exponent);"),),
    ),
}

# ``heaviside`` has two arities, which one C function cannot have: each is its own helper, and the
# call's argument count picks between them at print time.
C_TYPED_HELPER_SPECS.update(
    {
        "heaviside_1": (
            (("{T}", "value"),),
            ((C_ARITHMETIC_DTYPES, "{T}", "return (value > ({T})0) ? ({T})1 : ({T})0;"),),
        ),
        "heaviside_2": (
            (("{T}", "value"), ("{T}", "at_zero")),
            (
                (
                    C_ARITHMETIC_DTYPES,
                    "{T}",
                    "return (value < ({T})0) ? ({T})0 : ((value > ({T})0) ? ({T})1 : at_zero);",
                ),
            ),
        ),
    }
)

#: Runtime helper -> argument count -> the :data:`C_TYPED_HELPER_SPECS` entry for that arity.
C_HELPER_ARITIES: dict[str, dict[int, str]] = {"heaviside": {1: "heaviside_1", 2: "heaviside_2"}}


def c_typed_helpers() -> tuple[dict[str, str], dict[str, tuple[str, ...]], dict[str, tuple[str, ...]], dict[str, str]]:
    """``(definitions, dependencies, types per helper, result types)`` for :data:`C_TYPED_HELPER_SPECS`.

    Each instantiation is one ``static inline`` function named ``cpf_<helper>_<dace type>``, which is
    what a printer that resolved its argument types calls. A ``void`` helper has no result type.
    """
    definitions: dict[str, str] = {}
    dependencies: dict[str, tuple[str, ...]] = {}
    available: dict[str, tuple[str, ...]] = {}
    results: dict[str, str] = {}
    for name, (parameters, groups) in C_TYPED_HELPER_SPECS.items():
        for group_types, returns, body in groups:
            for dtype in group_types:
                ctype = C_HELPER_TYPES.get(dtype) or C_COMPLEX_HELPER_TYPES[dtype]
                helper = "cpf_%s_%s" % (name, dtype)
                filled = (
                    body.replace("{T}", ctype)
                    .replace("{t}", dtype)
                    .replace("{f}", C_DTYPE_LIBM_SUFFIXES.get(dtype, ""))
                    .replace("{U}", C_UNSIGNED_OF.get(dtype, ""))
                )
                declared = ", ".join(ptype.replace("{T}", ctype) + " " + pname for ptype, pname in parameters)
                statements = "\n".join("    " + line if line.strip() else line for line in filled.split("\n"))
                definitions[helper] = "static inline %s %s(%s) {\n%s\n}" % (
                    returns.replace("{T}", ctype),
                    helper,
                    declared,
                    statements,
                )
                dependencies[helper] = tuple(re.findall(r"\b(cpf_\w+)\(", filled))
                available[name] = available.get(name, ()) + (dtype,)
                if returns != "void":
                    results[helper] = dtype if returns == "{T}" else {"int": "int32"}[returns]
    return definitions, dependencies, available, results


C_TYPED_HELPER_DEFINITIONS, C_TYPED_HELPER_DEPENDENCIES, C_TYPED_HELPER_TYPES, C_TYPED_HELPER_RETURNS = (
    c_typed_helpers()
)
C_INLINE_DEFINITIONS.update(C_TYPED_HELPER_DEFINITIONS)

#: Typed helper name -> the dace type its call evaluates to, for a printer typing a nested call.
C_TYPED_HELPER_RESULTS: dict[str, str] = {
    **{helper: helper.rsplit("_", 1)[1] for helper in C_TYPED_MINMAX_DEFINITIONS},
    **C_TYPED_HELPER_RETURNS,
}


def c_helper_spec(name: str, count: int) -> str | None:
    """The :data:`C_TYPED_HELPER_SPECS` entry a call to ``name`` with ``count`` arguments names, or ``None``.

    :raises NotImplementedError: for an arity a split helper has no entry for.
    """
    if name in C_HELPER_ARITIES:
        if count not in C_HELPER_ARITIES[name]:
            raise NotImplementedError(
                f"CPF has no C {name} taking {count} arguments; it takes {sorted(C_HELPER_ARITIES[name])}"
            )
        return C_HELPER_ARITIES[name][count]
    return name if name in C_TYPED_HELPER_SPECS else None


def c_helper_operand_types(name: str, types: tuple[str | None, ...] | None) -> tuple[str | None, ...] | None:
    """The types of the arguments that pick ``name``'s helper, or ``None`` when the count is wrong."""
    parameters = C_TYPED_HELPER_SPECS[name][0]
    if types is None or len(types) != len(parameters):
        return None
    return tuple(dtype for (ptype, _), dtype in zip(parameters, types) if ptype == "{T}")


def c_helper_dispatch(name: str, types: tuple[str | None, ...] | None) -> str:
    """The dace type the typed helper ``name`` is instantiated at for arguments of ``types``.

    :raises NotImplementedError: if a picking argument's type is unknown, or no helper takes the type.
    """
    operands = c_helper_operand_types(name, types)
    if operands is None or any(dtype is None for dtype in operands):
        raise NotImplementedError(
            f"CPF cannot pick the C {name} helper: the printer resolved the argument types as {types}"
        )
    dtype = c_common_type(operands)
    if dtype not in C_TYPED_HELPER_TYPES[name]:
        raise NotImplementedError(f"CPF has no C {name} for {dtype} arguments; it takes {C_TYPED_HELPER_TYPES[name]}")
    return dtype


def c_helper_result_type(name: str, types: tuple[str | None, ...] | None) -> str | None:
    """The dace type a call to ``name`` evaluates to, or ``None`` if its operand types are unknown.

    :raises NotImplementedError: if no helper takes the operands' type (see :func:`c_helper_dispatch`).
    """
    operands = c_helper_operand_types(name, types)
    if operands is None or any(dtype is None for dtype in operands):
        return None
    return C_TYPED_HELPER_RESULTS.get("cpf_%s_%s" % (name, c_helper_dispatch(name, types)))


def c_helper_call(name: str, arguments: tuple[str, ...], types: tuple[str | None, ...] | None) -> str:
    """The call to the typed helper :func:`c_helper_dispatch` picks for ``name``; an out-parameter is
    passed the address of the lvalue the call site names."""
    dtype = c_helper_dispatch(name, types)
    passed = (
        "&(%s)" % argument if ptype.endswith("*") else argument
        for (ptype, _), argument in zip(C_TYPED_HELPER_SPECS[name][0], arguments)
    )
    return "cpf_%s_%s(%s)" % (name, dtype, ", ".join(passed))


#: dace type name -> its C spelling, for every scalar a typed instantiation can be made at.
C_SCALAR_SPELLINGS: dict[str, str] = {
    "bool": "bool",
    "int8": "int8_t",
    "int16": "int16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "uint16": "uint16_t",
    "uint32": "uint32_t",
    "uint64": "uint64_t",
    "float32": "float",
    "float64": "double",
    "complex64": "float _Complex",
    "complex128": "double _Complex",
}


def c_scan_definition(name: str, kind: str, operation: str, source: str, target: str, accumulator: str) -> str:
    """One prefix scan as a C function over its three independent types.

    Input, output and accumulator types are independent, which is why the C++ form is a template over
    ``<It, OutIt, T>``: the compaction a prefix sum exists for scans an ``int8_t`` 0/1 mask into
    ``int64_t`` ranks. The accumulator takes the SEED's type, never the input's -- folded at ``int8_t``
    the mask wraps at 128 -- and is a non-const local, so a ``const`` seed still gives OpenMP a
    reduction variable it can write.

    :param name: the instantiation's name.
    :param kind: ``'incl'`` or ``'excl'``. The ``scan`` directive splits the body into an input phase
                 and a scan phase, and ``exclusive`` names them the other way round.
    :param operation: a :data:`SCAN_OPERATIONS` key.
    :param source: the input element's dace type.
    :param target: the output element's dace type.
    :param accumulator: the seed's dace type.
    :returns: the definition.
    :raises NotImplementedError: for a min or max over a complex accumulator, which has no order.
    """
    if operation in ("min", "max") and accumulator in C_COMPLEX_DTYPES:
        raise NotImplementedError(f"CPF cannot render a {operation} scan over {accumulator}: the type has no order")
    ctype = C_SCALAR_SPELLINGS[accumulator]
    return "\n".join(
        (
            "#ifdef _OPENMP",
            "#include <omp.h>",
            "#endif",
            "static inline void %s(const %s *f, %s *o, long lo, long hi, %s seed) {"
            % (name, C_SCALAR_SPELLINGS[source], C_SCALAR_SPELLINGS[target], ctype),
            *blocked_scan_body(kind, operation, ctype, c_dialect=True),
            "}",
        )
    )


#: The duplicate check's first lines: a guard (``parallel`` true) whose team has one thread runs its scatter in
#: order either way, which is correct for any indices, so it reports a collision, the never-unsafe answer, and takes
#: the sequential path without the check's two passes over the indices (E5 ``scatter``: 3.5x at one thread). A
#: nested caller (``parallel`` false) still gets the exact answer.
COLLISION_SERIAL_EXIT = (
    "#ifdef _OPENMP",
    "    if (parallel && omp_get_max_threads() == 1) return 1;",
    "#else",
    "    if (parallel) return 1;",
    "#endif",
)
OPENMP_HEADER = ("#ifdef _OPENMP", "#include <omp.h>", "#endif")


def c_detect_collision_definition(name: str, index: str, tag: str) -> str:
    """The scatter guard's duplicate check as a C function, typed for its index and tag arrays.

    :param name: the instantiation's name.
    :param index: the index array's dace element type.
    :param tag: the tag array's dace element type.
    :returns: the definition.
    """
    return "\n".join(
        (
            "// Tagged-write + verify: pass 1 writes owner[idx[i]] = i, pass 2 ORs owner[idx[i]] != i. A slot",
            "// only one writer won reads back a different i. Indices outside [0, capacity) are skipped: the",
            "// guarded scatter never writes those slots.",
            *OPENMP_HEADER,
            "static inline long long %s(const %s *idx, long long n, %s *owner, long long capacity, bool parallel) {"
            % (name, C_SCALAR_SPELLINGS[index], C_SCALAR_SPELLINGS[tag]),
            *COLLISION_SERIAL_EXIT,
            "    long long c = 0;",
            "    #pragma omp parallel for if (parallel : parallel)",
            "    for (long long i = 0; i < n; ++i) {",
            "        const long long v = (long long)idx[i];",
            "        if (v >= 0 && v < capacity) owner[v] = (%s)i;" % C_SCALAR_SPELLINGS[tag],
            "    }",
            "    #pragma omp parallel for simd if (parallel : parallel) reduction(| : c)",
            "    for (long long i = 0; i < n; ++i) {",
            "        const long long v = (long long)idx[i];",
            "        if (v >= 0 && v < capacity) c |= ((long long)owner[v] != i) ? 1LL : 0LL;",
            "    }",
            "    return c;",
            "}",
        )
    )


def c_detect_collision_sized_definition(name: str, index: str, tagged: str) -> str:
    """The duplicate check sizing its own ``int64_t`` tag buffer from ``max(idx)``.

    A failed allocation reports a collision, the answer that is never unsafe: the guard then takes its
    sequential path (as does a scatter that cannot run in parallel, :data:`COLLISION_SERIAL_EXIT`).

    :param name: the instantiation's name.
    :param index: the index array's dace element type.
    :param tagged: the tag-array instantiation it calls.
    :returns: the definition.
    """
    return "\n".join(
        (
            *OPENMP_HEADER,
            "static inline long long %s(const %s *idx, long long n, bool parallel) {"
            % (name, C_SCALAR_SPELLINGS[index]),
            *COLLISION_SERIAL_EXIT,
            "    long long mx = 0;",
            "    #pragma omp parallel for simd if (parallel : parallel) reduction(max : mx)",
            "    for (long long i = 0; i < n; ++i) {",
            "        const long long v = (long long)idx[i];",
            "        mx = v > mx ? v : mx;",
            "    }",
            "    int64_t *owner = (int64_t *)malloc((size_t)(mx + 1) * sizeof(int64_t));",
            "    if (owner == NULL) return 1;",
            "    const long long c = %s(idx, n, owner, mx + 1, parallel);" % tagged,
            "    free(owner);",
            "    return c;",
            "}",
        )
    )


def c_sort_definition(name: str, element: str) -> str:
    """``std::sort`` over a contiguous range as a C function: an OpenMP task mergesort.

    The range halves into tasks down to a leaf of about a quarter of one thread's share, each leaf
    heapsorts (the comparison stays inline and typed, where ``qsort`` pays an indirect call per
    comparison), and each join merges through one buffer allocated once. Without OpenMP the pragmas
    are ignored and the same recursion runs on one thread.

    :param name: the instantiation's name.
    :param element: the range's dace element type.
    :returns: the definition.
    :raises NotImplementedError: for a complex element, which ``<`` does not order.
    """
    if element in C_COMPLEX_DTYPES:
        raise NotImplementedError(f"CPF cannot sort {element} values in C: the type has no order")
    ctype = C_SCALAR_SPELLINGS[element]
    heap, merge, sort_range = "cpf_heapsort_%s" % element, "cpf_merge_%s" % element, "cpf_sortrange_%s" % element
    return "\n".join(
        (
            "#ifdef _OPENMP",
            "#include <omp.h>",
            "#endif",
            "static inline void %s(%s *first, %s *last) {" % (heap, ctype, ctype),
            "    const long long len = (long long)(last - first);",
            "    long long start = len / 2;",
            "    long long end = len;",
            "    while (end > 1) {",
            "        %s tmp;" % ctype,
            "        long long root;",
            "        long long child;",
            "        if (start > 0) {",
            "            --start;",
            "        } else {",
            "            --end;",
            "            tmp = first[0];",
            "            first[0] = first[end];",
            "            first[end] = tmp;",
            "        }",
            "        root = start;",
            "        while ((child = 2 * root + 1) < end) {",
            "            if (child + 1 < end && first[child] < first[child + 1]) ++child;",
            "            if (!(first[root] < first[child])) break;",
            "            tmp = first[root];",
            "            first[root] = first[child];",
            "            first[child] = tmp;",
            "            root = child;",
            "        }",
            "    }",
            "}",
            "// Merges the sorted halves [first, middle) and [middle, last) in place through buffer.",
            "static inline void %s(%s *first, %s *middle, %s *last, %s *buffer) {"
            % (merge, ctype, ctype, ctype, ctype),
            "    const long long left = (long long)(middle - first);",
            "    long long i = 0;",
            "    %s *out = first;" % ctype,
            "    memcpy(buffer, first, (size_t)left * sizeof(%s));" % ctype,
            "    while (i < left && middle < last) *out++ = (*middle < buffer[i]) ? *middle++ : buffer[i++];",
            "    while (i < left) *out++ = buffer[i++];",
            "}",
            "static inline void %s(%s *first, %s *last, %s *buffer, long long leaf) {"
            % (sort_range, ctype, ctype, ctype),
            "    const long long len = (long long)(last - first);",
            "    %s *middle = first + len / 2;" % ctype,
            "    if (len <= leaf) {",
            "        %s(first, last);" % heap,
            "        return;",
            "    }",
            "    #pragma omp task",
            "    %s(first, middle, buffer, leaf);" % sort_range,
            "    %s(middle, last, buffer + len / 2, leaf);" % sort_range,
            "    #pragma omp taskwait",
            "    %s(first, middle, last, buffer);" % merge,
            "}",
            "static inline void %s(%s *first, %s *last) {" % (name, ctype, ctype),
            "    const long long len = (long long)(last - first);",
            "    long long threads = 1;",
            "    long long leaf;",
            "    %s *buffer;" % ctype,
            "#ifdef _OPENMP",
            "    threads = (long long)omp_get_max_threads();",
            "#endif",
            "    leaf = len / (4 * threads);",
            "    if (leaf < 4096) leaf = 4096;",
            "    buffer = len > leaf ? (%s *)malloc((size_t)len * sizeof(%s)) : NULL;" % (ctype, ctype),
            "    if (buffer == NULL) {",
            "        %s(first, last);" % heap,
            "        return;",
            "    }",
            "    #pragma omp parallel",
            "    #pragma omp single nowait",
            "    %s(first, last, buffer, leaf);" % sort_range,
            "    free(buffer);",
            "}",
        )
    )


#: One alternation over the dace types :data:`C_SCALAR_SPELLINGS` spells, longest first.
C_DTYPE_ALTERNATION: str = "|".join(sorted(C_SCALAR_SPELLINGS, key=len, reverse=True))

#: The name of a typed instantiation: its family, then one dace type per type parameter
#: (``cpf_scan_incl_sum_int8_int64_int64``). No group captures, so it can join another alternation.
C_INSTANCE_NAMES: str = (
    r"cpf_(?:scan_(?:incl|excl)_(?:sum|product|min|max)|reduce_(?:seq_)?(?:%s)|arg_pair_of"
    r"|arg_(?:max|min)(?:_last)?_(?:step|combine)|atomic_max|atomic_min|detect_collision_sized"
    r"|detect_collision|sort)(?:_(?:%s))+" % ("|".join(REDUCE_OPERATIONS), C_DTYPE_ALTERNATION)
)

#: :data:`C_INSTANCE_NAMES` with the family, the scan's kind and operation, and the type list captured.
C_INSTANCE_NAME: "re.Pattern" = re.compile(
    r"cpf_(scan_(incl|excl)_(sum|product|min|max)|detect_collision_sized"
    r"|detect_collision|sort)((?:_(?:%s))+)" % C_DTYPE_ALTERNATION
)


#: A typed argmax / argmin piece: its family, then the value's and the index's dace types.
C_ARG_INSTANCE_NAME: "re.Pattern" = re.compile(
    r"cpf_(arg_pair_of|arg_(?:max|min)(?:_last)?_(?:step|combine))_(%s)_(%s)"
    % (C_DTYPE_ALTERNATION, C_DTYPE_ALTERNATION)
)


def c_arg_pair_definition(value: str, index: str) -> str:
    """The (value, index) pair of ``value`` x ``index`` in C: its struct, its order and its constructor, as
    ``dace/arg_reduction.h`` declares them. A complex value orders lexicographically, real part first."""
    v, i = C_SCALAR_SPELLINGS[value], C_SCALAR_SPELLINGS[index]
    pair, tag = f"cpf_arg_pair_{value}_{index}", f"{value}_{index}"
    if value in C_COMPLEX_DTYPES:
        greater = "creal(a) > creal(b) || (creal(a) == creal(b) && cimag(a) > cimag(b))"
    else:
        greater = "a > b"
    return "\n".join(
        [
            f"typedef struct {{ {v} __ar_v; {i} __ar_i; }} {pair};",
            f"static inline int cpf_arg_greater_{tag}({v} a, {v} b) {{ return {greater}; }}",
            f"static inline {pair} cpf_arg_pair_of_{tag}({v} v, {i} i) {{ {pair} p = {{v, i}}; return p; }}",
        ]
    )


def c_arg_combine_definition(kind: str, value: str, index: str) -> str:
    """Two threads' argmax / argmin pairs merged into one: the better value wins, a tie goes to the lower index (the
    higher one for a ``_last`` kind), and a NaN pair wins as the sequential scan keeps it."""
    pair, tag = f"cpf_arg_pair_{value}_{index}", f"{value}_{index}"
    op = kind.removesuffix("_last")
    better = (
        f"cpf_arg_greater_{tag}(in.__ar_v, out.__ar_v)"
        if op == "max"
        else f"cpf_arg_greater_{tag}(out.__ar_v, in.__ar_v)"
    )
    wins_tie = ">" if kind != op else "<"
    return "\n".join(
        [
            f"static inline {pair} cpf_arg_{kind}_combine_{tag}({pair} out, {pair} in) {{",
            "    if (!(out.__ar_v == out.__ar_v) || !(in.__ar_v == in.__ar_v))",
            "        return out.__ar_v == out.__ar_v ? in : out;",
            f"    return ({better} || (in.__ar_v == out.__ar_v && in.__ar_i {wins_tie} out.__ar_i)) ? in : out;",
            "}",
        ]
    )


def c_arg_step_definition(kind: str, value: str, index: str) -> str:
    """One thread's argmax / argmin step over the pair of ``value`` x ``index``: a strictly better value wins, and
    an equal one too for a ``_last`` kind."""
    v, i, tag = C_SCALAR_SPELLINGS[value], C_SCALAR_SPELLINGS[index], f"{value}_{index}"
    pair = f"cpf_arg_pair_{tag}"
    op = kind.removesuffix("_last")
    better = f"cpf_arg_greater_{tag}(v, held.__ar_v)" if op == "max" else f"cpf_arg_greater_{tag}(held.__ar_v, v)"
    if kind != op:
        better += " || v == held.__ar_v"
    return (
        f"static inline {pair} cpf_arg_{kind}_step_{tag}({pair} held, {v} v, {i} i) {{\n"
        f"    return {better} ? cpf_arg_pair_of_{tag}(v, i) : held;\n}}"
    )


#: A typed reduction: ``seq_`` or not, the operation, then the input's and the seed's dace types.
C_REDUCE_INSTANCE_NAME: "re.Pattern" = re.compile(
    r"cpf_reduce_(seq_)?(%s)_(%s)_(%s)" % ("|".join(REDUCE_OPERATIONS), C_DTYPE_ALTERNATION, C_DTYPE_ALTERNATION)
)


def c_reduce_definition(name: str, operation: str, parallel: bool, source: str, accumulator: str) -> str:
    """One ``dace::reduce`` entry as a C function over its input and seed types; the accumulator takes the seed's.

    A complex sum or product reduces through a combiner declared in the function; any other complex fold runs
    sequentially, OpenMP having no reduction for it.

    :raises NotImplementedError: for a min or max over a complex accumulator, which has no order.
    """
    if operation in ("min", "max") and accumulator in C_COMPLEX_DTYPES:
        raise NotImplementedError(
            f"CPF cannot render a {operation} reduction over {accumulator}: the type has no order"
        )
    ctype = C_SCALAR_SPELLINGS[accumulator]
    is_complex = accumulator in C_COMPLEX_DTYPES
    declared = is_complex and operation in COMPLEX_REDUCE_COMBINERS
    body = reduce_body(
        operation,
        ctype,
        parallel and (declared or not is_complex),
        c_dialect=True,
        complex_initial="{identity}" if declared else "",
        simd=not is_complex,
    )
    return "\n".join(
        (
            "static inline %s %s(const %s *in, long count, long stride, %s seed) {"
            % (ctype, name, C_SCALAR_SPELLINGS[source], ctype),
            *body,
            "}",
        )
    )


@functools.lru_cache(maxsize=None, typed=True)
def c_instance(name: str) -> tuple[str, tuple[str, ...]] | None:
    """``(definition, dependencies)`` of the typed C instantiation ``name``, or ``None`` for any other name.

    A helper that owns a loop is one function per type combination a call site uses, and its name
    carries the combination, so the definition is recovered from the finished text exactly as
    :func:`helpers_used` recovers every other helper.

    :raises NotImplementedError: for a combination the family cannot order.
    """
    atomic = re.fullmatch(r"cpf_atomic_(max|min)_(%s)" % C_DTYPE_ALTERNATION, name)
    if atomic is not None:
        op, dtype = atomic.groups()
        ctype = C_SCALAR_SPELLINGS[dtype]
        lines = [f"static inline void {name}({ctype} *x, {ctype} v) {{", *atomic_minmax_body(op, ctype), "}"]
        return "\n".join(lines), ()
    piece = C_ARG_INSTANCE_NAME.fullmatch(name)
    if piece is not None:
        family, value, index = piece.groups()
        if family == "arg_pair_of":
            return c_arg_pair_definition(value, index), ()
        if family.endswith("_combine"):
            return c_arg_combine_definition(family[len("arg_") : -len("_combine")], value, index), (
                f"cpf_arg_pair_of_{value}_{index}",
            )
        return c_arg_step_definition(family[len("arg_") : -len("_step")], value, index), (
            f"cpf_arg_pair_of_{value}_{index}",
        )
    reduction = C_REDUCE_INSTANCE_NAME.fullmatch(name)
    if reduction is not None:
        sequential, operation, source, accumulator = reduction.groups()
        return c_reduce_definition(name, operation, sequential is None, source, accumulator), ()
    match = C_INSTANCE_NAME.fullmatch(name)
    if match is None:
        return None
    family, kind, operation = match.group(1), match.group(2), match.group(3)
    types = tuple(match.group(4)[1:].split("_"))
    if kind is not None and len(types) == 3:
        return c_scan_definition(name, kind, operation, types[0], types[1], types[2]), ()
    if family == "detect_collision" and len(types) == 2:
        return c_detect_collision_definition(name, types[0], types[1]), ()
    if family == "detect_collision_sized" and len(types) == 1:
        tagged = "cpf_detect_collision_%s_int64" % types[0]
        return c_detect_collision_sized_definition(name, types[0], tagged), (tagged,)
    if family == "sort" and len(types) == 1:
        return c_sort_definition(name, types[0]), ()
    return None


def c_or_tile_instance(name: str) -> tuple[str, tuple[str, ...]] | None:
    """A typed C instantiation, or a tile function (:mod:`dace.tile_functions`), whose text is C already."""
    return c_instance(name) or tile_functions.definition(name)


def no_instances(name: str) -> tuple[str, tuple[str, ...]] | None:
    """The C++ dialects name no typed instantiation: their templates instantiate themselves."""
    return None


#: The chunk sizer, identical to the C++ one: it takes one concrete type, so it is a plain function
#: in both dialects.
C_INLINE_DEFINITIONS["find_first_chunk"] = (
    "#ifdef _OPENMP\n"
    "#include <omp.h>\n"
    "#endif\n"
    "static inline long long find_first_chunk(long long span, bool parallel) {\n"
    "    const double chunk_scale = 8.0;\n"
    "    const long long chunks_per_thread = 4;\n"
    "    long long chunk = (long long)(chunk_scale * sqrt((double)span));\n"
    "    long long threads = 1;\n"
    "    long long ceiling;\n"
    "#ifdef _OPENMP\n"
    "    if (parallel) threads = (long long)omp_get_max_threads();\n"
    "#endif\n"
    "    ceiling = span / (chunks_per_thread * threads);\n"
    "    if (ceiling < 1) ceiling = 1;\n"
    "    if (chunk > ceiling) chunk = ceiling;\n"
    "    if (chunk < 1) chunk = 1;\n"
    "    return chunk;\n"
    "}"
)

#: Definitions each C definition calls, so a callee is defined before its caller.
C_DEFINITION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    helper: calls for helper, calls in C_TYPED_HELPER_DEPENDENCIES.items() if calls
}

#: What the C dialect refuses, and why. Empty: every construct CPF reaches has a C spelling.
C_UNSUPPORTED: dict[str, str] = {
    # A C unit refuses 16-bit floats outright (``refuse_unspellable_containers``), so it never calls these.
    **dict.fromkeys(
        ["cpf_lowp_base", *(f"cpf_lowp_{name}" for name in LOW_PRECISION_MATH)],
        "the C dialect has no 16-bit float; render the C++ dialect",
    ),
}

#: Helpers C answers with a REWRITE of the CALL SITE rather than a definition or a refusal -- a
#: third lane, and the only one, so the anti-rot tests can still insist every C++ helper is
#: accounted for. Each needs something only the call site knows: the scan's neutral elements and
#: the scan itself need the element types the call site names (:func:`c_scan_identities`,
#: :func:`c_scan`), the duplicate check needs its arrays' types (:func:`c_detect_collision`), and
#: the find-first takes a predicate, which in C++ is a lambda and in C becomes a function of its own
#: over the names the predicate reads (:func:`c_find_first`).
C_REWRITTEN_IN_NATIVE_CODE: frozenset[str] = frozenset(
    {"min_identity", "max_identity", "find_first_index", "detect_collision"}
    | {"scan_%s_%s" % (kind, operation) for kind in ("incl", "excl") for operation in SCAN_OPERATIONS}
    | {reduce_name(operation, parallel) for operation in REDUCE_OPERATIONS for parallel in (True, False)}
    | set(ARG_REDUCTION_CPP)
    | {"cpf_atomic_max", "cpf_atomic_min"}
)

#: Headers CPF's C output always includes. ``<stdbool.h>`` spells ``bool`` / ``true`` / ``false`` before C23, where
#: they became keywords and the header is a no-op: gcc 14 still defaults to ``gnu17``. ``<tgmath.h>`` is deliberately
#: absent -- see the section header above.
C_BASE_HEADERS: tuple[str, ...] = (
    "<stdint.h>",
    "<stdbool.h>",
    "<math.h>",
    "<limits.h>",
    "<float.h>",
    "<stdlib.h>",
    "<string.h>",
    "<assert.h>",
    "<complex.h>",
)

#: ``<complex.h>`` defines ``I``, and ``I`` is a plausible loop-index name in scientific code. The
#: macro is removed immediately after the include; complex literals are built with :data:`C_COMPLEX_BUILDERS`.
C_UNDEF_LINE: str = "#undef I  // <complex.h> defines I, which an SDFG may use as a container name"


class Tables(NamedTuple):
    """One dialect's complete lowering vocabulary.

    Bundled rather than looked up table by table so a new dialect cannot be half-added: every
    consumer takes the bundle, so a missing member is a construction error here instead of a
    ``dace::`` name reaching the output through the one table nobody remapped.
    """

    #: Runtime function -> the standard-library function with identical semantics.
    std_renames: dict[str, str]
    #: Runtime function -> ``(arity, format string over the printed arguments)``.
    rewrites: dict[str, tuple[int, str]]
    #: Function name -> the definition CPF emits for it.
    inline_definitions: dict[str, str]
    #: ``Max``/``Min`` -> the binary function they nest into.
    variadic_minmax: dict[str, str]
    #: Function name -> why this dialect cannot express it.
    unsupported: dict[str, str]
    #: ``dace::``-namespaced type -> its plain spelling.
    ctype_renames: dict[str, str]
    #: One alternation over every ``ctype_renames`` key, longest first so a prefix
    #: (``dace::uint``) cannot pre-empt a longer name (``dace::uint64``) -- the trailing ``\b``
    #: already prevents that, this only keeps the intent explicit. Lets :func:`rewrite_ctypes`
    #: rename every ctype in one pass instead of one pass per table entry.
    ctype_pattern: "re.Pattern"
    #: Headers every unit includes.
    base_headers: tuple[str, ...]
    #: Definition -> the definitions it calls.
    definition_dependencies: dict[str, tuple[str, ...]]
    #: Definition -> the headers its body needs, beyond ``base_headers``.
    definition_headers: dict[str, tuple[str, ...]]
    #: A call to one of ``inline_definitions`` in already-emitted code.
    helper_call: "re.Pattern"
    #: Every name this dialect knows, in any lane.
    known: set[str]
    #: Typed instantiation name -> ``(definition, dependencies)``, or ``None`` for any other name.
    instances: Callable[[str], tuple[str, tuple[str, ...]] | None]


#: An OPTIONAL explicit template-argument list between a helper's name and its call parentheses
#: (``get_scratch<ScanTag>(...)``, ``arg_pair_of<std::complex<double>, int64_t>(...)``). Deliberately narrow --
#: no parentheses, one level of nesting, one line -- so a comparison chain (``a < b && c > (d)``) cannot read as one.
EXPLICIT_TEMPLATE_ARGUMENTS = r"(?:<(?:[^<>();{}\n]|<[^<>();{}\n]*>)*>\s*)?"


def _tables(
    std_renames,
    rewrites,
    inline_definitions,
    minmax,
    unsupported,
    ctype_renames,
    base_headers,
    dependencies,
    definition_headers,
    typed: frozenset[str] = frozenset(),
    instances: Callable[[str], tuple[str, tuple[str, ...]] | None] = no_instances,
    instance_names: str = "",
) -> Tables:
    ctype_names = sorted(ctype_renames, key=len, reverse=True)
    helper_names = "|".join(
        sorted(inline_definitions, key=len, reverse=True) + ([instance_names] if instance_names else [])
    )
    return Tables(
        std_renames=std_renames,
        rewrites=rewrites,
        inline_definitions=inline_definitions,
        variadic_minmax=minmax,
        unsupported=unsupported,
        ctype_renames=ctype_renames,
        ctype_pattern=re.compile(r"(?:::)?\b(" + "|".join(re.escape(name) for name in ctype_names) + r")\b"),
        base_headers=base_headers,
        definition_dependencies=dependencies,
        definition_headers=definition_headers,
        helper_call=re.compile(r"(?<![\w:.])(" + helper_names + r")\s*" + EXPLICIT_TEMPLATE_ARGUMENTS + r"\("),
        known=(
            set(std_renames) | set(rewrites) | set(inline_definitions) | set(minmax) | set(unsupported) | set(typed)
        ),
        instances=instances,
    )


#: What a DEVICE unit adds to :data:`INLINE_DEFINITIONS`: the runtime functions a device library
#: expansion calls, written out the same way the host ones are. Device-only, so they are NOT in the
#: shared table -- every body here holds a kernel launch or ``__global__``, which a host compiler
#: cannot parse.
#:
#: They are reached the same way too. A ``FindFirst`` or ``Scan`` on a device graph expands to a
#: HOST tasklet plus a wrapper in the device unit, and the wrapper calls one of these; CPF's own
#: definition stands in for the runtime header the wrapper would otherwise include, exactly as
#: ``find_first_index`` and ``scan_incl_sum`` stand in for ``dace/scan.hpp`` on the host.
HIP_DEVICE_INLINE_DEFINITIONS: dict[str, str] = {
    "get_scratch": "//: The CUB scratch pool, as much of it as one entry point can hold: one buffer per tag,\n"
    "//: allocated on first use and grown, never shrunk, so a repeated call pays no allocation.\n"
    "//: The runtime pre-allocates and releases it from the entry points a repeated invocation\n"
    "//: shares; one self-contained call has neither, so the buffer owns itself and frees at\n"
    "//: static destruction. It is also not keyed by stream, as the runtime pool is: a CPF unit\n"
    "//: issues every launch on the one stream (see cpf_gpu_context), so its calls are ordered.\n"
    "struct ScanTag {};\n"
    "struct DetectFlagTag {};\n"
    "struct DetectOwnerTag {};\n"
    "struct SortTag {};\n"
    "struct cpf_scratch_block {\n"
    "    void *ptr = nullptr;\n"
    "    size_t capacity = 0;\n"
    "    ~cpf_scratch_block() { if (ptr != nullptr) { (void)gpuFree(ptr); } }\n"
    "};\n"
    "template <typename Tag>\n"
    "static inline void *get_scratch(size_t bytes, gpuStream_t stream, gpuError_t *status) {\n"
    "    static cpf_scratch_block block;\n"
    "    *status = gpuSuccess;\n"
    "    if (bytes <= block.capacity) return block.ptr;\n"
    "    if (block.ptr != nullptr) {\n"
    "        // In flight work may still be reading the old buffer.\n"
    "        *status = gpuStreamSynchronize(stream);\n"
    "        if (*status != gpuSuccess) return nullptr;\n"
    "        *status = gpuFree(block.ptr);\n"
    "        block.ptr = nullptr;\n"
    "        block.capacity = 0;\n"
    "        if (*status != gpuSuccess) return nullptr;\n"
    "    }\n"
    "    *status = gpuMalloc(&block.ptr, bytes);\n"
    "    if (*status != gpuSuccess) { block.ptr = nullptr; return nullptr; }\n"
    "    block.capacity = bytes;\n"
    "    return block.ptr;\n"
    "}",
    "find_first_index_device": "//: The device find-first: the smallest index in [begin, end) at which ``pred`` fires, or\n"
    "//: ``end``. The answer is an atomic min over the firing indices and is exact; the read of it\n"
    "//: at the top of each step races by design, and every value it can take is a real firing\n"
    "//: index, so a stale one costs pruning and never correctness.\n"
    "//:\n"
    "//: ``pred`` is a FUNCTOR and not a lambda so no caller needs --extended-lambda: the\n"
    "//: expansion appends the struct to the device code beside the wrapper that instantiates it.\n"
    "template <typename Pred>\n"
    "__global__ void find_first_kernel(long long begin, long long end, Pred pred,\n"
    "                                  unsigned long long *result) {\n"
    "    const long long stride = (long long)gridDim.x * (long long)blockDim.x;\n"
    "    long long i = begin + (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n"
    "    for (; i < end; i += stride) {\n"
    "        // Every index below this thread's next one is another thread's, so the first hit it\n"
    "        // finds walking upwards is the smallest it can contribute: it is done either way.\n"
    "        if (i >= (long long)*(volatile unsigned long long *)result) return;\n"
    "        if (pred(i)) { atomicMin(result, (unsigned long long)i); return; }\n"
    "    }\n"
    "}\n"
    "template <typename Pred>\n"
    "static inline gpuError_t find_first_index_device(long long begin, long long end, Pred pred, long long *out,\n"
    "                                                gpuStream_t stream) {\n"
    "    constexpr int block_threads = 256;\n"
    "    constexpr long long max_blocks = 1024;\n"
    "    *out = end;\n"
    "    if (begin >= end) return gpuSuccess;\n"
    "    gpuError_t status = gpuSuccess;\n"
    "    unsigned long long *result = (unsigned long long *)get_scratch<DetectFlagTag>(\n"
    "        sizeof(unsigned long long), stream, &status);\n"
    "    if (result == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    const unsigned long long sentinel = (unsigned long long)end;\n"
    "    status = gpuMemcpyAsync(result, &sentinel, sizeof(sentinel), gpuMemcpyHostToDevice, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    // ``sentinel`` is a local, so the copy has to be done before this frame goes away.\n"
    "    status = gpuStreamSynchronize(stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    long long blocks = (end - begin + block_threads - 1) / block_threads;\n"
    "    if (blocks > max_blocks) blocks = max_blocks;\n"
    "    find_first_kernel<<<(unsigned)blocks, block_threads, 0, stream>>>(begin, end, pred, result);\n"
    "    status = gpuGetLastError();\n"
    "    if (status != gpuSuccess) return status;\n"
    "    unsigned long long found = sentinel;\n"
    "    status = gpuMemcpyAsync(&found, result, sizeof(found), gpuMemcpyDeviceToHost, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    status = gpuStreamSynchronize(stream);\n"
    "    *out = (long long)found;\n"
    "    return status;\n"
    "}",
    "detect_collision_device": "//: The scatter guard's duplicate check ON THE DEVICE, the device side of detect_collision: pass 1\n"
    "//: tags owner[idx[i]] = i, pass 2 ORs owner[idx[i]] != i, so a slot that two positions name reads\n"
    "//: back the other one's tag and trips the flag. Both passes are grid-stride kernels over the device\n"
    "//: index array; only the 0/1 flag crosses back to the host. Values outside [0, capacity) are skipped:\n"
    "//: the guarded scatter never writes those slots. The tag buffer is device scratch (get_scratch) and\n"
    "//: needs no initialization: pass 2 reads only slots pass 1 wrote.\n"
    "template <typename T>\n"
    "__global__ void detect_tag_kernel(const T *idx, long long n, long long *owner, long long capacity) {\n"
    "    const long long stride = (long long)gridDim.x * (long long)blockDim.x;\n"
    "    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; i < n; i += stride) {\n"
    "        const long long v = (long long)idx[i];\n"
    "        if (v >= 0 && v < capacity) owner[v] = i;\n"
    "    }\n"
    "}\n"
    "template <typename T>\n"
    "__global__ void detect_verify_kernel(const T *idx, long long n, const long long *owner, long long capacity,\n"
    "                                     unsigned long long *flag) {\n"
    "    const long long stride = (long long)gridDim.x * (long long)blockDim.x;\n"
    "    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; i < n; i += stride) {\n"
    "        const long long v = (long long)idx[i];\n"
    "        // One write of the same value per tripping lane: no fold is needed for a flag.\n"
    "        if (v >= 0 && v < capacity && owner[v] != i) { *flag = 1ull; return; }\n"
    "    }\n"
    "}\n"
    "//: The check without a capacity: open addressing over 2n slots (at least), keyed by value. A slot holding\n"
    "//: ~0 is empty; a non-negative value can never be ~0. Memory follows n, not the values, so an index array\n"
    "//: of huge or uninitialized values cannot make the check allocate past the device.\n"
    "template <typename T>\n"
    "__global__ void detect_hash_kernel(const T *idx, long long n, unsigned long long *keys, unsigned long long mask,\n"
    "                                   unsigned long long *flag) {\n"
    "    const long long stride = (long long)gridDim.x * (long long)blockDim.x;\n"
    "    for (long long i = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x; i < n; i += stride) {\n"
    "        const long long v = (long long)idx[i];\n"
    "        if (v < 0) continue;\n"
    "        const unsigned long long key = (unsigned long long)v;\n"
    "        unsigned long long slot = (key * 0x9E3779B97F4A7C15ull) & mask;\n"
    "        while (true) {\n"
    "            const unsigned long long seen = atomicCAS(&keys[slot], ~0ull, key);\n"
    "            if (seen == ~0ull) break;\n"
    "            if (seen == key) { *flag = 1ull; break; }\n"
    "            slot = (slot + 1) & mask;\n"
    "        }\n"
    "    }\n"
    "}\n"
    "static inline long long cpf_detect_blocks(long long n) {\n"
    "    const long long blocks = (n + 255) / 256;\n"
    "    return blocks > 4096 ? 4096 : blocks;\n"
    "}\n"
    "//: 1 in *out when idx[0:n] holds a value twice within [0, capacity), else 0. Synchronizes stream.\n"
    "template <typename T>\n"
    "static inline gpuError_t detect_collision_device(const T *idx, long long n, long long capacity, long long *out,\n"
    "                                                gpuStream_t stream) {\n"
    "    *out = 0;\n"
    "    if (n <= 0 || capacity <= 0) return gpuSuccess;\n"
    "    gpuError_t status = gpuSuccess;\n"
    "    long long *owner = (long long *)get_scratch<DetectOwnerTag>((size_t)capacity * sizeof(long long), stream, &status);\n"
    "    if (owner == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    unsigned long long *flag = (unsigned long long *)get_scratch<DetectFlagTag>(sizeof(unsigned long long), stream,\n"
    "                                                                               &status);\n"
    "    if (flag == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    const long long blocks = cpf_detect_blocks(n);\n"
    "    status = gpuMemsetAsync(flag, 0, sizeof(unsigned long long), stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    detect_tag_kernel<<<(unsigned)blocks, 256, 0, stream>>>(idx, n, owner, capacity);\n"
    "    detect_verify_kernel<<<(unsigned)blocks, 256, 0, stream>>>(idx, n, owner, capacity, flag);\n"
    "    status = gpuGetLastError();\n"
    "    if (status != gpuSuccess) return status;\n"
    "    unsigned long long host_flag = 0;\n"
    "    status = gpuMemcpyAsync(&host_flag, flag, sizeof(host_flag), gpuMemcpyDeviceToHost, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    status = gpuStreamSynchronize(stream);\n"
    "    *out = (long long)(host_flag != 0);\n"
    "    return status;\n"
    "}\n"
    "//: The same check where the scattered array's extent is unknown: a hash table of the values, one pass.\n"
    "template <typename T>\n"
    "static inline gpuError_t detect_collision_device(const T *idx, long long n, long long *out, gpuStream_t stream) {\n"
    "    *out = 0;\n"
    "    if (n <= 0) return gpuSuccess;\n"
    "    unsigned long long slots = 1;\n"
    "    while (slots < 2ull * (unsigned long long)n) slots <<= 1;\n"
    "    gpuError_t status = gpuSuccess;\n"
    "    unsigned long long *keys = (unsigned long long *)get_scratch<DetectOwnerTag>(slots * sizeof(unsigned long long),\n"
    "                                                                             stream, &status);\n"
    "    if (keys == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    unsigned long long *flag = (unsigned long long *)get_scratch<DetectFlagTag>(sizeof(unsigned long long), stream,\n"
    "                                                                               &status);\n"
    "    if (flag == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    status = gpuMemsetAsync(keys, 0xFF, slots * sizeof(unsigned long long), stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    status = gpuMemsetAsync(flag, 0, sizeof(unsigned long long), stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    detect_hash_kernel<<<(unsigned)cpf_detect_blocks(n), 256, 0, stream>>>(idx, n, keys, slots - 1, flag);\n"
    "    status = gpuGetLastError();\n"
    "    if (status != gpuSuccess) return status;\n"
    "    unsigned long long host_flag = 0;\n"
    "    status = gpuMemcpyAsync(&host_flag, flag, sizeof(host_flag), gpuMemcpyDeviceToHost, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    status = gpuStreamSynchronize(stream);\n"
    "    *out = (long long)(host_flag != 0);\n"
    "    return status;\n"
    "}",
    "inclusive_affine": "//: The linear recurrence out[k] = c[k]*out[k-s] + d[k], class r entered at out[r-s] = seed[r].\n"
    "//: The carry is the affine MAP x -> a*x + b rather than a value, and map composition is\n"
    "//: associative, so a plain prefix scan over the maps computes the recurrence.\n"
    "//:\n"
    "//: The seed is folded into each residue class's first element rather than handed to cub as an\n"
    "//: init value, and that is numerically load-bearing: the element comes out as the CONSTANT map\n"
    "//: {0, c*seed + d}, so every prefix including it carries a == 0 and the coefficient product never\n"
    "//: spans more than one composed segment -- which is what keeps it off the overflow the closed form\n"
    "//: hits. The maps are laid out class by class (k = r + j*s at r*len + j), so the same constant\n"
    "//: first map restarts the one scan at every class boundary and it computes all s chains.\n"
    "template <typename E>\n"
    "struct cpf_affine_map { E a; E b; };\n"
    "template <typename E>\n"
    "struct cpf_affine_compose {\n"
    "    __device__ __forceinline__ cpf_affine_map<E> operator()(const cpf_affine_map<E> &x,\n"
    "                                                            const cpf_affine_map<E> &y) const {\n"
    "        return cpf_affine_map<E>{y.a * x.a, y.a * x.b + y.b};\n"
    "    }\n"
    "};\n"
    "template <typename E, typename C, typename D, typename S>\n"
    "__global__ void cpf_affine_pack_kernel(const C *__restrict__ c, const D *__restrict__ d,\n"
    "                                       cpf_affine_map<E> *__restrict__ m, const S *__restrict__ seed_ptr,\n"
    "                                       E seed_val, long long n, long long stride, long long len) {\n"
    "    const long long slot = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n"
    "    if (slot >= stride * len) return;\n"
    "    const long long r = slot / len, j = slot % len, k = r + j * stride;\n"
    "    if (k >= n) {\n"
    "        m[slot] = cpf_affine_map<E>{(E)1, (E)0};  // the identity pads a short class\n"
    "        return;\n"
    "    }\n"
    "    const E ck = (E)c[k];\n"
    "    const E dk = (E)d[k];\n"
    "    // A device-resident seed arrives as a pointer, one per class; a host-readable one by value.\n"
    "    if (j == 0) {\n"
    "        const E s = (seed_ptr != nullptr) ? (E)seed_ptr[r] : seed_val;\n"
    "        m[slot] = cpf_affine_map<E>{(E)0, ck * s + dk};\n"
    "    } else {\n"
    "        m[slot] = cpf_affine_map<E>{ck, dk};\n"
    "    }\n"
    "}\n"
    "template <typename E>\n"
    "__global__ void cpf_affine_unpack_kernel(const cpf_affine_map<E> *__restrict__ m, E *__restrict__ out,\n"
    "                                         long long n, long long stride, long long len) {\n"
    "    const long long slot = (long long)blockIdx.x * (long long)blockDim.x + (long long)threadIdx.x;\n"
    "    if (slot >= stride * len) return;\n"
    "    const long long k = slot / len + (slot % len) * stride;\n"
    "    if (k < n) out[k] = m[slot].b;  // every composed prefix is constant, so b IS the value\n"
    "}\n"
    "template <typename E, typename C, typename D, typename S>\n"
    "static inline gpuError_t inclusive_affine(const C *coef, const D *delta, const S *seed_ptr, E seed_val,\n"
    "                                          E *out, long long n, long long stride, gpuStream_t stream) {\n"
    "    using M = cpf_affine_map<E>;\n"
    "    constexpr int block_threads = 256;\n"
    "    if (n <= 0) return gpuSuccess;\n"
    "    if (stride > n) stride = n;\n"
    "    const long long len = (n + stride - 1) / stride;\n"
    "    const long long slots = stride * len;\n"
    "    cpf_affine_compose<E> op;\n"
    "    size_t cub_bytes = 0;\n"
    "    gpuError_t status = gpucub::DeviceScan::InclusiveScan(nullptr, cub_bytes, (M *)nullptr, (M *)nullptr,\n"
    "                                                         op, slots, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    // 256-byte alignment for the workspace that follows: cub assumes an allocation at least as\n"
    "    // aligned as gpuMalloc gives, and the maps sit in front of it in the one block.\n"
    "    const size_t map_bytes = (((size_t)slots * sizeof(M)) + 255u) & ~(size_t)255u;\n"
    "    void *scratch = get_scratch<ScanTag>(map_bytes + cub_bytes, stream, &status);\n"
    "    if (scratch == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    M *maps = (M *)scratch;\n"
    "    void *workspace = (char *)scratch + map_bytes;\n"
    "    const unsigned blocks = (unsigned)((slots + block_threads - 1) / block_threads);\n"
    "    cpf_affine_pack_kernel<E, C, D, S><<<blocks, block_threads, 0, stream>>>(coef, delta, maps, seed_ptr,\n"
    "                                                                            seed_val, n, stride, len);\n"
    "    status = gpuGetLastError();\n"
    "    if (status != gpuSuccess) return status;\n"
    "    status = gpucub::DeviceScan::InclusiveScan(workspace, cub_bytes, maps, maps, op, slots, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    cpf_affine_unpack_kernel<E><<<blocks, block_threads, 0, stream>>>(maps, out, n, stride, len);\n"
    "    return gpuGetLastError();\n"
    "}",
    "segments_inclusive": "//: Independent inclusive scans over one buffer, the device side of a strided or segmented Scan: scan k\n"
    "//: reads in[k * pitch + g * step] for g below its length -- the residue classes mod s are\n"
    "//: {n, s, 1, s, n}, the rows of a segmented scan {n, rows, n / rows, 1, n / rows}. Many interleaved\n"
    "//: classes take one thread each; otherwise every scan is cut into chunks of 4096 elements and scanned\n"
    "//: in three phases: every (scan, chunk) block folds its chunk, one block per scan scans those totals,\n"
    "//: and every block rescans its chunk entered with the total before it. Block b is scan b % count of\n"
    "//: chunk b / count.\n"
    "struct cpf_segments {\n"
    "    long n, count, pitch, step, cap;\n"
    "    __host__ __device__ long length(long k) const {\n"
    "        const long first = k * pitch;\n"
    "        if (first >= n) return 0;\n"
    "        const long m = (n - first + step - 1) / step;\n"
    "        return m < cap ? m : cap;\n"
    "    }\n"
    "};\n"
    "template <typename T, typename Op>\n"
    "__global__ void cpf_per_class_scan_kernel(const T *__restrict__ in, T *__restrict__ out, long n, long s, Op op) {\n"
    "    const long k = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;\n"
    "    if (k >= s || k >= n) return;\n"
    "    T acc = in[k];\n"
    "    out[k] = acc;\n"
    "    for (long j = k + s; j < n; j += s) { acc = op(acc, in[j]); out[j] = acc; }\n"
    "}\n"
    "template <typename T, typename Op>\n"
    "struct cpf_scan_carry {\n"
    "    T running;\n"
    "    Op op;\n"
    "    __device__ T operator()(T block_aggregate) { const T seed = running; running = op(running, block_aggregate); return seed; }\n"
    "};\n"
    "//: m elements spaced step apart, scanned by the whole block with seed folded in front.\n"
    "template <typename T, typename Op, int BLOCK>\n"
    "__device__ void cpf_block_scan(const T *__restrict__ in, T *__restrict__ out, long m, long step, Op op, T identity,\n"
    "                               T seed) {\n"
    "    typedef gpucub::BlockScan<T, BLOCK> BlockScanT;\n"
    "    __shared__ typename BlockScanT::TempStorage storage;\n"
    "    cpf_scan_carry<T, Op> carry{seed, op};\n"
    "    for (long base = 0; base < m; base += BLOCK) {\n"
    "        const long g = base + (long)threadIdx.x;\n"
    "        T value = (g < m) ? in[g * step] : identity;  // every lane reaches the barrier in the scan\n"
    "        BlockScanT(storage).InclusiveScan(value, value, op, carry);\n"
    "        if (g < m) out[g * step] = value;\n"
    "        __syncthreads();\n"
    "    }\n"
    "}\n"
    "//: Where this block's chunk starts in the buffer, and how many elements of its scan it holds.\n"
    "__device__ inline void cpf_chunk_of_block(const cpf_segments &seg, long chunk, long *offset, long *count) {\n"
    "    const long k = (long)blockIdx.x % seg.count;\n"
    "    const long begin = ((long)blockIdx.x / seg.count) * chunk;\n"
    "    const long m = seg.length(k);\n"
    "    *offset = k * seg.pitch + begin * seg.step;\n"
    "    *count = (m > begin) ? ((m - begin < chunk) ? (m - begin) : chunk) : 0;\n"
    "}\n"
    "template <typename T, typename Op, int BLOCK>\n"
    "__global__ void cpf_chunk_totals_kernel(const T *__restrict__ in, T *__restrict__ totals, cpf_segments seg, long chunk,\n"
    "                                        Op op, T identity) {\n"
    "    long offset, count;\n"
    "    cpf_chunk_of_block(seg, chunk, &offset, &count);\n"
    "    T acc = identity;\n"
    "    for (long g = (long)threadIdx.x; g < count; g += BLOCK) acc = op(acc, in[offset + g * seg.step]);\n"
    "    typedef gpucub::BlockReduce<T, BLOCK> BlockReduceT;\n"
    "    __shared__ typename BlockReduceT::TempStorage storage;\n"
    "    const T total = BlockReduceT(storage).Reduce(acc, op);\n"
    "    if (threadIdx.x == 0) totals[blockIdx.x] = total;\n"
    "}\n"
    "template <typename T, typename Op, int BLOCK>\n"
    "__global__ void cpf_segment_scan_kernel(const T *__restrict__ in, T *__restrict__ out, cpf_segments seg, Op op,\n"
    "                                        T identity) {\n"
    "    const long k = (long)blockIdx.x;\n"
    "    cpf_block_scan<T, Op, BLOCK>(in + k * seg.pitch, out + k * seg.pitch, seg.length(k), seg.step, op, identity,\n"
    "                                 identity);\n"
    "}\n"
    "template <typename T, typename Op, int BLOCK>\n"
    "__global__ void cpf_chunk_scan_kernel(const T *__restrict__ in, T *__restrict__ out, const T *__restrict__ totals,\n"
    "                                      cpf_segments seg, long chunk, Op op, T identity) {\n"
    "    long offset, count;\n"
    "    cpf_chunk_of_block(seg, chunk, &offset, &count);\n"
    "    const T seed = ((long)blockIdx.x >= seg.count) ? totals[(long)blockIdx.x - seg.count] : identity;\n"
    "    cpf_block_scan<T, Op, BLOCK>(in + offset, out + offset, count, seg.step, op, identity, seed);\n"
    "}\n"
    "template <typename T, typename Op>\n"
    "static inline gpuError_t segments_inclusive(const T *in, T *out, cpf_segments seg, Op op, T identity,\n"
    "                                            gpuStream_t stream) {\n"
    "    constexpr int block = 256;\n"
    "    constexpr long chunk = 16L * block;\n"
    "    if (seg.count <= 0 || seg.n <= 0) return gpuSuccess;\n"
    "    if (seg.pitch == 1 && seg.count >= 4096) {\n"
    "        cpf_per_class_scan_kernel<T, Op><<<(unsigned)((seg.count + block - 1) / block), block, 0, stream>>>(\n"
    "            in, out, seg.n, seg.step, op);\n"
    "        return gpuGetLastError();\n"
    "    }\n"
    "    const long chunks = (seg.length(0) + chunk - 1) / chunk;\n"
    "    if (chunks <= 1) {\n"
    "        cpf_segment_scan_kernel<T, Op, block><<<(unsigned)seg.count, block, 0, stream>>>(in, out, seg, op, identity);\n"
    "        return gpuGetLastError();\n"
    "    }\n"
    "    gpuError_t status = gpuSuccess;\n"
    "    T *totals = (T *)get_scratch<ScanTag>((size_t)(chunks * seg.count) * sizeof(T), stream, &status);\n"
    "    if (totals == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    const unsigned blocks = (unsigned)(chunks * seg.count);\n"
    "    cpf_chunk_totals_kernel<T, Op, block><<<blocks, block, 0, stream>>>(in, totals, seg, chunk, op, identity);\n"
    "    const cpf_segments of_totals{chunks * seg.count, seg.count, 1, seg.count, chunks};\n"
    "    cpf_segment_scan_kernel<T, Op, block><<<(unsigned)seg.count, block, 0, stream>>>(totals, totals, of_totals, op,\n"
    "                                                                                    identity);\n"
    "    cpf_chunk_scan_kernel<T, Op, block><<<blocks, block, 0, stream>>>(in, out, totals, seg, chunk, op, identity);\n"
    "    return gpuGetLastError();\n"
    "}",
    "strided_inclusive_sum": "template <typename T>\n"
    "static inline gpuError_t strided_inclusive_sum(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, 1, s, n}, cpf_cub_plus(), T(0), stream);\n"
    "}",
    "segmented_inclusive_sum": "template <typename T>\n"
    "static inline gpuError_t segmented_inclusive_sum(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, n / s, 1, n / s}, cpf_cub_plus(), T(0), stream);\n"
    "}",
    "strided_inclusive_product": "template <typename T>\n"
    "static inline gpuError_t strided_inclusive_product(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, 1, s, n}, cpf_cub_multiplies(), T(1), stream);\n"
    "}",
    "segmented_inclusive_product": "template <typename T>\n"
    "static inline gpuError_t segmented_inclusive_product(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, n / s, 1, n / s}, cpf_cub_multiplies(), T(1), stream);\n"
    "}",
    "strided_inclusive_min": "template <typename T>\n"
    "static inline gpuError_t strided_inclusive_min(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, 1, s, n}, cpf_cub_minimum(), min_identity<T>(), stream);\n"
    "}",
    "segmented_inclusive_min": "template <typename T>\n"
    "static inline gpuError_t segmented_inclusive_min(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, n / s, 1, n / s}, cpf_cub_minimum(), min_identity<T>(), stream);\n"
    "}",
    "strided_inclusive_max": "template <typename T>\n"
    "static inline gpuError_t strided_inclusive_max(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, 1, s, n}, cpf_cub_maximum(), max_identity<T>(), stream);\n"
    "}",
    "segmented_inclusive_max": "template <typename T>\n"
    "static inline gpuError_t segmented_inclusive_max(const T *in, T *out, long n, long s, gpuStream_t stream) {\n"
    "    return segments_inclusive(in, out, cpf_segments{n, s, n / s, 1, n / s}, cpf_cub_maximum(), max_identity<T>(), stream);\n"
    "}",
    # ``ArgReduce``'s CUDA expansion calls ``dace::cub::arg_reduce`` and, for a strided or transformed read,
    # ``dace::cub::gather_iterator``. Without these the node took its host OpenMP lowering, which read the
    # device operand on the host (SIGSEGV: argmax_with_index, tsvc_2_s318, tsvc_2_s3110 on CUDA).
    "gather_iterator": "//: A strided, transformed read ``xf(base[j * stride])``, indexed as an array is.\n"
    "struct IdentityXf {\n"
    "    template <typename T> __host__ __device__ T operator()(const T &v) const { return v; }\n"
    "};\n"
    "struct AbsXf {\n"
    "    template <typename T> __host__ __device__ T operator()(const T &v) const { return v < T(0) ? -v : v; }\n"
    "};\n"
    "template <typename T, typename Xf>\n"
    "struct cpf_strided_read {\n"
    "    const T *base;\n"
    "    long long stride;\n"
    "    __host__ __device__ T operator[](long long j) const { return Xf()(base[j * stride]); }\n"
    "};\n"
    "template <typename Xf, typename T>\n"
    "static inline cpf_strided_read<T, Xf> gather_iterator(const T *base, long long stride) {\n"
    "    return cpf_strided_read<T, Xf>{base, stride};\n"
    "}",
    "arg_reduce": "//: The device arg-reduction, answered on the host: each lane folds its grid-strided elements into a\n"
    "//: (value, index) pair, each block folds its lanes, and one block folds the blocks' pairs. A lane that has\n"
    "//: taken no element holds index -1 and loses every fold; a NaN past element 0 starts no lane, as the\n"
    "//: sequential loop never takes one. ``arg_combine`` breaks ties by index, so the answer is the loop's.\n"
    "struct ReduceTag {};\n"
    "template <bool Max, bool Last>\n"
    "struct ArgPairOp {\n"
    "    static constexpr bool max = Max, last = Last;\n"
    "};\n"
    "using ArgMaxOp = ArgPairOp<true, false>;\n"
    "using ArgMinOp = ArgPairOp<false, false>;\n"
    "template <bool Max, bool Last, typename P>\n"
    "__device__ inline P arg_lane_fold(const P &l, const P &r) {\n"
    "    return l.__ar_i < 0 ? r : (r.__ar_i < 0 ? l : arg_combine<Max, Last>(l, r));\n"
    "}\n"
    "template <bool Max, bool Last, typename P>\n"
    "__device__ inline void arg_block_fold(P mine, P *out) {\n"
    "    __shared__ P lanes[256];\n"
    "    lanes[threadIdx.x] = mine;\n"
    "    __syncthreads();\n"
    "    for (unsigned s = 128; s > 0; s >>= 1) {\n"
    "        if (threadIdx.x < s) lanes[threadIdx.x] = arg_lane_fold<Max, Last>(lanes[threadIdx.x], lanes[threadIdx.x + s]);\n"
    "        __syncthreads();\n"
    "    }\n"
    "    if (threadIdx.x == 0) *out = lanes[0];\n"
    "}\n"
    "template <bool Max, bool Last, typename InIt, typename T>\n"
    "__global__ void __launch_bounds__(256) arg_reduce_kernel(InIt in, long long items, arg_pair<T, long long> *partial) {\n"
    "    arg_pair<T, long long> mine{T(), -1};\n"
    "    for (long long j = (long long)blockIdx.x * 256 + threadIdx.x; j < items; j += (long long)gridDim.x * 256) {\n"
    "        const T x = in[j];\n"
    "        if (j == 0 || x == x) mine = mine.__ar_i < 0 ? arg_pair<T, long long>{x, j}\n"
    "                                                   : arg_combine<Max, Last>(mine, arg_pair<T, long long>{x, j});\n"
    "    }\n"
    "    arg_block_fold<Max, Last>(mine, &partial[blockIdx.x]);\n"
    "}\n"
    "template <bool Max, bool Last, typename T>\n"
    "__global__ void __launch_bounds__(256) arg_reduce_blocks(arg_pair<T, long long> *partial, long long blocks) {\n"
    "    arg_pair<T, long long> mine{T(), -1};\n"
    "    for (long long b = threadIdx.x; b < blocks; b += 256) mine = arg_lane_fold<Max, Last>(mine, partial[b]);\n"
    "    arg_block_fold<Max, Last>(mine, &partial[0]);\n"
    "}\n"
    "template <typename Op, typename InIt>\n"
    "static inline gpuError_t arg_reduce(InIt in, std::decay_t<decltype(in[0])> *val_out, long long *idx_out,\n"
    "                                    long long items, gpuStream_t stream) {\n"
    "    using T = std::decay_t<decltype(in[0])>;\n"
    "    using P = arg_pair<T, long long>;\n"
    "    long long blocks = (items + 255) / 256;\n"
    "    if (blocks > 1024) blocks = 1024;\n"
    "    gpuError_t status = gpuSuccess;\n"
    "    P *partial = (P *)get_scratch<ReduceTag>((size_t)blocks * sizeof(P), stream, &status);\n"
    "    if (partial == nullptr) return status != gpuSuccess ? status : gpuErrorMemoryAllocation;\n"
    "    arg_reduce_kernel<Op::max, Op::last><<<(unsigned)blocks, 256, 0, stream>>>(in, items, partial);\n"
    "    arg_reduce_blocks<Op::max, Op::last><<<1, 256, 0, stream>>>(partial, blocks);\n"
    "    status = gpuGetLastError();\n"
    "    if (status != gpuSuccess) return status;\n"
    "    P host;\n"
    "    status = gpuMemcpyAsync(&host, partial, sizeof(P), gpuMemcpyDeviceToHost, stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    status = gpuStreamSynchronize(stream);\n"
    "    if (status != gpuSuccess) return status;\n"
    "    if (val_out != nullptr) *val_out = host.__ar_v;\n"
    "    *idx_out = host.__ar_i;\n"
    "    return gpuSuccess;\n"
    "}",
}

#: The head of a host helper function in :data:`INLINE_DEFINITIONS`: ``static`` before ``inline`` or ``constexpr inline``.
HOST_FUNCTION_HEAD = re.compile(r"^static (?=(?:constexpr )?inline )", re.MULTILINE)


def host_device(definition: str) -> str:
    """``definition`` with every helper function ``__host__ __device__``: a kernel of the device unit calls
    ``cpf_min`` and its other helpers too, and the judge's nvcc has no ``--expt-relaxed-constexpr`` to admit a
    ``constexpr`` host function there."""
    return HOST_FUNCTION_HEAD.sub("static __host__ __device__ ", definition)


HIP_INLINE_DEFINITIONS: dict[str, str] = {
    **{name: host_device(definition) for name, definition in INLINE_DEFINITIONS.items()},
    **HIP_DEVICE_INLINE_DEFINITIONS,
}

HIP_DEFINITION_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    **DEFINITION_DEPENDENCIES,
    "find_first_index_device": ("get_scratch",),
    "arg_reduce": ("get_scratch", "arg_pair"),
    "detect_collision_device": ("get_scratch",),
    "inclusive_affine": ("get_scratch",),
    "segments_inclusive": ("get_scratch",),
    "strided_inclusive_sum": ("segments_inclusive",),
    "segmented_inclusive_sum": ("segments_inclusive",),
    "strided_inclusive_product": ("segments_inclusive",),
    "segmented_inclusive_product": ("segments_inclusive",),
    "strided_inclusive_min": ("segments_inclusive", "min_identity"),
    "segmented_inclusive_min": ("segments_inclusive", "min_identity"),
    "strided_inclusive_max": ("segments_inclusive", "max_identity"),
    "segmented_inclusive_max": ("segments_inclusive", "max_identity"),
}

#: The CUB scratch pool's tag types, which reach the text as template ARGUMENTS rather than as
#: calls. Renamed here, with the type table, because that is what runs over the whole unit -- a
#: name table entry would need a call to fire on.
HIP_CTYPE_RENAMES: dict[str, str] = {
    **CTYPE_RENAMES,
    "dace::cub::ScanTag": "ScanTag",
    "dace::cub::DetectFlagTag": "DetectFlagTag",
    **{f"dace::cub::{name}": name for name in ("ArgMaxOp", "ArgMinOp", "ArgPairOp", "IdentityXf", "AbsXf")},
    "dace::cub::DetectOwnerTag": "DetectOwnerTag",
    "dace::cub::SortTag": "SortTag",
    # The toolkit's own bfloat16, which ``dace::bfloat16`` IS in a device build (``dace/types.h``): two bytes, the
    # bit layout of HB's ``__npb_bf16`` (``uint16_t``) storage, so a pointer to either crosses the C ABI alike.
    "dace::bfloat16": "__hip_bfloat16",
}

#: The CUDA unit's spelling of the same types: :data:`HIP_CTYPE_RENAMES` with NVIDIA's bfloat16.
CUDA_CTYPE_RENAMES: dict[str, str] = {**HIP_CTYPE_RENAMES, "dace::bfloat16": "__nv_bfloat16"}

#: Environments a DEVICE rendering supplies for itself, so that
#: ``framecode.generate_standalone_footer`` does not refuse them. Everything else with code to run
#: or a library to link is still refused: a single entry point has nowhere to run an initializer.
#:
#: ``CUDA`` declares the toolkit as a CMake package. A device dialect is BUILT by that toolkit's
#: compiler, so the package is there by construction, and there is nothing to find and nothing to
#: link. ``ScanScratch``, ``DetectScratch`` and ``SortScratch`` pre-allocate and release the CUB scratch pool;
#: :data:`HIP_DEVICE_INLINE_DEFINITIONS`'s ``get_scratch`` allocates on first use and frees at
#: static destruction, so both halves of that handshake are inside the unit.
DEVICE_PROVIDED_ENVIRONMENTS: frozenset[str] = frozenset({"CUDA", "ScanScratch", "DetectScratch", "SortScratch"})


class VendorLibrary(NamedTuple):
    """What a unit that calls one vendor library carries in place of the DaCe environment that wraps it.

    The DaCe environment reaches the library through DaCe's own headers (``dace_blas.h``, the handle
    wrappers), an initializer and its CMake discovery; a standalone unit has none of those. It includes the
    library's own system headers, defines at file scope what the DaCe wrapper would have provided, and names
    the library on the caller's link line (:attr:`dace.codegen.cpf.Rendering.link`). The library itself is
    the environment's to provide -- the judge image installs it like any other system library.
    """

    #: System headers, in include order, angle brackets included.
    headers: tuple[str, ...]
    #: The caller's link-line tokens, or a function of the finished unit's text that names them.
    link: "tuple[str, ...] | Callable[[str], tuple[str, ...]]"
    #: File-scope definitions the expansion's calls rely on, after the headers.
    definitions: str = ""
    #: Further headers only a C++ unit includes.
    cxx_headers: tuple[str, ...] = ()
    #: DaCe's own wrapper header for the library (the handle classes the expansions call through), relative to
    #: the ``dace`` package. CPF carries it inline, its namespace respelled (:func:`vendor_wrapper`).
    wrapper: str = ""
    #: The :data:`VENDOR_FAMILIES` whose calls say a unit uses this library.
    families: tuple[str, ...] = ()

    def link_tokens(self, code: str) -> tuple[str, ...]:
        """The link-line tokens of a unit whose text is ``code``."""
        return self.link(code) if callable(self.link) else self.link


#: OpenBLAS's own pthreads pool would compete with the unit's OpenMP team for the cores (DaCe's
#: ``DeferOpenBlasThreadsToOpenMP``: contour_integral's LU solves ran 12x slower that way), so a pthreads build is
#: kept on one thread when the unit is loaded. Weak, so a BLAS without the OpenBLAS extensions still links.
OPENBLAS_THREADS_DEFINITION = """\
#pragma weak openblas_get_parallel
#pragma weak openblas_set_num_threads
// A pthreads OpenBLAS runs its own pool beside the OpenMP team; keep it on one thread, as DaCe does.
__attribute__((constructor)) static void cpf_defer_openblas_threads(void) {
    if (openblas_get_parallel && openblas_set_num_threads && openblas_get_parallel() == 1) {
        openblas_set_num_threads(1);
    }
}
"""


def openblas_link(code: str) -> tuple[str, ...]:
    """OpenBLAS, after the separate LAPACKE of a distro that packages it apart (Debian, Ubuntu: liblapacke); an OpenBLAS
    that bundles LAPACKE has no such library."""
    # Deferred: the environment module imports dace, which imports this module.
    from dace.libraries.blas.environments.openblas import OpenBLAS

    search = tuple(flag for path in OpenBLAS.library_dirs() for flag in (f"-L{path}", f"-Wl,-rpath,{path}"))
    calls_lapacke = re.search(r"\bLAPACKE_\w+\s*\(", code) is not None
    libraries = (
        ("-llapacke", "-lopenblas") if calls_lapacke and ctypes.util.find_library("lapacke") else ("-lopenblas",)
    )
    return search + libraries


def fftw3_link(code: str) -> tuple[str, ...]:
    """The FFTW3 libraries ``code`` calls: ``libfftw3`` for double precision, ``libfftw3f`` for single. Never a
    threaded companion: CPF expands for the single-threaded planner
    (:func:`~dace.libraries.fft.environments.fftw3.single_threaded_planner`)."""
    return tuple(
        f"-l{library}"
        for prefix, library in (("fftw_", "fftw3"), ("fftwf_", "fftw3f"))
        if re.search(r"\b%s\w+\s*\(" % prefix, code)
    )


#: The FFT expansions' plan cache (``dace::fft::Plans``); a C unit plans per call instead (``fftw3_call``).
FFT_PLANS_WRAPPER = "libraries/fft/include/dace_fft.h"

#: DaCe environment name -> how a standalone unit calls that library. One family per architecture: OpenBLAS (CBLAS
#: and LAPACKE) and FFTW3 on the host, the toolkit's own libraries on CUDA and ROCm.
VENDOR_LIBRARIES: dict[str, VendorLibrary] = {
    "OpenBLAS": VendorLibrary(
        ("<cblas.h>", "<lapacke.h>"), openblas_link, OPENBLAS_THREADS_DEFINITION, families=("blas", "lapack")
    ),
    # The expansion refuses an unplannable layout by throwing.
    "FFTW3": VendorLibrary(
        ("<fftw3.h>",), fftw3_link, cxx_headers=("<stdexcept>",), wrapper=FFT_PLANS_WRAPPER, families=("fftw",)
    ),
    "cuBLAS": VendorLibrary((), ("-lcublas",), wrapper="libraries/blas/include/dace_cublas.h", families=("cublas",)),
    "cuSolverDn": VendorLibrary(
        (), ("-lcusolver",), wrapper="libraries/lapack/include/dace_cusolverdn.h", families=("cusolver",)
    ),
    "cuFFT": VendorLibrary(
        ("<cufft.h>", "<cufftXt.h>", "<stdexcept>", "<string>"),
        ("-lcufft",),
        wrapper=FFT_PLANS_WRAPPER,
        families=("cufft",),
    ),
    # hipBLAS and hipSOLVER take the cuBLAS and cuSOLVER wrappers, which spell their HIP counterparts under WITH_HIP.
    "hipBLAS": VendorLibrary((), ("-lhipblas",), wrapper="libraries/blas/include/dace_cublas.h", families=("hipblas",)),
    "hipSolverDn": VendorLibrary(
        (), ("-lhipsolver",), wrapper="libraries/lapack/include/dace_cusolverdn.h", families=("hipsolver",)
    ),
    "hipFFT": VendorLibrary(
        ("<hipfft/hipfft.h>", "<hipfft/hipfftXt.h>", "<stdexcept>", "<string>"),
        ("-lhipfft",),
        wrapper=FFT_PLANS_WRAPPER,
        families=("hipfft",),
    ),
}

#: Per device dialect, the vendor libraries of the other device toolkit.
OTHER_DEVICE_LIBRARIES: dict[Dialect, frozenset[str]] = {
    Dialect.STANDALONE_HIP: frozenset({"cuBLAS", "cuSolverDn", "cuFFT"}),
    Dialect.STANDALONE_CUDA: frozenset({"hipBLAS", "hipSolverDn", "hipFFT"}),
}

#: The namespace a unit's inlined vendor wrappers live in, in place of ``dace``.
VENDOR_NAMESPACE = "cpf_vendor"

#: A qualified name into one of the wrappers' namespaces, as an expansion or a state field spells it.
VENDOR_QUALIFIED: "re.Pattern" = re.compile(r"(?<![\w:])(?:::)?dace::(blas|lapack|fft)::")


def respell_vendor_namespaces(code: str) -> str:
    """``dace::blas::`` / ``dace::lapack::`` / ``dace::fft::`` as the inlined wrappers spell them (:data:`VENDOR_NAMESPACE`)."""
    return VENDOR_QUALIFIED.sub(lambda match: f"{VENDOR_NAMESPACE}::{match.group(1)}::", code)


#: The condition DaCe's GPU wrappers select their HIP spelling under.
HIP_BRANCH = re.compile(r"\s*#\s*if\s+defined\(__HIPCC__\)\s*\|\|\s*defined\(WITH_HIP\)\s*$")


def vendor_wrapper(path: str, seen: set[str], hip: bool = False) -> tuple[list[str], str]:
    """DaCe's wrapper header at ``path`` (relative to the ``dace`` package) as inline text: its system includes,
    and its body with ``namespace dace`` respelled to :data:`VENDOR_NAMESPACE`. A wrapper it includes by quoted
    path comes first, once per unit (``seen``); ``#pragma once`` and comments-only lines are dropped.

    The header's HIP/CUDA branch is resolved for the unit (``hip``), and a ``#define`` of that branch becomes
    what it stands for: an object-like rename is applied to the text, a configuration flag (``HIPBLAS_V2``, the
    default from ROCm 7 on) is dropped. A nested ``#if`` of the branch stays inline, its includes with it.

    :returns: ``(system headers, definitions)``.
    """
    import dace  # the package directory; dace is fully imported by the time a unit renders

    full = os.path.normpath(os.path.join(os.path.dirname(dace.__file__), path))
    if full in seen:
        return [], ""
    seen.add(full)
    headers: list[str] = []
    body: list[str] = []
    renames: dict[str, str] = {}
    # Inside the HIP/CUDA conditional: whether its current branch is the unit's, and how deep a nested #if is
    branch: bool | None = None
    depth = 0
    with open(full) as handle:
        for line in handle:
            if branch is None and HIP_BRANCH.match(line):
                branch = hip
                continue
            if branch is not None:
                directive = re.match(r"\s*#\s*(if|ifdef|ifndef|else|elif|endif)\b", line)
                if directive and depth == 0 and directive.group(1) == "else":
                    branch = not hip
                    continue
                if directive and depth == 0 and directive.group(1) == "endif":
                    branch = None
                    continue
                if directive and directive.group(1).startswith("if"):
                    depth += 1
                elif directive and directive.group(1) == "endif":
                    depth -= 1
                if not branch:
                    continue
                define = re.match(r"\s*#\s*define\s+(\w+)(?:\s+(\w+))?", line)
                if define:
                    if define.group(2):
                        renames[define.group(1)] = define.group(2)
                    continue
                if depth > 0:
                    body.append(line.rstrip("\n"))
                    continue
            include = re.match(r'\s*#\s*include\s*([<"])([^>"]+)[>"]', line)
            if include and include.group(1) == "<":
                headers.append(f"<{include.group(2)}>")
            elif include:
                nested = os.path.relpath(
                    os.path.normpath(os.path.join(os.path.dirname(full), include.group(2))),
                    os.path.dirname(dace.__file__),
                )
                nested_headers, nested_body = vendor_wrapper(nested, seen, hip)
                headers += nested_headers
                body.append(nested_body)
            elif not re.match(r"\s*#\s*pragma\s+once", line):
                body.append(line.rstrip("\n"))
    text = "\n".join(body).strip("\n")
    if renames:
        text = re.sub(r"\b(%s)\b" % "|".join(map(re.escape, renames)), lambda match: renames[match.group(1)], text)
    text = re.sub(r"\bnamespace\s+dace\b", f"namespace {VENDOR_NAMESPACE}", text)
    text = re.sub(r"//\s*namespace dace\b", f"// namespace {VENDOR_NAMESPACE}", text)
    return headers, respell_vendor_namespaces(text)


#: A state field holding a wrapper class that owns a library handle (``cublasCreate`` / ``cusolverDnCreate`` on first
#: ``Get()``), in the ``dace`` spelling of the environment's ``state_fields`` or the respelled one.
LIBRARY_HANDLE_FIELD: "re.Pattern" = re.compile(
    rf"^(?P<indent>[ \t]*)(?:::)?(?:dace|{VENDOR_NAMESPACE})::"
    r"(?P<type>(?:blas|lapack)::\w+Handle)[ \t]+(?P<name>\w+);[ \t]*$",
    re.MULTILINE,
)


def share_library_handles(code: str) -> str:
    """``code`` with each library handle field of its state struct (:data:`LIBRARY_HANDLE_FIELD`) an ``inline static``
    member, so the unit holds one handle across calls instead of one per call.

    A standalone entry builds its state struct on its own stack, so a handle held by value in it was created by every
    call (``cublasCreate`` on the first ``Get()``, a cuSOLVER workspace with it) and destroyed on return. As a static
    member the field keeps its name and its calls (``__state->cublas_handle.Get()`` reads a static member through the
    pointer), the aggregate initialization of the struct skips it, and the struct stays one definition in both units
    of the kernel, which share the one handle. It is created by the first call that uses it and destroyed at exit.
    Calls share it, so they run one at a time: the entry sets the handle's stream and pointer mode on every call and
    drains the device before it returns. Unchanged when the unit holds no handle.
    """
    return LIBRARY_HANDLE_FIELD.sub(
        lambda m: f"{m['indent']}inline static {VENDOR_NAMESPACE}::{m['type']} {m['name']};", code
    )


#: A vendor library's family name, and the call prefix that says a unit calls into it. Family names are the ones
#: a library catalog keys these libraries by (HPCAgent-Bench's ``envs/libraries.yaml``).
VENDOR_FAMILIES: tuple[tuple[str, "re.Pattern"], ...] = (
    ("blas", re.compile(r"\bcblas_\w+\s*\(")),
    ("lapack", re.compile(r"\bLAPACKE_\w+\s*\(")),
    ("fftw", re.compile(r"\bfftwf?_\w+\s*\(")),
    ("cublas", re.compile(r"\bcublas\w+\s*\(")),
    ("cusolver", re.compile(r"\bcusolverDn\w+\s*\(")),
    ("cufft", re.compile(r"\bcufft\w+\s*\(")),
    ("hipblas", re.compile(r"\bhipblas\w+\s*\(")),
    ("hipsolver", re.compile(r"\bhipsolverDn\w+\s*\(")),
    ("hipfft", re.compile(r"\bhipfft\w+\s*\(")),
)


#: A vendor library family's name, and the identifiers -- calls, types, enumerators -- that say a unit names it.
#: Wider than :data:`VENDOR_FAMILIES`: a device unit repeats the host's state struct, whose ``hipfftHandle``
#: field needs the header although that unit calls nothing of the library.
VENDOR_NAMES: dict[str, "re.Pattern"] = {
    "blas": re.compile(r"\b(?:cblas_|Cblas)\w+"),
    "lapack": re.compile(r"\b(?:LAPACKE_\w+|LAPACK_(?:ROW|COL)_MAJOR|lapack_(?:int|complex_\w+))\b"),
    "fftw": re.compile(r"\bfftwf?_\w+"),
    "cublas": re.compile(r"\b(?:cublas|CUBLAS_)\w+"),
    "cusolver": re.compile(r"\b(?:cusolver|CUSOLVER_)\w+"),
    "cufft": re.compile(r"\b(?:cufft|CUFFT_)\w+"),
    "hipblas": re.compile(r"\b(?:hipblas|HIPBLAS_)\w+"),
    "hipsolver": re.compile(r"\b(?:hipsolver|HIPSOLVER_)\w+"),
    "hipfft": re.compile(r"\b(?:hipfft|HIPFFT_)\w+"),
}


def vendor_families(code: str) -> tuple[str, ...]:
    """The vendor library families ``code`` calls into, in :data:`VENDOR_FAMILIES` order."""
    return tuple(family for family, call in VENDOR_FAMILIES if call.search(code))


def vendor_environments(environments) -> list:
    """The environment classes among ``environments`` that :data:`VENDOR_LIBRARIES` covers, in key order. A
    code object names its environments by their registration path
    (``dace.libraries.blas.environments.openblas.OpenBLAS``); the table, by class name."""
    from dace.library import get_environment  # dace.library imports the code generators, which import this

    found = {env.__name__: env for env in map(get_environment, environments) if env.__name__ in VENDOR_LIBRARIES}
    return [found[name] for name in sorted(found)]


def vendor_libraries(environments, code: str, dialect: Dialect | None = None) -> list[VendorLibrary]:
    """The :data:`VENDOR_LIBRARIES` entries of ``environments`` (code-object names) that ``code`` uses, in key
    order. An environment is listed for the library a node PREFERS, also when its expansion fell back to the pure
    one (an int64 GEMV has no CBLAS call); such a unit carries none of the library's headers or link tokens. A
    library is used when ``code`` names one of its :attr:`~VendorLibrary.families` (:data:`VENDOR_NAMES`), or its
    inlined wrapper's namespace (a state struct's handle field, which the device unit repeats without the call)."""
    named = {family for family, names in VENDOR_NAMES.items() if names.search(code)}
    wrapped = VENDOR_QUALIFIED.search(code) is not None or f"{VENDOR_NAMESPACE}::" in code
    # A HIP unit shares the CUDA libraries' wrappers and so their namespace, but never links the CUDA ones
    foreign = OTHER_DEVICE_LIBRARIES.get(_active_dialect if dialect is None else dialect, frozenset())
    return [
        library
        for name, library in (
            (env.__name__, VENDOR_LIBRARIES[env.__name__]) for env in vendor_environments(environments)
        )
        if name not in foreign and (named.intersection(library.families) or (library.wrapper and wrapped))
    ]


#: Environments EVERY standalone rendering supplies for itself. ``ParallelSTL`` links TBB for
#: libstdc++'s ``std::execution`` backend; CPF drops the policy (:data:`POLICY_SORT_CALL`), so the
#: unit needs no library.
PROVIDED_ENVIRONMENTS: frozenset[str] = frozenset({"ParallelSTL"})

#: ``std::sort`` under a ``std::execution`` policy, up to its first range argument. The policy is
#: dropped: C++ calls the plain standard ``std::sort``, C the OpenMP mergesort :func:`c_sort` types.
POLICY_SORT_CALL: "re.Pattern" = re.compile(r"(?:::)?std::sort\s*\(\s*(?:::)?std::execution::\w+\s*,\s*")


def device_definitions(dialect: Dialect) -> dict[str, str]:
    """:data:`HIP_INLINE_DEFINITIONS` with the device ones' ``gpu*`` calls spelled for ``dialect``'s backend."""
    return {name: device_spell_out(definition, dialect) for name, definition in HIP_INLINE_DEFINITIONS.items()}


#: Dialect -> its vocabulary. ``RUNTIME`` has none: a runtime rendering emits ``dace::`` names and
#: never consults these tables at all, so asking for its bundle is a bug worth a ``KeyError``.
TABLES: dict[Dialect, Tables] = {
    Dialect.STANDALONE: _tables(
        STD_RENAMES,
        REWRITES,
        INLINE_DEFINITIONS,
        VARIADIC_MINMAX,
        UNSUPPORTED,
        {**CTYPE_RENAMES, **HOST_LOW_PRECISION_CTYPES},
        BASE_HEADERS,
        DEFINITION_DEPENDENCIES,
        DEFINITION_HEADERS,
        instances=tile_functions.definition,
        instance_names=tile_functions.CALL_NAMES,
    ),
    Dialect.STANDALONE_C: _tables(
        C_STD_RENAMES,
        C_REWRITES,
        C_INLINE_DEFINITIONS,
        C_VARIADIC_MINMAX,
        C_UNSUPPORTED,
        C_CTYPE_RENAMES,
        C_BASE_HEADERS,
        C_DEFINITION_DEPENDENCIES,
        {},
        frozenset(C_TYPED_MATH) | frozenset(C_TYPED_HELPER_SPECS) | {"iround", "heaviside"},
        c_or_tile_instance,
        f"{C_INSTANCE_NAMES}|{tile_functions.CALL_NAMES}",
    ),
    # The HIP unit is C++, so it takes the C++ vocabulary and adds to it: the ROCm toolkit's own
    # headers, which ship with the compiler that builds the unit, and the device counterparts of
    # the runtime functions a DEVICE library expansion calls (:data:`HIP_INLINE_DEFINITIONS`).
    Dialect.STANDALONE_HIP: _tables(
        STD_RENAMES,
        REWRITES,
        device_definitions(Dialect.STANDALONE_HIP),
        VARIADIC_MINMAX,
        UNSUPPORTED,
        HIP_CTYPE_RENAMES,
        BASE_HEADERS + HIP_BASE_HEADERS,
        HIP_DEFINITION_DEPENDENCIES,
        DEFINITION_HEADERS,
    ),
    # The CUDA unit shares that vocabulary: its device definitions call ``gpu*`` names, spelled per
    # backend by :func:`device_definitions`, so only the toolkit headers differ.
    Dialect.STANDALONE_CUDA: _tables(
        STD_RENAMES,
        REWRITES,
        device_definitions(Dialect.STANDALONE_CUDA),
        VARIADIC_MINMAX,
        UNSUPPORTED,
        CUDA_CTYPE_RENAMES,
        BASE_HEADERS + CUDA_BASE_HEADERS,
        HIP_DEFINITION_DEPENDENCIES,
        DEFINITION_HEADERS,
    ),
}

#: Every runtime function the C dialect knows about, in any lane.
C_KNOWN: set[str] = TABLES[Dialect.STANDALONE_C].known


def tables_for(dialect: Dialect | None = None) -> Tables:
    """The lowering vocabulary of ``dialect``, or of the ambient dialect when none is given.

    :raises ValueError: for :attr:`Dialect.RUNTIME`, which has no vocabulary at all -- a runtime
                        rendering emits ``dace::`` names and never consults these tables, so
                        reaching here under it means the caller forgot a :func:`dialect_scope`.
    """
    resolved = dialect if dialect is not None else _active_dialect
    if resolved not in TABLES:
        raise ValueError(
            f"{resolved} has no CPF lowering tables; the standalone dialects are "
            f"{sorted(d.value for d in TABLES)}. Name one, or run inside a dialect_scope."
        )
    return TABLES[resolved]


#: Plain scalar type spellings a library expansion may write a FUNCTIONAL cast with
#: (``double(0)``, the ``Scan`` expansion's seed). C has no functional cast, so these become
#: ordinary cast expressions. Anchored on a closed list of type names rather than on "identifier
#: followed by ``(``", which would rewrite every call in the body.
C_CAST_TYPES: tuple[str, ...] = (
    # A BLAS expansion's complex ``alpha`` / ``beta`` (``dace::complex128(1)``), before or after the type rename.
    "dace::complex128",
    "dace::complex64",
    "double _Complex",
    "float _Complex",
    "long double",
    "unsigned long long",
    "unsigned long",
    "unsigned int",
    "unsigned char",
    "long long",
    "double",
    "float",
    "bool",
    "char",
    "short",
    "int",
    "long",
    "int8_t",
    "int16_t",
    "int32_t",
    "int64_t",
    "uint8_t",
    "uint16_t",
    "uint32_t",
    "uint64_t",
    "size_t",
)

_C_STATIC_CAST = re.compile(r"\bstatic_cast\s*<\s*([^<>;{}]+?)\s*>\s*\(")
_C_FUNCTIONAL_CAST = re.compile(r"(?<![\w:.])(" + "|".join(C_CAST_TYPES) + r")\s*\(")

#: A complex type a library expansion constructs from its two components (``dace::complex128(1.0, 0.0)``, a BLAS
#: ``alpha``), before or after the type rename -> the :data:`C_COMPLEX_BUILDERS` function that builds it in C. The
#: cast expression :data:`_C_FUNCTIONAL_CAST` writes would be ``(double _Complex)(1.0, 0.0)``: a cast of the comma
#: operator, so the imaginary component alone.
C_COMPLEX_CONSTRUCTORS: dict[str, str] = {
    **C_COMPLEX_BUILDERS,
    **{C_CTYPE_RENAMES[name]: builder for name, builder in C_COMPLEX_BUILDERS.items()},
}
_C_COMPLEX_CONSTRUCTOR = re.compile(r"(?<![\w:.])(" + "|".join(map(re.escape, C_COMPLEX_CONSTRUCTORS)) + r")\s*\(")

#: The scan's min / max neutral element per element type, as a C constant expression. Mirrors
#: ``dace::scan::min_identity`` / ``max_identity``: infinity where the type has one, otherwise the
#: extreme value. The C++ version is one function template; C needs the type spelled out, and the
#: call site already spells it (``min_identity<double>()``), so a constant is enough and no
#: definition has to be emitted.
C_SCAN_IDENTITIES: dict[str, tuple[str, str]] = {
    "int": ("INT_MAX", "INT_MIN"),
    "long": ("LONG_MAX", "LONG_MIN"),
    "long long": ("LLONG_MAX", "LLONG_MIN"),
    "float": ("INFINITY", "-INFINITY"),
    "double": ("INFINITY", "-INFINITY"),
    "long double": ("INFINITY", "-INFINITY"),
}

#: ``min_identity<double>()`` as a hand-written expansion writes it, qualified or not.
_C_IDENTITY_CALL = re.compile(r"(?:::)?(?:\w+::)*\b(min|max)_identity\s*<\s*([A-Za-z_][A-Za-z_ ]*?)\s*>\s*\(\s*\)")


def c_scan_identities(code: str) -> str:
    """Replace every ``min_identity<T>()`` / ``max_identity<T>()`` with its C constant.

    Must run BEFORE the qualified-name rewrite: the name carries an explicit template argument, so
    it is not an ordinary call the name table could map.

    :param code: the C++ body, with its ctypes already renamed.
    :returns: the body with the identities spelled as C.
    :raises NotImplementedError: on an element type with no ordered extreme (complex).
    """

    def replace(match: "re.Match") -> str:
        kind, ctype = match.group(1), match.group(2)
        identities = C_SCAN_IDENTITIES.get(ctype)
        if identities is None:
            raise NotImplementedError(
                f"CPF cannot spell the scan {kind} identity for {ctype!r}: it has no ordered "
                "extreme value, so only sum and product scans are defined for it."
            )
        return identities[0] if kind == "min" else identities[1]

    return _C_IDENTITY_CALL.sub(replace, code)


#: ``target = dace::find_first_index((begin), (end), [&](long long __i) -> bool { return (pred); },
#: parallel);`` -- the one statement ``ExpandFindFirstCPU`` writes.
#: Anchored on the whole statement, target included. The bounds are captured as ONE group and spliced
#: through unread: the expansion parenthesizes each of them, so they arrive as two call arguments
#: however many commas the extents contain.
_C_FIND_FIRST_CALL = re.compile(
    r"([^;{}\n]+?)\s*=\s*(?:::)?(?:[A-Za-z_]\w*::)*find_first_index\s*\(\s*"
    r"(.+?),\s*\[&\]\s*\(\s*long long\s+([A-Za-z_]\w*)\s*\)\s*->\s*bool\s*\{\s*return\s+(.+?)\s*;\s*\}"
    r"\s*,\s*([A-Za-z_]\w*)\s*\)\s*;",
    re.DOTALL,
)

#: An identifier a predicate READS: not a member after ``.`` or ``->``, and not a called function.
C_READ_IDENTIFIER: "re.Pattern" = re.compile(r"(?<![\w.])(?<!->)([A-Za-z_]\w*)\b(?!\s*\()")


def c_find_first_definition(name: str, index: str, predicate: str, parameters: tuple[str, ...]) -> str:
    """The cancelling parallel search as a C function over one predicate.

    The answer is a min-reduction and is exact; the hint is shared and races by design -- every value
    it takes is a real firing index, so a lost update costs pruning, never correctness.

    :param name: the function's name.
    :param index: the index name the predicate subscripts with.
    :param predicate: the predicate in C, parenthesized.
    :param parameters: the declarations of the names the predicate reads.
    :returns: the definition.
    """
    return "\n".join(
        (
            "static inline long long %s(long long cpf_ff_begin, long long cpf_ff_end, bool cpf_ff_parallel%s) {"
            % (name, "".join(", " + parameter for parameter in parameters)),
            "    const long long cpf_ff_simd = 64;",
            "    long long cpf_ff_best = cpf_ff_end;",
            "    long long cpf_ff_hint = cpf_ff_end;",
            "    if (cpf_ff_begin >= cpf_ff_end) return cpf_ff_end;",
            "    const long long cpf_ff_span = cpf_ff_end - cpf_ff_begin;",
            "    const long long cpf_ff_chunk = find_first_chunk(cpf_ff_span, cpf_ff_parallel);",
            "    const long long cpf_ff_chunks = (cpf_ff_span + cpf_ff_chunk - 1) / cpf_ff_chunk;",
            "    #pragma omp parallel for schedule(dynamic, 1) if (parallel : cpf_ff_parallel) "
            "reduction(min : cpf_ff_best)",
            "    for (long long cpf_ff_c = 0; cpf_ff_c < cpf_ff_chunks; ++cpf_ff_c) {",
            "        const long long cpf_ff_from = cpf_ff_begin + cpf_ff_c * cpf_ff_chunk;",
            "        long long cpf_ff_seen;",
            "        #pragma omp atomic read",
            "        cpf_ff_seen = cpf_ff_hint;",
            "        if (cpf_ff_from >= cpf_ff_seen) continue;",
            "        long long cpf_ff_hi = cpf_ff_from + cpf_ff_chunk;",
            "        if (cpf_ff_hi > cpf_ff_end) cpf_ff_hi = cpf_ff_end;",
            "        if (cpf_ff_hi > cpf_ff_seen) cpf_ff_hi = cpf_ff_seen;",
            "        long long cpf_ff_found = cpf_ff_end;",
            "        for (long long cpf_ff_b = cpf_ff_from; cpf_ff_b < cpf_ff_hi; cpf_ff_b += cpf_ff_simd) {",
            "            long long cpf_ff_to = cpf_ff_b + cpf_ff_simd;",
            "            long long cpf_ff_block = cpf_ff_end;",
            "            if (cpf_ff_to > cpf_ff_hi) cpf_ff_to = cpf_ff_hi;",
            "            // A vectorized loop cannot break, so the block is the early-exit granularity.",
            "            #pragma omp simd reduction(min : cpf_ff_block)",
            "            for (long long %s = cpf_ff_b; %s < cpf_ff_to; ++%s) {" % (index, index, index),
            "                const long long cpf_ff_v = %s ? %s : cpf_ff_end;" % (predicate, index),
            "                cpf_ff_block = cpf_ff_v < cpf_ff_block ? cpf_ff_v : cpf_ff_block;",
            "            }",
            "            if (cpf_ff_block < cpf_ff_end) {",
            "                cpf_ff_found = cpf_ff_block;",
            "                break;",
            "            }",
            "        }",
            "        if (cpf_ff_found < cpf_ff_end) {",
            "            long long cpf_ff_cur;",
            "            if (cpf_ff_found < cpf_ff_best) cpf_ff_best = cpf_ff_found;",
            "            #pragma omp atomic read",
            "            cpf_ff_cur = cpf_ff_hint;",
            "            if (cpf_ff_found < cpf_ff_cur) {",
            "                #pragma omp atomic write",
            "                cpf_ff_hint = cpf_ff_found;",
            "            }",
            "        }",
            "    }",
            "    return cpf_ff_best;",
            "}",
        )
    )


def c_find_first(code: str, site: NativeSite | None) -> str:
    """Rewrite a ``find_first_index`` call over a C++ lambda into a call to a search function of its own.

    C has no lambda and no way to hand a capturing predicate to a function, so the predicate is pasted
    into a function the site defines, and each name it reads that the site types becomes a parameter.
    Deliberately narrow: it matches the exact statement the two CPU expansions write, and anything else
    is left alone for ``dace.codegen.cpf.verify`` to report as an unlowered ``dace::`` name.

    :param code: the body, C already apart from the search.
    :param site: the typed names at the call site; its ``functions`` receives the search function.
    :returns: the body with the search spelled as C.
    :raises NotImplementedError: at a site that types no names, or for a read name C cannot spell.
    """

    def replace(match: "re.Match") -> str:
        if site is None:
            raise NotImplementedError(
                "CPF cannot render a find-first in C where no names are typed: the search "
                "function could not declare what its predicate reads"
            )
        index, predicate = match.group(3), match.group(4).strip()
        if not (predicate.startswith("(") and c_encloses(predicate, 0)):
            predicate = "(%s)" % predicate
        read = sorted(({hit.group(1) for hit in C_READ_IDENTIFIER.finditer(predicate)} & set(site.names)) - {index})
        parameters: list[str] = []
        for name in read:
            dtype, pointer = site.names[name]
            if dtype not in C_SCALAR_SPELLINGS:
                raise NotImplementedError(f"CPF cannot pass {name!r} to a C find-first: {dtype} has no C spelling here")
            parameters.append(("const %s *%s" if pointer else "%s %s") % (C_SCALAR_SPELLINGS[dtype], name))
        function = "cpf_find_first_%s" % site.label + ("_%d" % len(site.functions) if site.functions else "")
        site.functions.append(c_find_first_definition(function, index, predicate, tuple(parameters)))
        return "%s = %s(%s, %s%s);" % (
            match.group(1).strip(),
            function,
            match.group(2).strip(),
            match.group(5),
            "".join(", " + name for name in read),
        )

    return _C_FIND_FIRST_CALL.sub(replace, code)


#: A ``std::`` name that reaches the emitted text WITHOUT passing an expression printer -- a
#: library expansion's hand-written C++ body, or a pass that writes one. Neither lowering lane sees
#: those, and in C a ``std::`` name is not a name at all, so this is the table that answers them.
#:
#: Only the type-generic maths (``std::isnan`` ...) is derived from the printer tables. Every other
#: ``std::`` maths function is overloaded on its argument type, which a hand-written body does not
#: name, so it is left for ``dace.codegen.cpf.verify`` to report: renamed onto the ``double``
#: function it would compute a ``float`` argument at the wrong precision.
#:
#: The rest are the names with no printer entry at all. ``std::abort`` is the assumption guard:
#: canonicalization traps a violated symbol assumption with ``if ((N < 0)) { std::abort(); }`` and
#: DEDUPS its own guards by searching tasklet bodies for that literal text, so the spelling is
#: fixed at the source. The ``mem*`` trio and ``std::size_t`` are spelled the same in C once the
#: namespace goes. ``std::sort`` becomes the heapsort typed for its range (:func:`c_sort`), and
#: ``std::copy`` the ``memmove`` it performs (:func:`c_copy`).
C_NATIVE_RENAMES: dict[str, str] = {
    cpp: C_STD_RENAMES[runtime]
    for runtime, cpp in STD_RENAMES.items()
    if cpp.startswith("std::") and runtime in C_STD_RENAMES
}
C_NATIVE_RENAMES.update(
    {
        "std::abort": "abort",
        "std::memcpy": "memcpy",
        "std::memmove": "memmove",
        "std::memset": "memset",
        "std::size_t": "size_t",
    }
)

#: One alternation over every :data:`C_NATIVE_RENAMES` key, longest first. Every key is a distinct
#: literal ``std::`` name and no replacement value contains ``std::``, so the tens of sequential
#: passes this used to be never interacted -- folding them into one pass over the original text
#: changes nothing they could produce.
_C_NATIVE_RENAME_PATTERN = re.compile(
    r"(?:::)?\b(" + "|".join(re.escape(name) for name in sorted(C_NATIVE_RENAMES, key=len, reverse=True)) + r")\b"
)

#: An ``#include`` of a C++ standard header, which is named without an extension. A native body writes
#: one for the algorithm it calls (``<algorithm>`` for ``std::sort``); C has no such header.
C_CXX_INCLUDE: "re.Pattern" = re.compile(r"^[ \t]*#[ \t]*include[ \t]*<[A-Za-z_]\w*>[ \t]*\n?", re.MULTILINE)

#: ``std::min<T>(`` / ``std::max<T>(``, as an expansion that knows its element type writes them.
C_TYPED_MINMAX_CALL = re.compile(r"(?:::)?\bstd::(min|max)\s*<\s*([A-Za-z_][\w ]*?)\s*>\s*\(")

#: C scalar spelling -> dace type name, for an element type an expansion wrote as a template argument.
C_CTYPE_DTYPES: dict[str, str] = {
    "bool": "bool",
    "char": "int8",
    "int8_t": "int8",
    "short": "int16",
    "int16_t": "int16",
    "int": "int32",
    "int32_t": "int32",
    "long": "int64",
    "long long": "int64",
    "int64_t": "int64",
    "uint8_t": "uint8",
    "uint16_t": "uint16",
    "unsigned int": "uint32",
    "uint32_t": "uint32",
    "unsigned long": "uint64",
    "unsigned long long": "uint64",
    "uint64_t": "uint64",
    "size_t": "uint64",
    "float": "float32",
    "double": "float64",
}


def c_typed_minmax(match: "re.Match") -> str:
    """The typed C helper call that opens in place of a ``std::min<T>(`` / ``std::max<T>(``."""
    operation, ctype = match.group(1), match.group(2)
    if ctype not in C_CTYPE_DTYPES:
        raise NotImplementedError(f"CPF cannot spell std::{operation}<{ctype}> in C: no C helper takes that type")
    return "cpf_%s_%s(" % (operation, c_common_type((C_CTYPE_DTYPES[ctype],)))


#: The opening of a ``std::copy`` call, qualified or not.
C_COPY_CALL = re.compile(r"(?:::)?\bstd::copy\s*\(")


def c_copy(code: str) -> str:
    """Spell each ``std::copy(first, last, out)`` as the ``memmove`` it performs.

    The expansions that write it copy a contiguous range of a trivially copyable element type, and
    nothing CPF renders reads the output end the C++ call returns. A call whose arguments do not split
    into three is left alone, so ``dace.codegen.cpf.verify`` reports it by name.

    :param code: the body as the expansion wrote it.
    :returns: the body with every such copy as a ``memmove``.
    """

    def rewrite(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        if len(arguments) != 3:
            return None
        first, last, out = arguments
        return "memmove((%s), (%s), (size_t)((%s) - (%s)) * sizeof(*(%s)))" % (out, first, last, first, first)

    return c_rewrite_calls(code, C_COPY_CALL, rewrite)


def c_rewrite_calls(
    code: str, opening: "re.Pattern", rewrite: Callable[["re.Match", tuple[str, ...]], str | None]
) -> str:
    """Replace each call ``opening`` finds by ``rewrite(match, arguments)``; ``None`` leaves the call alone.

    :param code: the body.
    :param opening: matches a call up to and including its ``(``.
    :param rewrite: the replacement for one call, given its arguments split on top-level commas.
    :returns: the body with each rewritten call replaced.
    """
    pieces: list[str] = []
    position = 0
    for match in opening.finditer(code):
        if match.start() < position:
            continue
        depth, end = 1, match.end()
        while end < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[end], 0)
            end += 1
        arguments = c_call_arguments(code[match.end() : end - 1]) if depth == 0 else None
        replacement = rewrite(match, arguments) if arguments is not None else None
        if replacement is None:
            continue
        pieces.append(code[position : match.start()])
        pieces.append(replacement)
        position = end
    return "".join(pieces) + code[position:]


def c_encloses(text: str, start: int) -> bool:
    """Whether the bracket opening at ``text[start]`` closes at the last character of ``text``."""
    depth = 0
    for position in range(start, len(text)):
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(text[position], 0)
        if depth == 0:
            return position == len(text) - 1
    return False


#: C type spelling -> dace type name, for the type a cast or a scan identity spells.
C_SPELLED_DTYPES: dict[str, str] = {
    **C_CTYPE_DTYPES,
    **{spelling: dtype for dtype, spelling in C_SCALAR_SPELLINGS.items()},
}

#: One alternation over :data:`C_SPELLED_DTYPES`, longest first.
C_TYPE_SPELLING: str = "|".join(re.escape(name) for name in sorted(C_SPELLED_DTYPES, key=len, reverse=True))

#: The opening of a value that spells its own type: a scan identity, ``static_cast<T>(``, ``(T)(`` or
#: ``T(``. Each alternative captures the type.
C_SPELLED_VALUE: "re.Pattern" = re.compile(
    r"(?:(?:::)?(?:\w+::)*(?:min|max)_identity\s*<\s*(%s)\s*>|static_cast\s*<\s*(%s)"
    r"\s*>|\(\s*(%s)\s*\)|(%s))\s*\(" % ((C_TYPE_SPELLING,) * 4)
)

#: The opening of a subscript of a named array.
C_SUBSCRIPTED: "re.Pattern" = re.compile(r"([A-Za-z_]\w*)\s*\[")


def c_offset_base(argument: str) -> str:
    """``argument`` with its enclosing parentheses dropped and every parenthesized leftmost operand of a ``+``
    opened: ``((A + i) + j * M)`` -> ``A + i``. The leftmost operand of a pointer sum is the pointer, so the
    text left starts with the array an offset expression points into."""
    text = argument.strip()
    while text.startswith("("):
        if c_encloses(text, 0):
            text = text[1:-1].strip()
            continue
        close = c_closing(text, 0)
        if close is None or not text[close + 1 :].lstrip().startswith("+"):
            break
        text = text[1:close].strip()
    return text


def c_closing(text: str, start: int) -> int | None:
    """The position of the bracket closing the one that opens at ``text[start]``, or ``None``."""
    depth = 0
    for position in range(start, len(text)):
        depth += {"(": 1, "[": 1, ")": -1, "]": -1}.get(text[position], 0)
        if depth == 0:
            return position
    return None


def c_native_array_dtype(argument: str, site: NativeSite | None, call: str) -> str:
    """The element type of the array ``argument`` names, as ``site`` types it.

    :raises NotImplementedError: if the argument is not a typed array in scope at the site.
    """
    # An offset into an array, ``(A + (N * i))``, points at the same element type as ``A`` itself, and so does an
    # offset of an offset, ``(A + i) + j * M``, which is how a reduction over a column of a slice reaches it.
    text = c_offset_base(argument)
    offset = re.match(r"([A-Za-z_]\w*)\s*\+", text)
    name = offset.group(1) if offset is not None else text
    entry = site.names.get(name) if site is not None else None
    if entry is None or not entry[1] or entry[0] not in C_SCALAR_SPELLINGS:
        where = "no names are typed at this site" if site is None else "it is not a typed array in scope"
        raise NotImplementedError(f"CPF cannot type the array {argument!r} that {call} takes in C: {where}")
    return entry[0]


def c_native_value_dtype(argument: str, site: NativeSite | None, call: str) -> str:
    """The type of the scalar ``argument``: a typed name, an element of a typed array, or a value whose
    cast or scan identity spells its type.

    :raises NotImplementedError: for any other expression.
    """
    text = argument.strip()
    names = site.names if site is not None else {}
    entry = names.get(text)
    if entry is not None and not entry[1] and entry[0] in C_SCALAR_SPELLINGS:
        return entry[0]
    subscripted = C_SUBSCRIPTED.match(text)
    array = names.get(subscripted.group(1)) if subscripted is not None else None
    if (
        array is not None
        and array[1]
        and array[0] in C_SCALAR_SPELLINGS
        and c_encloses(text, required(subscripted).end() - 1)
    ):
        return array[0]
    # An element of an offset into a typed array, ``(A + i)[0]``: an element of ``A``.
    if text.startswith("("):
        close = c_closing(text, 0)
        rest = text[close + 1 :].lstrip() if close is not None else ""
        if rest.startswith("[") and c_encloses(rest, 0):
            try:
                return c_native_array_dtype(text[1:close], site, call)
            except NotImplementedError:
                pass
    # A scalar the expansion reads through its address, ``(&s)[0]``: the scalar itself.
    addressed = re.fullmatch(r"\(\s*&\s*([A-Za-z_]\w*)\s*\)\s*\[\s*0\s*\]", text)
    scalar = names.get(addressed.group(1)) if addressed is not None else None
    if scalar is not None and not scalar[1] and scalar[0] in C_SCALAR_SPELLINGS:
        return scalar[0]
    spelled = C_SPELLED_VALUE.match(text)
    if spelled is not None and c_encloses(text, spelled.end() - 1):
        return C_SPELLED_DTYPES[next(group for group in spelled.groups() if group is not None)]
    raise NotImplementedError(
        f"CPF cannot type the value {argument!r} that {call} takes in C: it is neither a "
        "typed name nor a value that spells its type"
    )


#: An ``ipow`` call in a hand-written body: the runtime's qualified name, or the bare one ``str`` of a dace expression
#: prints, which an expansion formats into its call (a BLAS GEMV's ``Lb**3`` rows are ``ipow(Lb, 3)``).
C_NATIVE_IPOW_CALL: "re.Pattern" = re.compile(r"(?<![\w:])(?:(?:::)?dace::math::)?ipow\s*\(")
#: A typed helper's call as C spells it, ``cpf_<helper>_<dace type>(``: its result has that type.
C_TYPED_HELPER_CALL: "re.Pattern" = re.compile(r"cpf_[a-z_]+?_(%s)\s*\(" % "|".join(C_SCALAR_SPELLINGS))


def c_native_scalar_dtype(argument: str, site: NativeSite | None, call: str) -> str:
    """The type of a scalar ``argument`` of a hand-written body: an integer or floating literal, a typed helper's
    call (:data:`C_TYPED_HELPER_CALL`), else what :func:`c_native_value_dtype` reads off the site.

    :raises NotImplementedError: for any other expression.
    """
    text = argument.strip()
    if re.fullmatch(r"[+-]?\d+[lLuU]*", text):
        return "int64"
    if re.fullmatch(r"[+-]?(\d+\.\d*|\.\d+|\d+)([eE][+-]?\d+)?[fF]?", text):
        return "float32" if text[-1] in "fF" else "float64"
    helper = C_TYPED_HELPER_CALL.match(text)
    if helper is not None and c_encloses(text, text.index("(", helper.start())):
        return helper.group(1)
    return c_native_value_dtype(text, site, call)


def c_native_ipow(code: str, site: NativeSite | None) -> str:
    """Every ``ipow`` call of a hand-written body as the typed C helper for its base's type.

    C has no overloading, so ``ipow`` is one helper per base type (``cpf_ipow_int64``), picked by the type of the
    base; the C++ dialect's ``ipow`` template is not there. Rewritten innermost first, so an ``ipow`` base is typed
    by the helper it already became.

    :raises NotImplementedError: if a base is not a value the site types.
    """
    for match in reversed(list(C_NATIVE_IPOW_CALL.finditer(code))):
        open_at = match.end() - 1
        close = c_closing(code, open_at)
        arguments = c_call_arguments(code[open_at + 1 : close]) if close is not None else None
        if arguments is None or len(arguments) != 2:
            continue
        base_type = c_native_scalar_dtype(arguments[0], site, "ipow")
        code = code[: match.start()] + c_helper_call("ipow", arguments, (base_type, "int64")) + code[close + 1 :]
    return code


def c_instance_call(name: str, arguments: tuple[str, ...]) -> str:
    """A call to the typed instantiation ``name``. The instantiation is built here, so a type combination
    it cannot take is refused at the call site that asked for it."""
    c_instance(name)
    return "%s(%s)" % (name, ", ".join(arguments))


#: The opening of a ``std::sort`` call, qualified or not.
C_SORT_CALL: "re.Pattern" = re.compile(r"(?:::)?\bstd::sort\s*\(")

#: The opening of a prefix-scan call as the ``Scan`` expansion writes it, qualified or not.
C_SCAN_CALL: "re.Pattern" = re.compile(r"(?:::)?(?:[A-Za-z_]\w*::)*\bscan_(incl|excl)_(sum|product|min|max)\s*\(")


def c_sort(code: str, site: NativeSite | None) -> str:
    """Spell each ``std::sort(first, last)`` as a call to the heapsort typed for the range's element.

    :param code: the body as the expansion wrote it.
    :param site: the typed names at the call site.
    :returns: the body with every sort as a typed call.
    :raises NotImplementedError: if the element type is unknown or has no order.
    """

    def rewrite(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        if len(arguments) != 2:
            return None
        return c_instance_call("cpf_sort_%s" % c_native_array_dtype(arguments[0], site, "std::sort"), arguments)

    return c_rewrite_calls(code, C_SORT_CALL, rewrite)


#: The opening of a reduction call once :func:`respell_reduce_entry_points` has named it.
C_REDUCE_CALL: "re.Pattern" = re.compile(r"\breduce_(seq_)?(%s)\s*\(" % "|".join(REDUCE_OPERATIONS))


#: The dace ctypes that are not their C fixed-width spelling, by dace type.
C_NATIVE_CTYPE_DTYPES: dict[str, str] = {
    "int": "int32",
    "unsigned int": "uint32",
    "long long": "int64",
    "unsigned long long": "uint64",
    "short": "int16",
    "unsigned short": "uint16",
    "char": "int8",
    "unsigned char": "uint8",
}

#: ``dace::arg_pair<V, I>`` and its calls, ``arg_pair_of`` / ``arg_max_step`` / ``arg_min_step``, as the
#: ``ArgReduce`` CPU lowering writes them once :func:`rewrite_ctypes` has spelled ``V`` and ``I`` in C.
C_ARG_PIECE: "re.Pattern" = re.compile(
    r"(?:::)?(?:\bdace::)?\barg_(pair|pair_of|(?:max|min)(?:_last)?_(?:step|combine))\s*<\s*([^<>,]+?)\s*,\s*([^<>,]+?)\s*>"
)


def c_arg_pairs(code: str) -> str:
    """Spell each argmax / argmin piece as the C instance typed for its value and index.

    :raises NotImplementedError: for a value or index type C has no spelling for.
    """
    # A dace ctype may spell a fixed-width type its own way (``int`` for int32).
    dtypes_of = {spelling: name for name, spelling in C_SCALAR_SPELLINGS.items()} | C_NATIVE_CTYPE_DTYPES

    def rewrite(match: "re.Match") -> str:
        piece, value, index = match.group(1), match.group(2), match.group(3)
        if value not in dtypes_of or index not in dtypes_of:
            raise NotImplementedError(f"CPF cannot render an argmax pair of {value} and {index} in C")
        return f"cpf_arg_{piece}_{dtypes_of[value]}_{dtypes_of[index]}"

    return C_ARG_PIECE.sub(rewrite, code)


def c_reduce(code: str, site: NativeSite | None) -> str:
    """Spell each reduction call as a call to the reduction typed for its input and its seed.

    :param code: the body as the expansion wrote it.
    :param site: the typed names at the call site.
    :returns: the body with every reduction as a typed call.
    :raises NotImplementedError: if a type is unknown, or the operation has no order over the seed's type.
    """

    def rewrite(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        if len(arguments) != 4:
            return None
        call = reduce_name(match.group(2), match.group(1) is None)
        types = (c_native_array_dtype(arguments[0], site, call), c_native_value_dtype(arguments[3], site, call))
        return c_instance_call("cpf_%s_%s" % (call, "_".join(types)), arguments)

    return c_rewrite_calls(code, C_REDUCE_CALL, rewrite)


def c_scan(code: str, site: NativeSite | None) -> str:
    """Spell each prefix-scan call as a call to the scan typed for its input, its output and its seed.

    :param code: the body as the expansion wrote it.
    :param site: the typed names at the call site.
    :returns: the body with every scan as a typed call.
    :raises NotImplementedError: if a type is unknown, or the operation has no order over the seed's type.
    """

    def rewrite(match: "re.Match", arguments: tuple[str, ...]) -> str | None:
        if len(arguments) != 5:
            return None
        call = "scan_%s_%s" % (match.group(1), match.group(2))
        types = (
            c_native_array_dtype(arguments[0], site, call),
            c_native_array_dtype(arguments[1], site, call),
            c_native_value_dtype(arguments[4], site, call),
        )
        return c_instance_call("cpf_%s_%s" % (call, "_".join(types)), arguments)

    return c_rewrite_calls(code, C_SCAN_CALL, rewrite)


def c_native_renames(code: str) -> str:
    """Spell the ``std::`` names a hand-written body carries the C way.

    ``std::min<T>`` / ``std::max<T>`` become the typed helper for ``T``. An untyped ``std::min``
    names no type to pick a helper by, so it is left for ``dace.codegen.cpf.verify`` to report.

    :param code: the body as the expansion or pass wrote it.
    :returns: the body with each :data:`C_NATIVE_RENAMES` name replaced.
    """
    if "std::" not in code:
        return code
    code = C_TYPED_MINMAX_CALL.sub(c_typed_minmax, code)
    return _C_NATIVE_RENAME_PATTERN.sub(lambda match: C_NATIVE_RENAMES[match.group(1)], code)


#: ``std::numeric_limits<T>::max()`` / ``::lowest()`` per element type, as C constants. The two
#: reduction seeds a tile reduction writes, and the only two members CPF sees. C has no class
#: template to ask, so the constant is spelled here from ``<limits.h>`` and ``<float.h>``;
#: ``lowest()`` is the most negative FINITE value, which for a float type is ``-<T>_MAX`` and NOT
#: ``<T>_MIN`` (that one is the smallest positive normal).
C_NUMERIC_LIMITS: dict[str, tuple[str, str]] = {
    "int": ("INT_MAX", "INT_MIN"),
    "unsigned int": ("UINT_MAX", "0"),
    "long": ("LONG_MAX", "LONG_MIN"),
    "unsigned long": ("ULONG_MAX", "0"),
    "long long": ("LLONG_MAX", "LLONG_MIN"),
    "unsigned long long": ("ULLONG_MAX", "0"),
    "float": ("FLT_MAX", "-FLT_MAX"),
    "double": ("DBL_MAX", "-DBL_MAX"),
    "long double": ("LDBL_MAX", "-LDBL_MAX"),
}

#: ``std::numeric_limits<double>::lowest()`` as an expansion writes it, qualified or not.
_C_NUMERIC_LIMIT_CALL = re.compile(
    r"(?:::)?(?:\w+::)*\bnumeric_limits\s*<\s*([A-Za-z_][A-Za-z_ ]*?)\s*>\s*"
    r"::\s*(max|lowest)\s*\(\s*\)"
)


def c_numeric_limits(code: str) -> str:
    """Replace every ``numeric_limits<T>::max()`` / ``::lowest()`` with its C constant.

    Must run BEFORE the qualified-name rewrite: the name carries an explicit template argument and
    a member call, so it is not an ordinary call the name table could map. Any other member is left
    alone and reaches ``dace.codegen.cpf.verify``, which reports it against the node that wrote it.

    :param code: the C++ body, with its ctypes already renamed.
    :returns: the body with the two extremes spelled as C.
    :raises NotImplementedError: on an element type with no C constant for its extremes.
    """

    def replace(match: "re.Match") -> str:
        ctype, member = match.group(1).strip(), match.group(2)
        extremes = C_NUMERIC_LIMITS.get(ctype)
        if extremes is None:
            raise NotImplementedError(
                f"CPF cannot spell numeric_limits<{ctype}>::{member}() in C: the type has no "
                "standard constant for its extremes."
            )
        return extremes[0] if member == "max" else extremes[1]

    return _C_NUMERIC_LIMIT_CALL.sub(replace, code)


#: ``target = dace::detect_collision(idx, (n)[, owner, (capacity)], parallel);`` -- the one
#: statement the two ``ScatterConflictCheck`` CPU expansions write. The arguments are split by
#: :func:`c_call_arguments` rather than by the pattern, so an argument carrying a comma inside
#: parentheses cannot split the call.
_C_DETECT_COLLISION_CALL = re.compile(
    r"([^;{}\n]+?)\s*=\s*(?:::)?(?:[A-Za-z_]\w*::)*detect_collision\s*\((.*?)\)\s*;", re.DOTALL
)


def c_call_arguments(printed: str) -> tuple[str, ...] | None:
    """Split one call's printed argument list on its TOP-LEVEL commas.

    :param printed: the text between a call's outermost parentheses.
    :returns: the arguments, or ``None`` if the brackets do not balance -- which means the pattern
              that produced ``printed`` stopped inside a nested call, and the caller must leave the
              text alone.
    """
    arguments: list[str] = []
    depth = 0
    current = []
    for character in printed:
        if character in "([":
            depth += 1
        elif character in ")]":
            depth -= 1
            if depth < 0:
                return None
        if character == "," and depth == 0:
            arguments.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if depth != 0:
        return None
    arguments.append("".join(current).strip())
    return tuple(arguments)


def c_detect_collision(code: str, site: NativeSite | None) -> str:
    """Rewrite a ``detect_collision`` call into a call to the C function typed for its arrays.

    C cannot spell the C++ overload pair as a call: the index and tag element types are template
    parameters and the two arities are two functions. The call site knows both -- five arguments
    means a tag array the caller wired, three means the function sizes one of its own -- and ``site``
    types the arrays.

    Must run BEFORE the qualified-name rewrite, which would otherwise leave the C++ call shape in
    place with only its namespace stripped. A call of any other arity is LEFT ALONE, so it reaches
    ``dace.codegen.cpf.verify`` and is reported against the node that emitted it.

    :param code: the C++ body, with its ctypes already renamed.
    :param site: the typed names at the call site.
    :returns: the body with the check spelled as C.
    :raises NotImplementedError: if an array's element type is unknown at the site.
    """

    def replace(match: "re.Match") -> str:
        arguments = c_call_arguments(match.group(2))
        if arguments is None:
            return match.group(0)
        target = match.group(1).strip()
        if len(arguments) == 5:
            name = "cpf_detect_collision_%s_%s" % (
                c_native_array_dtype(arguments[0], site, "detect_collision"),
                c_native_array_dtype(arguments[2], site, "detect_collision"),
            )
        elif len(arguments) == 3:
            name = "cpf_detect_collision_sized_%s" % c_native_array_dtype(arguments[0], site, "detect_collision")
        else:
            return match.group(0)
        return "%s = %s;" % (target, c_instance_call(name, arguments))

    return _C_DETECT_COLLISION_CALL.sub(replace, code)


def c_cast_native_code(code: str) -> str:
    """Rewrite the C++ casts in a hand-written body to C casts.

    Native tasklet bodies are emitted verbatim, so this is the only point at which a library
    expansion's own C++ can be re-spelled. The two forms that actually appear are the ``Scan``
    expansion's ``static_cast<long>(n)`` length and its ``double(0)`` seed.

    :param code: the body, after :func:`rewrite_native_code` has re-spelled its ``dace::`` names.
    :returns: the body with both cast forms written as ``(type)(...)``.
    """
    code = _C_STATIC_CAST.sub(lambda match: "(%s)(" % match.group(1), code)
    return _C_FUNCTIONAL_CAST.sub(lambda match: "(%s)(" % match.group(1), c_complex_constructors(code))


def c_complex_constructors(code: str) -> str:
    """Rewrite every two-component complex construction in a hand-written body to its C builder.

    :param code: the body, with or without its ``dace::`` complex types renamed.
    :returns: the body with each ``T(re, im)`` written as ``cpf_complex<bits>(re, im)``; a one-argument
              construction is a cast, left to :func:`c_cast_native_code`.
    """
    pieces, done = [], 0
    for match in _C_COMPLEX_CONSTRUCTOR.finditer(code):
        if match.start() < done:
            continue
        depth, close = 1, match.end()
        while close < len(code) and depth:
            depth += {"(": 1, ")": -1}.get(code[close], 0)
            close += 1
        arguments = c_call_arguments(code[match.end() : close - 1]) if depth == 0 else None
        if arguments is None or len(arguments) != 2:
            continue
        builder = C_COMPLEX_CONSTRUCTORS[match.group(1)]
        pieces += [code[done : match.start()], "%s(%s)" % (builder, ", ".join(map(c_complex_constructors, arguments)))]
        done = close
    return "".join(pieces) + code[done:]
