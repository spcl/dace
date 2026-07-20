# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Per-program cache of preprocessed and canonicalized callee ASTs.

Inlining a nested ``@dace.program`` call runs the two expensive, caller-state-
independent stages — closure preprocessing and canonicalization — through this
cache, so a callee invoked from multiple call sites with the same
specialization (argument descriptors, constant values, bound method object)
parses exactly once per :func:`~dace.frontend.python.nextgen.build_schedule_tree`
invocation. The cache lives on the
:class:`~dace.frontend.python.nextgen.semantics.context.ProgramContext`, so
nested inline scopes share it automatically.

Entries are keyed by callee identity plus a specialization key derived from
the mapped arguments (see ``rules/calls.py::_map_arguments``); the cached
value holds the *canonical body* (which lowering mutates — every consumer
must deep-copy it), the callee's resolved globals, and the preprocessing
closure (shared deliberately: closure-array descriptor identity drives
qualified-name deduplication in ``register_closure_array``).

The get-or-parse primitive is future-based and thread-safe so a later
speculative parallel warm-up can share work with the sequential lowering
path; failures cache too (the future re-raises for every call site,
preserving the per-site callback fallback).
"""
import ast
import threading
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Hashable, Iterator, List, Set, Tuple


@dataclass
class CalleeParse:
    """The caller-independent parse of one specialized callee."""
    canonical_body: List[ast.stmt]  #: Canonical statements — deep-copy before lowering (lowering mutates)
    program_globals: Dict[str, Any]  #: The callee's resolved globals (for the inline scope)
    closure: Any  #: The preprocessing closure (callbacks, constants, closure arrays)
    filename: str  #: Source filename of the callee


@dataclass
class CalleeParseCache:
    """Future-based per-parse cache of specialized callee parses."""
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _entries: Dict[Hashable, Future] = field(default_factory=dict)

    def get_or_parse(self, key: Hashable, parse: Callable[[], CalleeParse]) -> CalleeParse:
        """
        Return the cached parse for ``key``, running ``parse`` exactly once
        per key across all requesters. A raised exception caches as well and
        re-raises for every requester of the same key.
        """
        with self._lock:
            future = self._entries.get(key)
            owner = future is None
            if owner:
                future = Future()
                self._entries[key] = future
        if not owner:
            return future.result()
        try:
            result = parse()
        except BaseException as error:
            future.set_exception(error)
            raise
        future.set_result(result)
        return result


def warm_nested_parses(body: List[ast.stmt], context) -> None:
    """
    Speculatively pre-parse nested ``@dace.program`` callees of a canonical
    program body, in parallel, before sequential lowering begins.

    Call sites whose arguments are statically resolvable against the initial
    binding environment (program arguments, closure arrays, literals) get
    their exact specialization key computed through the same
    ``_map_arguments`` code path lowering uses; anything else is skipped and
    parses inline on demand. Speculation is sound by construction: a
    mismatched guess is only a cache miss, never a wrong reuse, because
    lowering always recomputes the authoritative key.

    Unique callees parse in a thread pool (with a progress bar for lengthy
    parses, matching the old frontend's ``resolve_function_calls``); each
    worker recursively warms ITS callee's own nested calls depth-first in the
    same thread — bottom-up ordering without nested executor submission
    (which could deadlock a saturated pool). Workers touch only their own
    inputs and the (thread-safe) shared cache; the caller's context is read,
    never written. Parse failures stay cached and surface per call site as
    callback fallbacks during lowering.

    :param body: The canonical (post-Stage-1) statement list.
    :param context: The program's semantic context
                    (:class:`~dace.frontend.python.nextgen.semantics.context.ProgramContext`);
                    its ``parse_cache`` receives the warmed entries.
    """
    from dace.cli.progress import optional_progressbar
    tasks = _collect_warm_tasks(body, context, visited=set())
    if not tasks:
        return
    if len(tasks) == 1:
        _run_warm_task(context.parse_cache, *tasks[0])
        return
    with ThreadPoolExecutor() as pool:
        futures = [pool.submit(_run_warm_task, context.parse_cache, key, thunk) for key, thunk in tasks]
        for _ in optional_progressbar(as_completed(futures), title='Parsing nested DaCe functions', n=len(futures)):
            pass


def _run_warm_task(cache: CalleeParseCache, key: Hashable, thunk: Callable[[], CalleeParse]) -> None:
    try:
        cache.get_or_parse(key, thunk)
    except Exception:
        pass  # Cached; the call site falls back to a callback during lowering


def _collect_warm_tasks(body: List[ast.stmt], context,
                        visited: Set[int]) -> List[Tuple[Hashable, Callable[[], CalleeParse]]]:
    """
    Warm tasks (cache key + parse thunk) for the statically-resolvable
    ``@dace.program`` call sites of a canonical body, deduplicated by key.
    ``visited`` carries the callee functions already on the warm path
    (recursion guard, mirroring the lowering-time ``inline_stack`` check).
    """
    # Deferred imports: this module must stay import-light (the semantic
    # context imports it), and the rules package imports the semantic layer.
    from dace.frontend.python.nextgen.lowering.rules import calls
    from dace.frontend.python.nextgen.semantics.inference import InferenceService

    state = SimpleNamespace(context=context, inference=InferenceService(context))
    tasks: List[Tuple[Hashable, Callable[[], CalleeParse]]] = []
    seen: Set[Hashable] = set()
    for call, callee in _dace_program_calls(body, context):
        if id(callee.f) in visited:
            continue
        try:
            argtypes, callee_globals, _, _, injected_defaults, spec_key, _ = calls._map_arguments(call, callee, state)
        except Exception:
            continue  # Not statically resolvable here; parses inline on demand
        key = (id(callee), callee.resolve_functions, id(callee.methodobj), spec_key)
        if key in seen:
            continue
        seen.add(key)
        tasks.append((key,
                      _warm_thunk(callee, argtypes, callee_globals, injected_defaults, context.parse_cache,
                                  visited | {id(callee.f)})))
    return tasks


def _warm_thunk(callee: Any, argtypes: Dict[str, Any], callee_globals: Dict[str, Any], injected_defaults: set,
                cache: CalleeParseCache, visited: Set[int]) -> Callable[[], CalleeParse]:
    """
    A parse thunk that, after parsing its callee, synchronously warms the
    callee's own nested calls (depth-first, in the same worker thread) against
    the shared cache.
    """

    def parse() -> CalleeParse:
        from dace.frontend.python.nextgen.lowering.rules import calls
        from dace.frontend.python.nextgen.semantics.context import ProgramContext
        result = calls._parse_callee(callee, argtypes, callee_globals, injected_defaults)
        child_context = ProgramContext(callee.name, result.filename, argtypes, result.program_globals, {})
        for child_key, child_thunk in _collect_warm_tasks(result.canonical_body, child_context, visited):
            _run_warm_task(cache, child_key, child_thunk)
        return result

    return parse


def _dace_program_calls(body: List[ast.stmt], context) -> Iterator[Tuple[ast.Call, Any]]:
    """
    The ``@dace.program`` call sites of a canonical body: callees embedded by
    preprocessing as constant nodes, or names resolving through the program
    globals. Opaque/tasklet leaves hide their contents (``_fields = ()``), so
    the walk never descends into interpreter-bound or tasklet code.
    """
    from dace.frontend.python.parser import DaceProgram  # Deferred to avoid an import cycle
    for statement in body:
        for node in ast.walk(statement):
            if not isinstance(node, ast.Call):
                continue
            callee = None
            if isinstance(node.func, ast.Constant) and isinstance(node.func.value, DaceProgram):
                callee = node.func.value
            elif isinstance(node.func, ast.Name):
                value = context.globals.get(node.func.id)
                if isinstance(value, DaceProgram):
                    callee = value
            if callee is not None:
                yield node, callee
