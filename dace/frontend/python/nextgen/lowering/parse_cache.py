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
from concurrent.futures import Future
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Hashable, List


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
