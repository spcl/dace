# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Shared errors and helpers for the next-generation Python frontend.
"""
import ast
from typing import Optional


class FrontendError(Exception):
    """Base class for next-generation frontend errors."""

    def __init__(self, message: str, filename: Optional[str] = None, node: Optional[ast.AST] = None):
        self.filename = filename
        self.node = node
        location = ''
        if node is not None and hasattr(node, 'lineno'):
            location = f' (in File "{filename or "<unknown>"}", line {node.lineno})'
        super().__init__(message + location)


class UnsupportedFeatureError(FrontendError):
    """
    Raised when a program uses a Python feature the frontend does not support
    yet.

    :param category: Stable kebab-case gap category (e.g. ``memlet-parse``,
        ``join-merge``) used for callback provenance: when the error falls back
        to a Python callback, the category is rendered as a ``[category]``
        prefix on the callback reason. ``None`` lets the catch site supply its
        own default.
    """

    def __init__(self,
                 message: str,
                 filename: Optional[str] = None,
                 node: Optional[ast.AST] = None,
                 category: Optional[str] = None):
        super().__init__(message, filename, node)
        self.category = category


class CanonicalViolationError(FrontendError):
    """
    Raised when the output of the canonicalization stage violates the Canonical
    Python AST (CPA) contract. This always indicates a frontend bug, not a user
    error: canonicalization is total and must either produce canonical nodes or
    explicit ``OpaqueStmt`` markers.
    """
    pass


class TreeVerificationError(FrontendError):
    """
    Raised when a produced schedule tree violates the frontend-legal node
    contract (e.g., leftover ``StatementNode``, dangling container references).
    This always indicates a frontend bug, not a user error.
    """
    pass


#: Data-descriptor attributes (``A.T``, ``A.real``, ...) whose replacement-
#: registry ATTRIBUTE-family entries (``Replacements.get_attribute`` /
#: ``get_attribute_descriptor_inference``, see
#: ``dace.frontend.common.op_repository``) the next-generation frontend has
#: dedicated lowering for (``lowering.dispatch.resolve_attribute_data``).
#:
#: Shared between ``semantics.inference`` and ``lowering.dispatch`` so the two
#: stay in lockstep: inference must only report a ``'data'`` result for an
#: attribute name in this set, because dispatch is only guaranteed to handle
#: those names before an expression reaches the generic elementwise
#: computation mechanism. A registry attribute with inference support but no
#: matching lowering path would otherwise be typed successfully and then have
#: its (unresolved) attribute access substituted verbatim into generated
#: tasklet code instead of degrading cleanly to a callback.
SUPPORTED_DATA_ATTRIBUTES = frozenset({'T', 'real', 'imag', 'flat'})

#: Real module paths under which a registry-facing name is actually defined,
#: rewritten to the shorter form used as replacement-registry keys (e.g. a
#: callee resolved to its real module ``dace.frontend.python.wrappers.ndarray``
#: normalizes to the registry key ``dace.ndarray``). Keyed by the real-module
#: prefix (with a trailing dot) so a longest-match-first scan isn't needed --
#: entries currently don't nest.
_QUALNAME_MODULE_REWRITES = {
    'dace.frontend.python.wrappers.': 'dace.',
    'dace.frontend.operations.': 'dace.',
}


def normalize_qualname(name: Optional[str]) -> Optional[str]:
    """
    Collapse a resolved qualified name to the registry-facing form used as
    keys in :class:`~dace.frontend.common.op_repository.Replacements`.

    Single shared normalization point for the next-generation frontend: every
    site that turns a callee expression into a qualified name
    (:meth:`~dace.frontend.python.nextgen.semantics.inference.InferenceService.resolve_callee`
    is the only one) should route through this function, rather than each
    downstream consumer (descriptor inference, call-lowering dispatch)
    hand-rolling its own retry against a
    ``getattr(call.func, 'qualname', None) or astutils.rname(call.func)``
    fallback for the same ``_QUALNAME_MODULE_REWRITES``-shaped mismatch.

    :param name: A qualified name, or None.
    :return: ``name`` rewritten to its registry-facing form (unchanged if it
             does not start with a known real-module prefix), or None if
             ``name`` is None.
    """
    if name is None:
        return None
    for prefix, replacement in _QUALNAME_MODULE_REWRITES.items():
        if name.startswith(prefix):
            return replacement + name[len(prefix):]
    return name
