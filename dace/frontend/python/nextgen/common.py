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
