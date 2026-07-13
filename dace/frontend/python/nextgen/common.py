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
    """Raised when a program uses a Python feature the frontend does not support yet."""
    pass


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
