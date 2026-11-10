# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Exceptions of the AI library."""


class AIExpansionError(Exception):
    """
    Raised when a tasklet cannot be generated: a missing SDK or credentials, a provider error, an
    invalid answer, or code that still does not compile after the configured repair attempts.
    """
