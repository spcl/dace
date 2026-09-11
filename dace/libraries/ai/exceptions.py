# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
""" Exceptions raised by the AI-generated library node expansion. """


class AIExpansionError(Exception):
    """
    Raised when an AI-generated library node expansion cannot be produced.

    This covers every failure mode that is not a bug in DaCe itself: a missing provider SDK, a
    missing or empty API key, a provider-side error, a malformed model response, or generated code
    that still fails to compile after the configured number of repair attempts.
    """
    pass
