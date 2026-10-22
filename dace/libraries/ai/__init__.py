# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""
Library node implementations written by a language model.

Provides the reserved ``'ai'`` implementation, available on every library node, and
:class:`~dace.libraries.ai.nodes.ai_node.AINode`, a library node described in natural language
(``dace.ai(...)`` in the Python frontend). The model is selected by the ``ai.provider``
configuration entry.
"""

try:
    import pydantic  # noqa: F401
except ImportError as e:
    raise ImportError("The DaCe AI library needs pydantic: pip install 'dace[agent]'") from e

from dace.library import register_library
from dace.sdfg.nodes import AI_IMPLEMENTATION_NAME

from .expansion import ExpandAI
from .exceptions import AIExpansionError
from .nodes import AINode, AITasklet
from .iterate import refine, history, rollback, show, pin, unpin, sessions
from .diagnose import build, repair
from .exchange import generate_prompts, read_prompt_responses

# Environments generated earlier are registered again, so that previously expanded SDFGs resolve them
from .environments import load_generated_environments

load_generated_environments()

register_library(__name__, "AI")
