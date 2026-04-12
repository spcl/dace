# Copyright 2019-2025 ETH Zurich and the DaCe authors. All rights reserved.

from .utils import *

# ``from x import *`` is used to not break compatibility with previous versions of dace
from .array_creation import *
from .array_creation_dace import *
from .array_creation_cupy import *
from .array_manipulation import *
from .array_metadata import *
from .filtering import *
from .fft import *
from .linalg import *
from .misc import *
from .mpi import *
from .operators import *
from .pymath import *
from .reduction import *
from .ufunc import *

# Lightweight descriptor-inference registrations for the schedule-tree frontend.
# Imported for side effects (populates Replacements._dtype_rep).
from . import type_inference as _type_inference  # noqa: F401
