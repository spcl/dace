# Copyright 2019-2021 ETH Zurich and the DaCe authors. All rights reserved.
""" Precompiled DaCe program/method cache. """

from collections import OrderedDict
from dace import config, data as dt
from dace.sdfg.sdfg import SDFG
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Set, Tuple

# Type hints
ArgTypes = Dict[str, dt.Data]
ConstantTypes = Dict[str, Any]
EvalCallback = Callable[[str], Any]


# Adapted from https://stackoverflow.com/a/2437645/6489142
class LimitedSizeDict(OrderedDict):
    def __init__(self, *args, **kwds):
        self.size_limit = kwds.pop("size_limit", None)
        OrderedDict.__init__(self, *args, **kwds)
        self._check_size_limit()

    def __setitem__(self, key, value):
        OrderedDict.__setitem__(self, key, value)
        self._check_size_limit()

    def _check_size_limit(self):
        if self.size_limit is not None:
            while len(self) > self.size_limit:
                self.popitem(last=False)


def _make_hashable(obj):
    try:
        hash(obj)
        return obj
    except TypeError:
        return repr(obj)


@dataclass
class ProgramCacheKey:
    """ A key object representing a single instance of a DaCe program. """
    arg_types: ArgTypes
    closure_types: ArgTypes
    closure_constants: ConstantTypes

    def __init__(self, arg_types: ArgTypes, closure_types: ArgTypes, closure_constants: ConstantTypes) -> None:
        self.arg_types = arg_types
        self.closure_types = closure_types
        self.closure_constants = closure_constants
        # Freeze entry
        self._tuple = (
            tuple((k, str(v.to_json())) for k, v in sorted(arg_types.items())),
            tuple((k, str(v.to_json())) for k, v in sorted(closure_types.items())),
            tuple((k, _make_hashable(v)) for k, v in sorted(closure_constants.items())),
        )

    def __hash__(self) -> int:
        return hash(self._tuple)

    def __eq__(self, o: 'ProgramCacheKey') -> bool:
        return self._tuple == o._tuple


@dataclass
class ProgramCacheEntry:
    """
    A value object representing a cache entry of a DaCe program. Contains
    the parsed SDFG and the compiled SDFG object.
    """
    sdfg: SDFG
    compiled_sdfg: 'dace.codegen.compiled_sdfg.CompiledSDFG'


class DaceProgramCache:
    def __init__(self, evaluate: EvalCallback, size: Optional[int] = None) -> None:
        """ 
        Initializes a DaCe program cache.
        :param evaluate: A callback that can evaluate constants at call time.
        :param size: The cache size (if not given, uses the default value from
                     the configuration).
        """
        self.eval_callback = evaluate
        self.size = size or config.Config.get('frontend', 'cache_size')
        self.cache: OrderedDict[ProgramCacheKey, ProgramCacheEntry] = LimitedSizeDict(size_limit=size)

    def clear(self):
        """ Clears the program cache. """
        self.cache.clear()

    def _evaluate_constants(self, constants: Set[str], extra_constants: Dict[str, Any] = None) -> ConstantTypes:
        # Evaluate closure constants at call time
        return {k: self.eval_callback(k, extra_constants) for k in constants}

    def _evaluate_descriptors(self, arrays: Set[str], extra_constants: Dict[str, Any] = None) -> ConstantTypes:
        # Evaluate closure array types at call time
        return {k: dt.create_datadescriptor(self.eval_callback(k, extra_constants)) for k in arrays}

    def make_key(self,
                 argtypes: ArgTypes,
                 specified_args: Set[str],
                 closure_types: Set[str],
                 closure_constants: Set[str],
                 extra_constants: Dict[str, Any] = None) -> ProgramCacheKey:
        """ Creates a program cache key from the given arguments. """
        adescs = self._evaluate_descriptors(closure_types, extra_constants)
        cvals = self._evaluate_constants(closure_constants, extra_constants)
        # Filter out default arguments that were unspecified (for unique cache keys)
        argtypes = {k: v for k, v in argtypes.items() if k in specified_args}
        key = ProgramCacheKey(argtypes, adescs, cvals)
        return key

    def add(self, key: ProgramCacheKey, sdfg: SDFG, compiled_sdfg: 'dace.codegen.compiled_sdfg.CompiledSDFG') -> None:
        """ Adds a new entry to the program cache. """
        self.cache[key] = ProgramCacheEntry(sdfg, compiled_sdfg)

    def get(self, key: ProgramCacheKey) -> ProgramCacheEntry:
        """
        Returns an existing entry if in the program cache, or raises KeyError
        otherwise.
        """
        return self.cache[key]

    def has(self, key: ProgramCacheKey) -> bool:
        """ Returns True iff the given entry exists in the program cache. """
        if len(self.cache) == 0:
            return False
        return key in self.cache

    def pop(self) -> None:
        """ Remove the first entry from the cache. """
        self.cache.popitem(last=False)
