# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
import os
from dace import config, library
from typing import Dict, List


@library.environment
class TBLIS:
    """TBLIS: native, transpose-free CPU tensor contraction (BLIS-style microkernel).

    Discovered via the ``TBLIS_ROOT`` env var (``$TBLIS_ROOT/include`` + ``$TBLIS_ROOT/lib``);
    falls back to a system-installed ``-ltblis``/``-ltci`` on the default search path.
    """

    cmake_minimum_version = None
    cmake_packages: List[str] = []
    cmake_variables: Dict[str, str] = {}
    cmake_compile_flags: List[str] = []
    cmake_link_flags = ["-lpthread"]
    cmake_files: List[str] = []

    headers = ["tblis/tblis.h"]
    state_fields: List[str] = []
    init_code = ""
    finalize_code = ""
    dependencies: List[str] = []

    @staticmethod
    def cmake_includes():
        if 'TBLIS_ROOT' in os.environ:
            return [os.path.join(os.environ['TBLIS_ROOT'], 'include')]
        return []

    @staticmethod
    def cmake_libraries():
        # libtci is TBLIS's threading-runtime companion; both ship from the same build.
        if 'TBLIS_ROOT' in os.environ:
            prefix = config.Config.get('compiler', 'library_prefix')
            suffix = config.Config.get('compiler', 'library_extension')
            libdir = os.path.join(os.environ['TBLIS_ROOT'], 'lib')
            libs = []
            for name in ('tblis', 'tci'):
                libfile = os.path.join(libdir, prefix + name + '.' + suffix)
                if os.path.isfile(libfile):
                    libs.append(libfile)
            if libs:
                return libs
        return ['tblis', 'tci']
