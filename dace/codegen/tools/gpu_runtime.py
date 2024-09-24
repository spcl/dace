# Copyright 2019-2023 ETH Zurich and the DaCe authors. All rights reserved.
"""
GPU runtime testing functionality. Used for checking error codes after GPU-capable SDFG execution.
"""
import ctypes
from typing import Optional


class GPURuntime:
    """
    GPU runtime object containing the library (CUDA / HIP) and some functions to query errors.
    """

    def __init__(self, backend_name: str, path: str) -> None:
        self.backend = backend_name
        self.library = ctypes.CDLL(path)

        # Prefetch runtime functions
        self._geterrorstring = getattr(self.library, f'{self.backend}GetErrorString')
        self._geterrorstring.restype = ctypes.c_char_p
        self._getlasterror = getattr(self.library, f'{self.backend}GetLastError')

    def get_error_string(self, err: int) -> str:
        # Obtain the error string
        return self._geterrorstring(err).decode('utf-8')

    def get_last_error(self) -> int:
        return self._getlasterror()

    def get_last_error_string(self) -> Optional[str]:
        res: int = self._getlasterror()
        if res == 0:
            return None

        # Obtain the error string
        return self.get_error_string(res)
