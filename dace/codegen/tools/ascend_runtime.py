import ctypes
from typing import Optional

class AscendRuntime:
    def __init__(self, runtime_lib_path: str, acl_lib_path: str) -> None:
        self.runtime_library = ctypes.CDLL(runtime_lib_path, mode=ctypes.RTLD_GLOBAL)
        self.acl_library = ctypes.CDLL(acl_lib_path, mode=ctypes.RTLD_GLOBAL)

        self._getlasterror_msg = getattr(self.acl_library, f'aclGetRecentErrMsg')

    #def load_kernel_lib(self, kernel_lib_path: str):
    #    self.kernel_library = ctypes.CDLL(kernel_lib_path, mode=ctypes.RTLD_GLOBAL)

    def get_error_string(self, err: int) -> str:
        return self._getlasterror_msg().decode('utf-8')

    def get_last_error(self) -> int:
        res: int = self._getlasterror_msg().decode('utf-8')
        if res == "":
            return None

        return 1

    def get_last_error_string(self) -> Optional[str]:
        return self._getlasterror_msg().decode('utf-8')