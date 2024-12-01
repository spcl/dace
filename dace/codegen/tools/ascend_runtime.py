import ctypes
from typing import Optional, Any

class AscendRuntime:
    def __init__(self, path: str) -> None:
        self.library = ctypes.CDLL(path)

    def get_error_string(self, err: int) -> str:
        return ""

    def get_last_error(self) -> int:
        return ""

    def get_last_error_string(self) -> Optional[str]:
        return ""