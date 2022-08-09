
from typing import Iterable
from ssapy import FunctionReturn


class int:

    def __add__(self: int, other: int) -> FunctionReturn[int, {}]:  # type: ignore
        ...

    def __radd__(self: int, other: int) -> FunctionReturn[int, {}]:  # type: ignore
        ...

    def __sub__(self: int, other: int) -> FunctionReturn[int, {}]:  # type: ignore
        ...

    def __and__(self: int, other: int) -> FunctionReturn[int, {}]:  # type: ignore
        ...


class float:

    def __add__(self: float, other: float) -> FunctionReturn[float, {}]:  # type: ignore
        ...

    def __radd__(self: float, other: float) -> FunctionReturn[float, {}]:  # type: ignore
        ...

    def __rsub__(self: float, other: float) -> FunctionReturn[float, {}]:  # type: ignore
        ...


class str:

    def __add__(self: str, other: str) -> FunctionReturn[str, {}]:  # type: ignore
        ...

    def __radd__(self: str, other: str) -> FunctionReturn[str, {}]:  # type: ignore
        ...


def range(stop: int, /) -> FunctionReturn[Iterable[int], {}]:  # type: ignore
    ...
