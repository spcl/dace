
from typing import Iterator, Tuple, TypeVar, Optional


T = TypeVar("T")


def for_step(iterator: Iterator[T]) -> Tuple[Optional[T], bool]:

    try:
        value = next(iterator)
        return value, True
        
    except StopIteration:
        return None, False


def for_iter(iterator: Iterator[T]) -> Iterator[Tuple[T, bool]]:

    for value in iterator:
        yield value, True
    else:
        yield None, False
