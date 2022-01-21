# -*- coding: utf-8 -*-

from abc import ABCMeta
from typing import Any, Iterable, Iterator, Sized, Tuple, TypeVar, Union, overload
import numpy as np

T = TypeVar('T')
T1 = TypeVar('T1')
T2 = TypeVar('T2')
T3 = TypeVar('T3')
T4 = TypeVar('T4')
T5 = TypeVar('T5')


Array = Union['np.ndarray[Any, np.dtype[Any]]']


class FakeGenericMeta(type):
    def __getitem__(cls, item: Any) -> Any: ...


class FakeGenericABCMeta(FakeGenericMeta, ABCMeta):
    pass


class SizedIterable(Sized, Iterable[T], metaclass=ABCMeta):
    pass

@overload
def zip_strict(__iter1: Iterable[T1]) -> Iterator[Tuple[T1]]: ...
@overload
def zip_strict(__iter1: Iterable[T1], __iter2: Iterable[T2]) -> Iterator[Tuple[T1, T2]]: ...
@overload
def zip_strict(__iter1: Iterable[T1], __iter2: Iterable[T2], __iter3: Iterable[T3]) -> Iterator[Tuple[T1, T2, T3]]: ...
@overload
def zip_strict(
    __iter1: Iterable[T1], __iter2: Iterable[T2], __iter3: Iterable[T3], __iter4: Iterable[T4]
) -> Iterator[Tuple[T1, T2, T3, T4]]: ...
@overload
def zip_strict(
    __iter1: Iterable[T1], __iter2: Iterable[T2], __iter3: Iterable[T3], __iter4: Iterable[T4], __iter5: Iterable[T5]
) -> Iterator[Tuple[T1, T2, T3, T4, T5]]: ...
@overload
def zip_strict(
    __iter1: Iterable[Any],
    __iter2: Iterable[Any],
    __iter3: Iterable[Any],
    __iter4: Iterable[Any],
    __iter5: Iterable[Any],
    __iter6: Iterable[Any],
    *iterables: Iterable[Any],
) -> Iterator[Tuple[Any, ...]]: ...


@overload
def zipstar(it: Iterable[Tuple[T1]]) -> Tuple[Tuple[T1, ...]]: ...
@overload
def zipstar(it: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]: ...
@overload
def zipstar(it: Iterable[Tuple[T1, T2, T3]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...], Tuple[T3, ...]]: ...
@overload
def zipstar(
    it: Iterable[Tuple[T1, T2, T3, T4]]
) -> Tuple[Tuple[T1, ...], Tuple[T2, ...], Tuple[T3, ...], Tuple[T4, ...]]: ...
@overload
def zipstar(
    it: Iterable[Tuple[T1, T2, T3, T4, T5]]
) -> Tuple[Tuple[T1, ...], Tuple[T2, ...], Tuple[T3, ...], Tuple[T4, ...], Tuple[T5, ...]]: ...
@overload
def zipstar(it: Iterable[Iterable[Any]]) -> Tuple[Tuple[Any, ...], ...]: ...


@overload
def zipstar_strict(it: Iterable[Tuple[T1]]) -> Tuple[Tuple[T1, ...]]: ...
@overload
def zipstar_strict(it: Iterable[Tuple[T1, T2]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...]]: ...
@overload
def zipstar_strict(it: Iterable[Tuple[T1, T2, T3]]) -> Tuple[Tuple[T1, ...], Tuple[T2, ...], Tuple[T3, ...]]: ...
@overload
def zipstar_strict(
    it: Iterable[Tuple[T1, T2, T3, T4]]
) -> Tuple[Tuple[T1, ...], Tuple[T2, ...], Tuple[T3, ...], Tuple[T4, ...]]: ...
@overload
def zipstar_strict(
    it: Iterable[Tuple[T1, T2, T3, T4, T5]]
) -> Tuple[Tuple[T1, ...], Tuple[T2, ...], Tuple[T3, ...], Tuple[T4, ...], Tuple[T5, ...]]: ...
@overload
def zipstar_strict(it: Iterable[Iterable[Any]]) -> Tuple[Tuple[Any, ...], ...]: ...
