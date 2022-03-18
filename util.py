# -*- coding: utf-8 -*-

from typing import Any, Sized, Union

import numpy as np

Array = Union['np.ndarray[Any, np.dtype[Any]]']


class FakeGenericMeta(type):
    def __getitem__(cls, _item):
        return cls


_zip_strict_sentinel = object()


def _zip_strict_iters(*iterables):
    if not iterables:
        return
    iterators = tuple(map(iter, iterables))
    del iterables
    while True:
        pops = tuple(next(it, _zip_strict_sentinel) for it in iterators)
        if any(pop is _zip_strict_sentinel for pop in pops):
            break
        yield pops
    not_exhausted = tuple(i for i, pop in enumerate(pops) if pop is not _zip_strict_sentinel)
    if not_exhausted:
        raise ValueError('Some zip_strict arguments not exhausted: {}'.format(not_exhausted))


# Guess who's getting impatient waiting for Python 3.10?
def zip_strict(*iterables):
    if all(isinstance(it, Sized) for it in iterables):
        if len(set(map(len, iterables))) > 1:
            raise ValueError('Arguments to zip_strict have inconsistent lengths: {}'.format(tuple(map(len, iterables))))
        return zip(*iterables)
    return _zip_strict_iters(*iterables)


def zipstar(it):
    return tuple(zip(*it))


def zipstar_strict(it):
    return tuple(zip_strict(*it))
