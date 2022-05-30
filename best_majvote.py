#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

# pytype: skip-file

from __future__ import annotations

import ctypes
import functools
import itertools
import multiprocessing
import pickle
import sys
import warnings
from multiprocessing.sharedctypes import RawArray
from typing import TYPE_CHECKING, Generic, TypedDict, TypeVar, Union

import numba as nb
import numpy as np
from numpy.typing import NDArray
from scipy.stats import PearsonRConstantInputWarning, pearsonr
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

from algorithm import mcc_like_geo_youden as mcc, multilabel_confusion_matrix
from util import zip_strict

if TYPE_CHECKING:
    from typing import Any, Collection, Dict, Iterable, Iterator, List, Optional, Tuple

T = TypeVar('T')
CTypeable = Union[
    np.bool_,
    np.byte, np.short, np.intc, np.int_, np.longlong,
    np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong,
    np.single, np.double, np.longdouble,
]
SCT = TypeVar('SCT', bound=CTypeable, covariant=True)
DTypeLike = Union[np.dtype[SCT], type[SCT]]
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float32]


MIN_LEN = 3
BATCH_SIZE = 64


class EvalPickle(TypedDict):
    label_count: int
    y_true: List[Optional[bool]]
    y_pred: List[Optional[bool]]
    y_u: List[Optional[float]]


class WorkerVars(TypedDict, total=False):
    y_true: IntArray
    y_preds: IntArray
    y_us: FloatArray
    numpy_err_def: List[Any]
    numpy_err_gy: List[Any]


class SharedArray(Generic[SCT]):
    __slots__ = ('data', 'dtype', 'shape')

    data: ctypes.Array[Any]
    dtype: np.dtype[SCT]
    shape: Tuple[int, ...]

    def __init__(self, dtype: np.dtype[SCT], shape: Tuple[int, ...]) -> None:
        # NB: would use as_ctypes_type but mypy seems confused by the overloads
        ctype = type(np.ctypeslib.as_ctypes(dtype.type()))
        self.data = RawArray(ctype, np.prod(shape).item())
        self.dtype = dtype
        self.shape = shape

    def numpy(self) -> NDArray[SCT]:
        # NB: memoryview is needed to convince mypy that data is bytes-like
        return np.frombuffer(memoryview(self.data), dtype=self.dtype).reshape(*self.shape)

    @classmethod
    def fromnumpy(cls, arr: NDArray[SCT]) -> SharedArray[SCT]:
        obj = cls(arr.dtype, arr.shape)
        np.copyto(obj.numpy(), arr, casting='no')
        return obj

    @classmethod
    def fromiter(cls, it: Iterable[Any], dtype: DTypeLike[SCT]) -> SharedArray[SCT]:
        return cls.fromnumpy(np.asarray(tuple(it), dtype=dtype))


SharedIntArray = SharedArray[np.int64]
SharedFloatArray = SharedArray[np.float32]


def corr(a: NDArray[np.generic], b: NDArray[np.generic]) -> np.float64:
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=PearsonRConstantInputWarning)  # Acceptable
        return pearsonr(a, b)[0]


def load(path: str) -> EvalPickle:
    with open(path, 'rb') as pf:
        return pickle.load(pf)


# Assumes caller ignores output for columns with any element > 1
@nb.njit(fastmath=True)  # type: ignore[misc]
def mode(pt: Tuple[IntArray, ...]) -> IntArray:
    p = np.stack(pt)
    assert p.ndim == 2
    # Sum along axis 0
    one_count = p.sum(axis=0)
    # Compute thresholds
    thresh = len(p) // 2
    # Apply thresholds
    return (one_count > thresh).astype(np.int64)  # type: ignore[return-value]


# By uncertainty sorting - throws out k elements per column
# Assumes caller ignores output for columns with any element > 1
@nb.njit(fastmath=True)  # type: ignore[misc]
def mode_uk_real(pt: Tuple[IntArray, ...], ut: Tuple[FloatArray, ...], k: int) -> IntArray:
    p = np.stack(pt)
    u = np.stack(ut)
    assert p.ndim == 2
    assert u.ndim == 2
    assert p.shape == u.shape
    assert 0 < k < p.shape[0]

    for _ in range(k):  # k iterations
        # 2D argmax(axis=0)
        maxu = np.empty(shape=(u.shape[1],), dtype=np.int64)
        for i in range(u.shape[1]):
            maxu[i] = u[:, i].argmax()
        # Exclude these elements from next argmax
        for i in range(len(maxu)):
            u[maxu[i], i] = -np.inf
        # Exclude the relevant members of p
        for i in range(len(maxu)):
            p[maxu[i], i] = 0

    # Sum along axis 0
    one_count = p.sum(axis=0)
    # Compute thresholds
    thresh = (len(p) - k) // 2
    # Apply thresholds
    return (one_count > thresh).astype(np.int64)  # type: ignore[return-value]


def mode_uk(p: Tuple[IntArray, ...], u: Tuple[FloatArray, ...], k: int) -> IntArray:
    return mode(p) if k == 0 else mode_uk_real(p, u, k)


# By uncertainty threshold - throws out elements below u threshold per column
# Gets weird if all elements in a column are below u threshold
# Assumes caller ignores output for columns with any element > 1
@nb.njit(fastmath=True)  # type: ignore[misc]
def mode_uthr(pt: Tuple[IntArray, ...], u: Tuple[FloatArray, ...], uthr: float) -> IntArray:
    p = np.stack(pt)
    assert p.ndim == 2
    assert u[0].ndim == 1
    assert p.shape[0] == len(u)
    assert p.shape[1] == u[0].shape[0]
    assert 0 < uthr < 1

    # Count exclusions along axis 0
    ex_count = np.zeros(shape=p.shape[1], dtype=np.int64)
    for i in range(len(p)):
        # Threshold uncertainty on uthr
        ex = u[i] > uthr
        # Exclude the respective elements of p
        p[i][ex] = 0
        ex_count += ex.astype(np.int64)

    # Sum along axis 0
    one_count = p.sum(axis=0)
    # Compute thresholds, minus excluded #
    thresh = np.full(shape=one_count.shape, fill_value=len(p) // 2)
    thresh -= ex_count
    # Apply thresholds
    return (one_count > thresh).astype(np.int64)  # type: ignore[return-value]


class AllOddCombinations:
    def __init__(self, seq: Collection[int], min_len: int = 1, max_len: Optional[int] = None, k: int = 0,
                 do_weights: bool = True) -> None:
        assert min_len >= 1 and min_len % 2 == 1
        if max_len is None:
            max_len = len(seq) - k
        else:
            assert min_len + k <= max_len + k + 1  # free zeros/ones -> minimum +1
            assert max_len + k <= len(seq)
        self._seq = seq
        self._min_len = min_len + k
        self._max_len = max_len + k
        self._k = k
        self._do_weights = do_weights

    def __len__(self) -> int:
        total = 0
        for combo in self._get_combos():
            uniq = len(combo)
            if (uniq - self._k) % 2 == 1:
                total += 1
            if self._do_weights:
                cslen = sum(1 for e in combo if e > 1)
                start = uniq + 1 - self._k
                total += cslen * len(range(
                    2 - start % 2,
                    uniq - 2,
                    2,
                ))
        return total

    def __iter__(self) -> Iterator[Tuple[int, ...]]:
        for combo in self._get_combos():
            uniq = len(combo)
            if (uniq - self._k) % 2 == 1:
                yield combo
            if not self._do_weights:
                continue
            for dup in itertools.count(1):
                if dup >= uniq - 2:
                    break  # Un-dupped can no longer overrule dupped
                if (uniq + dup - self._k) % 2 == 0:
                    continue  # Not odd
                for i, e in enumerate(combo):
                    if e <= 1:  # 0 and 1 are ZEROS and ONES
                        continue  # Weighting these would be pointless
                    yield (*combo[:i + 1], *(e for _ in range(dup)), *combo[i + 1:])

    def _get_combos(self) -> Iterator[Tuple[int, ...]]:
        it: Iterator[Tuple[int, ...]]
        it = itertools.chain.from_iterable(
            itertools.combinations(self._seq, i)
            for i in range(self._min_len, self._max_len + 1)
        )
        yield from (c for c in it if not (0 in c and 1 in c))
        # Allow zeros and ones for free
        it = itertools.combinations(self._seq, self._max_len + 1)
        yield from (c for c in it if (0 in c) != (1 in c))


def getscore(cpathi: int) -> Tuple[int, Tuple[float, ...]]:
    y_true = worker_vars['y_true']
    cpred = worker_vars['y_preds'][cpathi]
    used_indices = [i for i in range(cpred.shape[1]) if cpred[0, i] != 2]
    MCM = multilabel_confusion_matrix(y_true[:, used_indices], cpred[:, used_indices])
    try:
        np.seterrobj(worker_vars['numpy_err_gy'])
        assert len(used_indices) == len(MCM)
        scores = dict(zip(used_indices, map(mcc, MCM)))
    finally:
        np.seterrobj(worker_vars['numpy_err_def'])
    return cpathi, tuple(scores.get(i, 0) for i in range(cpred.shape[1]))


def getscore_combo(cpathis: Tuple[int, ...], k: int) -> Tuple[Tuple[int, ...], Tuple[float, ...]]:
    y_true = worker_vars['y_true']
    cpshape = worker_vars['y_preds'][cpathis[0]].shape
    cpreds = tuple(worker_vars['y_preds'][cp].reshape(-1) for cp in cpathis)
    cus = tuple(worker_vars['y_us'][cp].reshape(-1) for cp in cpathis)
    mpred = mode_uk(cpreds, cus, k).reshape(*cpshape)
    used_indices = [i for i in range(cpshape[1]) if not any(cp[i] == 2 for cp in cpreds)]
    if len(used_indices) < cpshape[1]:
        y_true = y_true[:, used_indices]
    MCM = multilabel_confusion_matrix(y_true, mpred[:, used_indices])
    try:
        np.seterrobj(worker_vars['numpy_err_gy'])
        assert len(used_indices) == len(MCM)
        scores = dict(zip(used_indices, map(mcc, MCM)))
    finally:
        np.seterrobj(worker_vars['numpy_err_def'])
    return cpathis, tuple(scores.get(i, 0) for i in range(mpred.shape[1]))


def getscore_combo_batch(cpathi_batch: Tuple[Tuple[int, ...], ...], k: int) \
        -> Tuple[Tuple[Tuple[int, ...], Tuple[float, ...]], ...]:
    return tuple(getscore_combo(cpathis, k) for cpathis in cpathi_batch)


def list2np(l: List[Optional[bool]], numlabels: int) -> IntArray:
    # 2 is a placeholder
    def noneis2(x: Optional[bool]) -> int:
        return 2 if x is None else int(x)

    arr = np.fromiter(map(noneis2, l), dtype=np.int64)
    return arr.reshape(-1, numlabels)


def list2npu(l: List[Optional[float]], numlabels: int) -> FloatArray:
    # 1 means maximum uncertainty, doesn't really matter because of exclusion
    def noneis1(x: Optional[float]) -> float:
        return 1 if x is None else x

    arr = np.fromiter(map(noneis1, l), dtype=np.float32)
    return arr.reshape(-1, numlabels)


def collate(x: Iterable[T], n: int) -> Iterator[Tuple[T, ...]]:
    it = iter(x)
    while True:
        batch = tuple(itertools.islice(it, n))
        if not batch:
            return
        yield batch


worker_vars: WorkerVars = {}


def init_worker(y_true: SharedIntArray, y_preds: SharedIntArray, y_us: SharedFloatArray) -> None:
    worker_vars['y_true'] = y_true.numpy()[:, list(range(y_true.shape[1]))]
    worker_vars['y_preds'] = y_preds.numpy()
    worker_vars['y_us'] = y_us.numpy()
    worker_vars['numpy_err_def'] = err = np.geterrobj()
    err = err.copy()
    err[1] &= ~(7 << np.SHIFT_INVALID)  # invalid='ignore'
    worker_vars['numpy_err_gy'] = err


if __name__ == '__main__':
    maxlen_str, k_str, do_weights_str, *cpaths = sys.argv[1:]
    maxlen, k, do_weights, = int(maxlen_str), int(k_str), bool(int(do_weights_str))
    pickles: Dict[str, EvalPickle] = {cpath: load(cpath) for cpath in cpaths}
    del cpaths
    assert len(pickles) >= MIN_LEN
    numlabels = next(iter(pickles.values()))['label_count']
    y_true_l = next(p['y_true'] for p in pickles.values() if not any(x is None for x in p['y_true']))
    assert all(
        p['label_count'] == numlabels
        and len(p['y_true']) == len(y_true_l)
        and all((x is None or x == y) for x, y in zip_strict(p['y_true'], y_true_l))
        for p in pickles.values()
    )
    assert all(len(p['y_pred']) == len(y_true_l) for p in pickles.values())
    y_true = SharedArray.fromnumpy(list2np(y_true_l, numlabels))
    cpaths = []
    y_preds_l = []
    y_us_l = []

    # Artificial y_preds for biasing
    cpaths.extend(('ZEROS', 'ONES'))
    y_preds_l.append(list2np([False for i, _ in enumerate(y_true_l)], numlabels))  # ZEROS
    y_preds_l.append(list2np([True  for i, _ in enumerate(y_true_l)], numlabels))  # ONES
    # Maximum certainty to make sure they have an effect
    for _ in ('ZEROS', 'ONES'):
        y_us_l.append(list2npu([0 for i, _ in enumerate(y_true_l)], numlabels))

    cpaths.extend(pickles)
    y_preds_l.extend(list2np(pkl['y_pred'], numlabels) for pkl in pickles.values())
    y_us_l.extend(list2npu(pkl['y_u'], numlabels) for pkl in pickles.values())

    y_preds = SharedArray.fromnumpy(np.stack(y_preds_l))
    y_us = SharedArray.fromnumpy(np.stack(y_us_l))
    del pickles, y_true_l, y_preds_l, y_us_l

    best_score: List[float] = [0. for _ in range(numlabels)]
    best_combo: List[Optional[Tuple[int, ...]]] = [None for _ in range(numlabels)]

    def submit(cpaths: Tuple[int, ...], score: float, lbl: int) -> None:
        if score > best_score[lbl]:
            best_score[lbl] = score
            best_combo[lbl] = cpaths

    with multiprocessing.Pool(initializer=init_worker, initargs=(y_true, y_preds, y_us)) as p:
        print('Trying single...')
        for cpathi, scores in p.imap_unordered(getscore, range(2, y_preds.shape[0])):
            for lbl, score in enumerate(scores):
                submit((cpathi,), score, lbl)

        print('Trying combos...')
        gscbk = functools.partial(getscore_combo_batch, k=k)
        it = AllOddCombinations(range(y_preds.shape[0]), min_len=MIN_LEN, max_len=maxlen, k=k, do_weights=do_weights)
        for batch in p.imap_unordered(gscbk, collate(tqdm(it, leave=False, smoothing=.05), BATCH_SIZE)):
            for cpathis, scores in batch:
                for lbl, score in enumerate(scores):
                    submit(cpathis, score, lbl)

    def get_y_pred(cpathi: int) -> IntArray:
        return y_preds.numpy()[cpathi]

    def get_y_u(cpathi: int) -> IntArray:
        return y_us.numpy()[cpathi]

    def lblscore(cpathi: int) -> Optional[float]:
        if cpaths[cpathi] in ('ZEROS', 'ONES'):
            return None  # Weird numerical results, skip it
        true = y_true.numpy()[:, lbl]
        cpred = get_y_pred(cpathi)[:, lbl]
        C = confusion_matrix(true, cpred)
        return mcc(C)

    def lblu(cpathis: Tuple[int, ...]) -> Optional[float]:
        if all(cpaths[i] in ('ZEROS', 'ONES') for i in cpathis):
            return None  # Value is artificial, skip it
        cus = [get_y_u(i)[:, lbl] for i in cpathis if cpaths[i] not in ('ZEROS', 'ONES')]
        return float(np.mean(cus))

    for lbl, (lbest_score, lbest_combo) in enumerate(zip_strict(best_score, best_combo)):
        assert lbest_combo is not None
        print('Best combination (label {}):'.format(lbl))
        print('  Length: {}'.format(len(lbest_combo)))
        print('  MCC: {}'.format(lbest_score))
        print('  Uncertainty: {}'.format(lblu(lbest_combo)))
        print('  Paths: {}'.format(tuple(map(cpaths.__getitem__, lbest_combo))))
        if len(lbest_combo) == 1:
            continue

        print('  Individual MCCs: {}'.format(tuple(map(lblscore, lbest_combo))))
        print('  Individual uncertainties: {}'.format(tuple(lblu((i,)) for i in lbest_combo)))
        print('  Correlations:')
        for a, b in itertools.combinations(range(len(lbest_combo)), 2):
            apred = get_y_pred(lbest_combo[a])[:, lbl]
            bpred = get_y_pred(lbest_combo[b])[:, lbl]
            print('    {} with {}: {}'.format(a, b, corr(apred, bpred)))
