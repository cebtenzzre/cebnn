#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import sys
from typing import TYPE_CHECKING, Union

import numpy as np
from sklearn.metrics import confusion_matrix

from algorithm import mcc_like_geo_youden as mcc
from best_majvote import FloatArray, IntArray, corr, list2np, list2npu, load, mode_uk
from util import zip_strict, zipstar_strict

if TYPE_CHECKING:
    from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

MIN_LEN = 3


def getscore_combo(y_true: IntArray, y_preds: Tuple[IntArray, ...], y_us: Tuple[FloatArray, ...], k: int) -> float:
    mpred = mode_uk(y_preds, y_us, k)
    C = confusion_matrix(y_true, mpred)
    return mcc(C)


if __name__ == '__main__':
    cpaths: Sequence[str]
    k_str, numlabels_str, lblidx_str, *cpaths = sys.argv[1:]
    k, numlabels, lblidx = int(k_str), int(numlabels_str), int(lblidx_str)
    cpaths = tuple(cpaths)
    if len(cpaths) < MIN_LEN:
        raise ValueError('Expected at least {} cpaths, got {}'.format(MIN_LEN, len(cpaths)))
    if (len(cpaths) - k) % 2 == 0:
        raise ValueError('Expected odd number of cpaths, got {}'.format(len(cpaths)))

    y_true_l: Optional[List[Optional[bool]]] = None
    if TYPE_CHECKING:
        Pred = Union[str, IntArray]
        U = Optional[FloatArray]
    preds_list: List[Pred] = []
    uncertainties_list: List[U] = []
    for cpickle_path in cpaths:
        if cpickle_path in ('ZEROS', 'ONES'):
            # Dealt with later
            preds_list.append(cpickle_path)
            uncertainties_list.append(None)
            continue
        pkl = load(cpickle_path)
        pkl_true, pkl_preds, pkl_uncertainties = pkl['y_true'], pkl['y_pred'], pkl['y_u']
        if y_true_l is None:
            y_true_l = pkl_true
        else:
            assert len(pkl_true) == len(y_true_l)
            assert all((x is None or x == y) for x, y in zip_strict(pkl_true, y_true_l))
        assert len(pkl_preds) == len(pkl_uncertainties) == len(y_true_l)
        preds_list.append(list2np(pkl_preds, numlabels)[:, lblidx])
        uncertainties_list.append(list2npu(pkl_uncertainties, numlabels)[:, lblidx])

    assert y_true_l is not None
    y_true = list2np(y_true_l, numlabels)[:, lblidx]
    del y_true_l

    # Fill in 'ZEROS' and 'ONES' placeholders
    def gen(any_pred: IntArray, preds_list: Iterable[Pred], uncertainties_list: Iterable[U]) \
            -> Iterator[Tuple[IntArray, FloatArray]]:
        for yp, ypu in zip_strict(preds_list, uncertainties_list):
            if not isinstance(yp, str):
                assert ypu is not None
                yield yp, ypu
                continue
            if yp == 'ZEROS':
                ypa = np.zeros_like(any_pred)
            elif yp == 'ONES':
                ypa = np.ones_like(any_pred)
            else:
                raise AssertionError
            yield ypa, np.zeros(shape=any_pred.shape, dtype=np.float32)
    any_pred: IntArray = next(p for p in preds_list if isinstance(p, np.ndarray))
    y_preds, y_us = zipstar_strict(gen(any_pred, preds_list, uncertainties_list))
    del any_pred, preds_list, uncertainties_list

    score = getscore_combo(y_true, y_preds, y_us, k=k)

    def lblscore(cpathi: int) -> Optional[float]:
        if cpaths[cpathi] in ('ZEROS', 'ONES'):
            return None  # Weird numerical results, skip it
        cpred = y_preds[cpathi]
        C = confusion_matrix(y_true, cpred)
        return mcc(C)

    def lblu(cpathis: Tuple[int, ...]) -> Optional[float]:
        if all(cpaths[i] in ('ZEROS', 'ONES') for i in cpathis):
            return None  # Value is artificial, skip it
        cus = [y_us[i] for i in cpathis if cpaths[i] not in ('ZEROS', 'ONES')]
        return float(np.mean(cus))  # type: ignore[arg-type]

    combo = tuple(range(len(cpaths)))
    print('Given combination (label {}):'.format(lblidx))
    print('  Length: {}'.format(len(cpaths)))
    print('  MCC: {}'.format(score))
    print('  Uncertainty: {}'.format(lblu(combo)))
    print('  Paths: {}'.format(cpaths))
    print('  Individual MCCs: {}'.format(tuple(map(lblscore, combo))))
    print('  Individual uncertainties: {}'.format(tuple(lblu((cp,)) for cp in combo)))
    print('  Correlations:')
    for a, b in itertools.combinations(range(len(cpaths)), 2):
        apred = y_preds[a]
        bpred = y_preds[b]
        print('    {} with {}: {}'.format(a, b, corr(apred, bpred)))
