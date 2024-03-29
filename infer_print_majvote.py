#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

from __future__ import annotations

import csv
import itertools
import pickle
import sys
from typing import TYPE_CHECKING, Sequence

import numba as nb
import numpy as np

from best_majvote import FloatArray, IntArray, mode_uk
from infer_print import Preds
from util import zip_strict

if TYPE_CHECKING:
    from typing import Any, List, Optional, Tuple, Union

MAJVOTE_BLOCKSIZE = 16 * 1024  # Tradeoff between peak memory usage and performance


def load(path: str) -> Any:
    with open(path, 'rb') as pf:
        return pickle.load(pf)


# Mean uncertainty of all predictors that agree with the consensus.
@nb.njit(fastmath=True, error_model='numpy')  # type: ignore[misc]
def pos_umean(c: IntArray, pt: Tuple[IntArray, ...], ut: Tuple[FloatArray, ...]) -> IntArray:
    p = np.stack(pt)
    u = np.stack(ut)
    assert c.ndim == 1
    assert p.ndim == 2
    assert u.ndim == 2
    assert c.shape == p.shape[1:]
    assert p.shape == u.shape

    # Count included uncertainties along axis 0
    in_count = np.zeros(shape=p.shape[1], dtype=np.int64)
    for i in range(len(p)):
        agree = (p[i] == c)
        # Exclude the respective elements of u
        u[i][~agree] = 0
        # Add each sample's predictor count
        in_count += agree.astype(np.int64)

    # Sum along axis 0
    u_total = u.sum(axis=0)
    assert np.all(in_count)  # At least one predictor should agree
    # Divide to compute partial mean
    return u_total / in_count


if __name__ == '__main__':
    if len(sys.argv) < 6:
        raise ValueError('Expected at least 5 arguments, got {}'.format(len(sys.argv) - 1))

    # Predictions files generated by infer.py
    data_cnames_str, k_str, lblidx_str, *pred_cp_paths = sys.argv[1:]
    data_cnames, k, lblidx = data_cnames_str.split(','), int(k_str), int(lblidx_str)
    label = data_cnames[lblidx]
    del data_cnames_str, lblidx_str
    if len(pred_cp_paths) % 2 != 0:
        raise ValueError('Expected one cpickle per predictions file')

    sample_paths: Optional[Sequence[str]] = None
    preds_list: List[Union[str, IntArray]] = []
    uncertainties_list: List[Optional[FloatArray]] = []
    pred_cp_paths_it = iter(pred_cp_paths)
    for predfile_path, cpickle_path in zip_strict(pred_cp_paths_it, pred_cp_paths_it):
        if predfile_path in ('ZEROS', 'ONES'):
            # Dealt with later
            preds_list.append(predfile_path)
            uncertainties_list.append(None)
            continue

        thresholds: Sequence[float] = load(cpickle_path)['thresh']
        preds = Preds.load(predfile_path)
        if sample_paths is None:
            sample_paths = preds.sample_paths
        else:
            assert sample_paths == preds.sample_paths  # All predfiles should be for the same samples

        preds_list.append((preds.lbl_preds(label) > thresholds[lblidx]).astype(np.int64))
        uncertainties_list.append(preds.lbl_uncertainties(label))

    assert sample_paths is not None

    # Fill in 'ZEROS' and 'ONES' placeholders
    any_pred: IntArray = next(p for p in preds_list if isinstance(p, np.ndarray))
    y_preds_list: List[IntArray] = []
    y_us_list: List[FloatArray] = []
    for yp, ypu in zip_strict(preds_list, uncertainties_list):
        if not isinstance(yp, str):
            assert ypu is not None
            y_preds_list.append(yp)
            y_us_list.append(ypu)
            continue
        if yp == 'ZEROS':
            ypa = np.zeros_like(any_pred)
        elif yp == 'ONES':
            ypa = np.ones_like(any_pred)
        else:
            raise AssertionError
        y_preds_list.append(ypa)
        y_us_list.append(np.zeros(shape=any_pred.shape, dtype=np.float32))
    del any_pred, preds_list, uncertainties_list
    y_preds, y_us = np.stack(y_preds_list), np.stack(y_us_list)
    del y_preds_list, y_us_list

    mean_u: FloatArray = np.mean(y_us, axis=0)  # type: ignore[assignment]

    # Majority vote, in blocks
    mpred_blocks = []
    mu_blocks = []
    for bi in itertools.count():
        bs = MAJVOTE_BLOCKSIZE
        preds_block = y_preds[:, bs * bi:bs * (bi + 1)]
        us_block = y_us[:, bs * bi:bs * (bi + 1)]
        if not preds_block.shape[1]:
            assert not us_block.shape[1]
            break  # Out of data
        assert us_block.shape[1]
        pt, ut = tuple(preds_block), tuple(us_block)
        mpred_block = mode_uk(pt, ut, k)
        mpred_blocks.append(mpred_block)
        mu_blocks.append(pos_umean(mpred_block, pt, ut))
    del y_preds, y_us
    mpred = np.concatenate(mpred_blocks)
    mu = np.concatenate(mu_blocks)
    del mpred_blocks, mu_blocks

    writer = csv.writer(sys.stdout)
    writer.writerow(['sample_path', 'total_uncertainty', 'pos_uncertainty', 'label'])

    for fname, mpred, mu, u in zip_strict(sample_paths, mpred, mu, mean_u):
        writer.writerow([fname, u, mu, label if mpred > .99 else ''])
