# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from itertools import chain
from math import isclose, sqrt
from typing import TYPE_CHECKING

import numba as nb
import numpy as np
from sklearn.neighbors import BallTree

if TYPE_CHECKING:
    from typing import Any, Callable, Iterable, List, Optional, Sequence, Set, Tuple, TypeVar
    from util import Array
    Dataset = Sequence[Set[str]]
    T = TypeVar('T')


@dataclass(init=False)
class LabelStats:
    occurrences: int
    imbalance: float

    def __init__(self, occurrences: Optional[int] = None,
                 imbalance: Optional[float] = None) -> None:
        if occurrences is not None:
            self.occurrences = occurrences
        if imbalance is not None:
            self.imbalance = imbalance


def nearest_neighbors(ball_tree: BallTree, samples: Array, count: int) -> Sequence[int]:
    assert samples.shape[0] == 1
    return ball_tree.query(samples, k=count, return_distance=False)[0]


def adjusted_hamming_dist(sample: Array, neighbor: Array) -> float:
    assert sample.ndim == 1 and neighbor.ndim == 1
    assert sample.shape == neighbor.shape
    dist: int = np.count_nonzero(sample != neighbor)
    active: int = np.count_nonzero(sample | neighbor)
    return dist / active if active != 0 else 0.


@dataclass(frozen=True)
class OptionalLabel:
    label: Optional[str] = None

    def matches(self, lset: Set[str]) -> bool:
        return (not lset) if self.label is None else (self.label in lset)


class DatasetStats:
    dataset: Dataset
    labels: OrderedDict[OptionalLabel, LabelStats]
    mean_imbalance: float
    mean_pos_imbalance: float
    _max_occurrences: int

    def __init__(self, dataset: Dataset, ds_labels: Optional[Iterable[Optional[str]]] = None, mode: str = 'positive') \
            -> None:
        if mode not in ('positive', 'all'):
            raise ValueError('Unknown mode: {}'.format(mode))

        self.dataset = dataset
        self.mode = mode

        if ds_labels is None:
            ds_labels = sorted(set(chain.from_iterable(dataset)))

        def make_label(ol: OptionalLabel) -> Tuple[OptionalLabel, LabelStats]:
            return (ol, LabelStats(self._total_occurrences(ol)))

        self.labels = OrderedDict(make_label(OptionalLabel(l))
                                  for l in chain(ds_labels, (None,)))

        self._compute_max_and_imbalance()

    def clone_sample(self, sample: Set[str]) -> None:
        self._augment_sample(sample, 1)

    def remove_sample(self, sample: Set[str]) -> None:
        self._augment_sample(sample, -1)

    def _augment_sample(self, sample: Set[str], n: int) -> None:
        matching = {l for l, ls in self.labels.items() if l.matches(sample)}
        for label, lstats in self.labels.items():
            if label in matching:
                lstats.occurrences = max(0, lstats.occurrences + n)
        self._compute_max_and_imbalance()

    def _total_occurrences(self, label: OptionalLabel) -> int:
        return sum(1 for s in self.dataset if label.matches(s))

    def _compute_max_and_imbalance(self) -> None:
        self._max_occurrences = max(ls.occurrences for ls in self.labels.values())

        for lstats in self.labels.values():
            lstats.imbalance = self._calc_imbalance(lstats)

        assert len(self.labels) >= 2  # O/w it's not even binary
        pos_imbs = [ls.imbalance for l, ls in self.labels.items() if l.label is not None]
        self.mean_pos_imbalance = np.mean(pos_imbs).item()
        self.mean_imbalance = self.mean_pos_imbalance if self.mode == 'positive' \
            else np.mean([ls.imbalance for l, ls in self.labels.items()]).item()

    def _calc_imbalance(self, lstats: LabelStats) -> float:
        if not lstats.occurrences:
            return np.inf
        return self._max_occurrences / lstats.occurrences


# http://doi.org/10.1007/978-3-319-10840-7_1
def mlenn(dataset: Dataset, threshold: float = .75, num_neighbors: int = 3,
          ds_labels: Optional[Iterable[str]] = None, **kwargs: object) -> List[int]:  # noqa: U100
    dstats = DatasetStats(dataset, ds_labels)
    del ds_labels

    points: Array = np.empty((len(dataset), len(dstats.labels)), dtype=bool)
    for i, sample in enumerate(dataset):
        for j, label in enumerate(dstats.labels):
            points[i][j] = label.matches(sample)

    ball_tree = BallTree(points, metric='hamming')

    def keep_sample(sample_idx: int) -> bool:
        for l, ls in dstats.labels.items():
            if not l.matches(dataset[sample_idx]):
                continue  # Label not applicable
            if ls.imbalance > dstats.mean_imbalance:
                return True  # Preserve instance with minority labels

        num_differences: int = sum(
            1 for n in nearest_neighbors(ball_tree, points[sample_idx, np.newaxis], num_neighbors)
            if adjusted_hamming_dist(points[sample_idx], points[n]) > threshold)

        return float(num_differences) < num_neighbors / 2

    return [i for i in range(len(dataset)) if keep_sample(i)]


# http://doi.org/10.1016/j.neucom.2019.11.076
def mltl(dataset: Dataset, threshold_in: Optional[float] = None,
         ds_labels: Optional[Iterable[str]] = None, **kwargs: object) -> List[int]:  # noqa: U100
    dstats = DatasetStats(dataset, ds_labels)
    del ds_labels

    threshold: float
    if threshold_in is not None:
        threshold = threshold_in
    else:
        # Choose a normalized threshold automatically
        imbalanceness: float = 1 / sqrt(dstats.mean_imbalance)
        if imbalanceness >= 0.5:
            threshold = 0.5
        elif imbalanceness >= 0.3:
            threshold = 0.3
        else:
            threshold = 0.15

    points: Array = np.empty((len(dataset), len(dstats.labels)), dtype=bool)
    for i, sample in enumerate(dataset):
        for j, label in enumerate(dstats.labels):
            points[i][j] = label.matches(sample)

    ball_tree = BallTree(points, metric='hamming')

    majority_bags: List[List[int]] = [
        [i for i, s in enumerate(dataset) if l.matches(s)]
        for l, ls in dstats.labels.items()
        if ls.imbalance > dstats.mean_imbalance]

    checked_samples: Set[int] = set()
    samples_to_delete: List[int] = []

    for bag in majority_bags:
        for sample_idx in bag:
            if sample_idx in checked_samples:
                continue
            checked_samples.add(sample_idx)
            neighbor = nearest_neighbors(ball_tree, points[sample_idx, np.newaxis], count=1)[0]
            if adjusted_hamming_dist(points[sample_idx], points[neighbor]) >= threshold:
                samples_to_delete.append(sample_idx)

    return [i for i in range(len(dataset)) if i not in samples_to_delete]


@dataclass(frozen=True)
class Bag:
    label: OptionalLabel
    bag: List[int]
    lstats: LabelStats


def print_stats(dstats: DatasetStats, bags: Iterable[Bag]) -> None:
    print('  Mean: {:.4f}'.format(dstats.mean_imbalance))
    bag_labels = {bag.label for bag in bags}
    for label, lstats in dstats.labels.items():
        print('  {}: {:.4f}{}'.format(label.label, lstats.imbalance, ' minority' if label in bag_labels else ''))


def gr_or_close(a: float, b: float) -> bool:
    return a > b or isclose(a, b)


# http://doi.org/10.1016/j.neucom.2014.08.091
def ml_ros(dataset: Dataset, rand: np.random.RandomState, resample_limit: Optional[float] = None,
           imbalance_target: Optional[float] = None, ds_labels: Optional[Iterable[str]] = None,
           mode: str = 'positive') -> List[int]:
    if resample_limit is None:
        samples_to_clone = None
    else:
        samples_to_clone = round(len(dataset) * resample_limit)
        if not samples_to_clone:
            return list(range(len(dataset)))

    dstats = DatasetStats(dataset, ds_labels, mode)
    del ds_labels

    # List of pairs of (samples with label, label imbalance)
    minority_bags: List[Bag] = [
        Bag(l, [i for i, s in enumerate(dataset) if l.matches(s)], ls)
        for l, ls in dstats.labels.items()
        if not isclose(ls.imbalance, 1)]
    assert minority_bags

    print('==> Starting ML-ROS.')
    if imbalance_target is not None:
        print('  Target mean +imbalance: {}'.format(imbalance_target))
    if resample_limit is not None:
        print('  Clone limit: {:.2f}%'.format(resample_limit * 100))
    print_stats(dstats, minority_bags)

    new_samples: List[int] = list(range(len(dataset)))
    samples_cloned = 0

    while True:  # Instances cloning loop
        if imbalance_target is not None and gr_or_close(imbalance_target, dstats.mean_pos_imbalance):
            print('==> Completed ML-ROS (reached imbalance target).')
            break
        if samples_to_clone is not None and samples_cloned >= samples_to_clone:
            print('==> Completed ML-ROS (reached clone limit).')
            break
        last_cloned = samples_cloned
        # Clone a random sample from each minority bag
        for mbag in minority_bags:
            if not gr_or_close(mbag.lstats.imbalance, dstats.mean_imbalance):
                continue  # Skip this bag for now
            sample_idx: int = rand.choice(mbag.bag)
            new_samples.append(sample_idx)
            dstats.clone_sample(dataset[sample_idx])
            samples_cloned += 1
        if samples_cloned == last_cloned:
            print('==> Completed ML-ROS (ran out of samples).')
            break

    print('  Cloned {}% of the dataset -> +imbalance={:.2f}'.format(
        round(100 * samples_cloned / len(dataset)), dstats.mean_pos_imbalance))
    print_stats(dstats, minority_bags)
    return new_samples


# http://doi.org/10.1016/j.neucom.2014.08.091
def ml_rus(dataset: Dataset, rand: np.random.RandomState, resample_limit: Optional[float] = None,
           imbalance_target: Optional[float] = None, ds_labels: Optional[Iterable[str]] = None,
           mode: str = 'positive') -> List[int]:
    if resample_limit is None:
        samples_to_remove = None
    else:
        samples_to_remove = round(len(dataset) * resample_limit)
        if not samples_to_remove:
            return list(range(len(dataset)))

    dstats = DatasetStats(dataset, ds_labels, mode)
    del ds_labels

    # List of pairs of (samples with label, label imbalance)
    minority_bags: List[Bag] = [
        Bag(l, [i for i, s in enumerate(dataset) if l.matches(s)], ls)
        for l, ls in dstats.labels.items()
        if gr_or_close(ls.imbalance, dstats.mean_imbalance)]
    assert minority_bags

    print('==> Starting ML-RUS.')
    if imbalance_target is not None:
        print('  Target mean +imbalance: {}'.format(imbalance_target))
    if resample_limit is not None:
        print('  Removal limit: {:.2f}%'.format(resample_limit * 100))
    print_stats(dstats, minority_bags)

    mbag_samples: Set[int] = {i for mb in minority_bags for i in mb.bag}
    removable_samples: List[int] = [i for i, _ in enumerate(dataset) if i not in mbag_samples]
    removed_samples: Set[int] = set()

    while True:  # Instances removing loop
        if imbalance_target is not None and gr_or_close(imbalance_target, dstats.mean_pos_imbalance):
            print('==> Completed ML-RUS (reached imbalance target).')
            break
        if samples_to_remove is not None and len(removed_samples) >= samples_to_remove:
            print('==> Completed ML-RUS (reached removal limit).')
            break
        if not removable_samples:
            # No progress, abort
            print('==> Completed ML-RUS (ran out of samples).')
            break
        # Remove a random sample from the available bags
        sample_idx_idx: int = rand.randint(len(removable_samples))
        sample_idx: int = removable_samples[sample_idx_idx]
        del removable_samples[sample_idx_idx]
        removed_samples.add(sample_idx)
        dstats.remove_sample(dataset[sample_idx])

        mb_remove = [i for i, mb in enumerate(minority_bags)
                     if not gr_or_close(mb.lstats.imbalance, dstats.mean_imbalance)]
        if mb_remove:
            mb_remove_samples = {i for mi in mb_remove for i in minority_bags[mi].bag}
            removable_samples = [i for i in removable_samples if i not in mb_remove_samples]
            minority_bags = [mb for i, mb in enumerate(minority_bags) if i not in mb_remove]

    print('  Removed {}% of the dataset -> +imbalance={:.2f}'.format(
        round(100 * len(removed_samples) / len(dataset)), dstats.mean_pos_imbalance))
    print_stats(dstats, minority_bags)
    return [i for i in range(len(dataset)) if i not in removed_samples]


@nb.jit(forceobj=True)  # type: ignore[misc]
def multilabel_confusion_matrix(y_true: Array, y_pred: Array, binary: bool = False) -> Array:
    true_and_pred = np.multiply(y_true, y_pred)
    tp_sum = np.count_nonzero(true_and_pred, axis=0)
    pred_sum = np.count_nonzero(y_pred, axis=0)
    true_sum = np.count_nonzero(y_true, axis=0)

    fp = pred_sum - tp_sum
    fn = true_sum - tp_sum
    tp = tp_sum
    tn = y_true.shape[0] - tp - fp - fn

    MCM = np.array([tn, fp, fn, tp]).T.reshape(-1, 2, 2)
    if binary and len(MCM) == 1:
        return np.concatenate((np.flip(MCM), MCM))  # Like the sklearn version
    return MCM


# Somewhere between Youden's J statistic and the Fowlkes-Mallows index.
# Actually a geometric mean of sensitivity, specificity, and ppv.
# This version works best for threshold search.
@nb.jit(forceobj=True)  # type: ignore[misc]
def geo_youden1_inner(C: Array) -> np.float32:
    sensitivity = C[1, 1] / (C[1, 1] + C[1, 0])  # a.k.a. recall
    specificity = C[0, 0] / (C[0, 0] + C[0, 1])
    ppv         = C[1, 1] / (C[1, 1] + C[0, 1])  # a.k.a. precision
    return np.power(sensitivity * specificity * ppv, 1/3)


@nb.jit(forceobj=True)  # type: ignore[misc]
def geo_youden1(C: Array) -> np.float32:
    return geo_youden1_inner(C)


# Based on an attempt to fix some MCM confusion. Not great.
@nb.jit(forceobj=True)  # type: ignore[misc]
def geo_youden1_fix(C: Array) -> np.number[Any]:
    return np.mean((
        geo_youden1_inner(C),
        geo_youden1_inner(np.flip(C)),
    ))


# Somewhere between Youden's J statistic and the Fowlkes-Mallows index.
# Actually a geometric mean of sensitivity, specificity, and (ppv+npv)/2.
# This version works best for majority voting.
@nb.jit(forceobj=True)  # type: ignore[misc]
def geo_youden2(C: Array) -> np.float32:
    sensitivity = C[1, 1] / (C[1, 1] + C[1, 0])  # a.k.a. recall
    specificity = C[0, 0] / (C[0, 0] + C[0, 1])
    ppv         = C[1, 1] / (C[1, 1] + C[0, 1])  # a.k.a. precision
    npv         = C[0, 0] / (C[0, 0] + C[1, 0])
    return np.power(sensitivity * specificity * np.mean((ppv, npv)), 1/3)


@nb.jit(forceobj=True)  # type: ignore[misc]
def geo_youden3(C: Array) -> np.float32:
    sensitivity = C[1, 1] / (C[1, 1] + C[1, 0])  # a.k.a. recall
    specificity = C[0, 0] / (C[0, 0] + C[0, 1])
    precision   = C[1, 1] / (C[1, 1] + C[0, 1])

    beta = 1.5
    beta2 = beta ** 2
    recall = np.sqrt(sensitivity * specificity)  # ''recall''
    return ((1 + beta2) * precision * recall) / (beta2 * precision + recall)


@nb.jit(forceobj=True)  # type: ignore[misc]
def diag_odds_ratio1(C: Array) -> np.float32:
    tpr = C[1, 1] / (C[1, 1] + C[1, 0])
    tnr = C[0, 0] / (C[0, 0] + C[0, 1])
    fpr = C[0, 1] / (C[0, 1] + C[0, 0])
    fnr = C[1, 0] / (C[1, 0] + C[1, 1])
    return (tpr * tnr) / (1 + fpr * fnr)


@nb.jit(forceobj=True)  # type: ignore[misc]
def diag_odds_ratio2(C: Array) -> np.float32:
    tpr = C[1, 1] / (C[1, 1] + C[1, 0])
    tnr = C[0, 0] / (C[0, 0] + C[0, 1])
    fpr = C[0, 1] / (C[0, 1] + C[0, 0])
    fnr = C[1, 0] / (C[1, 0] + C[1, 1])
    return (1 + tpr * tnr) / (2 * (1 + fpr * fnr))


@nb.njit(fastmath=True, error_model='numpy')  # type: ignore[misc]
def mcc_like_geo_youden(C: Array) -> np.float32:
    C = C.astype(np.float32)  # For JIT reasons
    t_sum = C.sum(axis=1)
    p_sum = C.sum(axis=0)
    n_correct = np.trace(C)
    n_samples = p_sum.sum()
    cov_ytyp = n_correct * n_samples - np.dot(t_sum, p_sum)
    cov_ypyp = n_samples ** 2 - np.dot(p_sum, p_sum)
    cov_ytyt = n_samples ** 2 - np.dot(t_sum, t_sum)
    mcc = cov_ytyp / np.sqrt(cov_ytyt * cov_ypyp)
    return 0. if np.isnan(mcc) else mcc


@nb.njit(fastmath=True, error_model='numpy')  # type: ignore[misc]
def fbeta_like_geo_youden(C: Array) -> np.float32:
    beta = 1
    beta2 = beta ** 2
    precision = C[1, 1] / (C[1, 1] + C[0, 1])
    recall    = C[1, 1] / (C[1, 1] + C[1, 0])
    return ((1 + beta2) * precision * recall) / (beta2 * precision + recall)


def balanced(f: Callable[[Array], T]) -> Callable[[Array], T]:
    @nb.jit(forceobj=True)  # type: ignore[misc]
    def inner(C: Array) -> T:  # pytype: disable=invalid-annotation
        numer = np.sum(C) / 2
        C = np.stack((C[0] * numer / np.sum(C[0]),
                      C[1] * numer / np.sum(C[1])))
        return f(C)
    return inner


geo_youden1_bal = balanced(geo_youden1)
geo_youden2_bal = balanced(geo_youden2)
geo_youden3_bal = balanced(geo_youden3)
diag_odds_ratio1_bal = balanced(diag_odds_ratio1)
diag_odds_ratio2_bal = balanced(diag_odds_ratio2)
mcc_like_geo_youden_bal = balanced(mcc_like_geo_youden)
fbeta_like_geo_youden_bal = balanced(fbeta_like_geo_youden)
