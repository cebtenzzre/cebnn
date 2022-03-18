# -*- coding: utf-8 -*-

from __future__ import annotations

from itertools import chain, islice
from typing import TYPE_CHECKING

import numpy as np
from numpy.random import RandomState
from sklearn.model_selection._split import StratifiedGroupKFold, _BaseKFold
from skmultilearn.model_selection import IterativeStratification
from torch.utils.data.sampler import Sampler

from algorithm import ml_ros, ml_rus, mlenn, mltl

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union
    from util import Array
    Algorithm = Tuple[bool, Callable[..., List[int]]]


class ImbalancedDatasetSampler(Sampler[int]):
    """
    Samples elements randomly from a given list of indices for imbalanced dataset.

    Arguments:
        indices (list, optional): a list of indices
        num_samples (int, optional): number of samples to draw
        callback_get_label func: a callback which takes two arguments - dataset and index
    """

    supported_algorithms: Dict[str, Algorithm] = {
        'MLeNN':  (True,  mlenn),
        'MLTL':   (True,  mltl),
        'ML-ROS': (False, ml_ros),
        'ML-RUS': (False, ml_rus),
    }

    def __init__(self, dataset: Sequence[Set[str]], ds_labels: Optional[Iterable[str]],
                 algorithm: Union[str, Algorithm], alg_kwargs: Optional[Dict[str, Any]] = None,
                 num_samples: Optional[int] = None, rand: RandomState = np.random.mtrand._rand):
        super().__init__(None)

        # Compute ds_labels if not given
        self.ds_labels = ds_labels if ds_labels is not None \
            else tuple(sorted(frozenset(chain.from_iterable(dataset))))

        alg = self.supported_algorithms.get(algorithm) # type: ignore
        if alg is None:
            if not (isinstance(algorithm, tuple) and len(algorithm) == 2
                    and isinstance(algorithm[0], bool) and callable(algorithm[1])):
                raise ValueError('algorithm must be a valid key or a (bool, callable) tuple')
            alg = algorithm
        self.alg_is_deterministic, self.alg = alg
        self.alg_kwargs = {} if alg_kwargs is None else alg_kwargs

        # Default num_samples is len(dataset)
        if num_samples is not None and (not isinstance(num_samples, int) or num_samples <= 0):
            raise ValueError('num_samples must be a positive integer')
        self.num_samples = num_samples

        self.offset = 0
        self.rand = rand
        self._indices: List[Any]
        if self.alg_is_deterministic:
            # _indices has type List[int]
            self._indices = self.alg(dataset, ds_labels=self.ds_labels, rand=self.rand, **self.alg_kwargs)
        else:
            # _indices has type List[Iterable[int]]
            self._indices = []
            self.dataset = dataset
            self._get_more_nondet()

    def __iter__(self) -> Iterator[int]:
        if self.alg_is_deterministic:
            yield from self._det_iter()
            return
        if self.num_samples is None:
            assert not self.offset
            assert len(self._indices) < 2
            if not self._indices:
                self._get_more_nondet()
            yield from self._indices.pop(0)
            return

        yieldable = sum(len(l) for l in self._indices)
        while yieldable < self.offset + self.num_samples:
            yieldable += self._get_more_nondet()

        ret = islice(chain(*self._indices),
                     self.offset, self.offset + self.num_samples)

        self.offset += self.num_samples
        while self._indices and self.offset >= len(self._indices[0]):
            removed = self._indices.pop(0)
            assert removed
            self.offset -= len(removed)

        yield from ret

    def __len__(self) -> int:
        if self.num_samples is None:
            raise TypeError('Sampler has unknown length')
        return self.num_samples

    def _get_more_nondet(self) -> int:
        assert not self.alg_is_deterministic
        new_sds = self.alg(self.dataset, ds_labels=self.ds_labels, rand=self.rand, **self.alg_kwargs)
        assert new_sds
        self.rand.shuffle(new_sds)
        self._indices.append(new_sds)
        return len(new_sds)

    def _det_iter(self) -> Iterator[int]:
        if self.num_samples is None:
            assert not self.offset
            yield from self._indices
            return

        num_yielded, needed = 0, self.num_samples + self.offset
        while num_yielded < needed:
            to_yield = self._indices[self.offset:][:needed - num_yielded]
            yield from to_yield
            num_yielded += len(to_yield)
            self.offset = (self.offset + num_yielded) % len(self._indices)


def MLStratifiedGroupKFold(n_labels: int, n_splits: int = 3, \
                           random_state: Optional[Union[int, RandomState]] = None,
                           fold_ratios: Optional[Sequence[float]] = None) -> _BaseKFold:
    if n_labels > 1:
        if False:
            return IterativeStratification(n_splits=n_splits, order=n_labels, random_state=random_state,
                                           sample_distribution_per_fold=fold_ratios)
        raise NotImplementedError("I haven't decided on a good way to do this yet")
    return MyStratifiedGroupKFold(n_splits=n_splits, fold_ratios=fold_ratios)


class MyStratifiedGroupKFold(StratifiedGroupKFold):
    def __init__(self, n_splits: int = 5, shuffle: bool = False, random_state: Optional[Union[int, RandomState]] = None,
                 fold_ratios: Optional[Sequence[float]] = None) -> None:
        super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
        if fold_ratios is None:
            self.ratio_per_fold = np.array([1 for _ in range(self.n_splits)])
        else:
            if len(fold_ratios) != self.n_splits:
                raise ValueError('fold_ratios must have length {}'.format(self.n_splits))
            self.ratio_per_fold = np.asarray(fold_ratios)

    def _find_best_fold(self, y_counts_per_fold: Array, y_cnt: Array, group_y_counts: Array) -> int:
        best_fold = None
        min_eval = np.inf
        min_samples_in_fold = np.inf
        for i in range(self.n_splits):
            y_counts_per_fold[i] += group_y_counts
            # Summarise the distribution over classes in each proposed fold
            std_per_class = np.std(
                y_counts_per_fold / self.ratio_per_fold.reshape(-1, 1)
                                  / y_cnt.reshape(1, -1),
                axis=0)
            y_counts_per_fold[i] -= group_y_counts
            fold_eval = np.mean(std_per_class)
            samples_in_fold = np.sum(y_counts_per_fold[i]) / self.ratio_per_fold[i]
            is_current_fold_better = (
                fold_eval < min_eval or
                np.isclose(fold_eval, min_eval) and samples_in_fold < min_samples_in_fold
            )
            if is_current_fold_better:
                min_eval = fold_eval
                min_samples_in_fold = samples_in_fold
                best_fold = i
        assert best_fold is not None
        return best_fold
