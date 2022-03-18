#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
from functools import partial
from typing import TYPE_CHECKING, Callable, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple, Union

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

from datamunge import MLStratifiedGroupKFold
from util import Array, zip_strict

if TYPE_CHECKING:
    StrPath = Union[str, 'os.PathLike[str]']

CLASS_DIR = 'class'
WEIGHT_DIR = 'weight'


def seed_all(seed: int) -> None:
    # See https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)  # pytype: disable=module-attr
    os.environ['PYTHONHASHSEED'] = str(seed)


def files(it: Iterable['os.DirEntry[str]']) -> Iterable['os.DirEntry[str]']:
    return (e for e in it if e.is_file())


def dirs(it: Iterable['os.DirEntry[str]']) -> Iterable['os.DirEntry[str]']:
    return (e for e in it if e.is_dir())


# Python equivalent of 'find "$dir" -maxdepth 1 -xtype f'
def dir_to_file_set(path: StrPath) -> Set[str]:
    with os.scandir(path) as it:
        return {entry.name for entry in files(it)}


def set_fweights(it: Iterable['os.DirEntry[str]'], weights: Dict[str, float], weight: float) -> None:
    for wf in files(it):
        if wf.name not in weights or weight > weights[wf.name]:
            weights[wf.name] = weight


def write_file(fold: int, name: str, samples: Iterable[str], labels: Dict[str, Set[str]]) -> None:
    folder = 'fold{}'.format(fold)
    try:
        os.mkdir(folder)
    except FileExistsError:
        pass
    outfile = os.path.join(folder, '{}_tagged.csv'.format(name))
    with open(outfile, 'w') as outf:
        writer = csv.writer(outf)
        writer.writerow(['image_name', 'tags'])
        def write(fname: str) -> None:
            f_labels = (cl for cl, membs in labels.items() if fname in membs)
            writer.writerow([fname, ' '.join(f_labels)])
        for fname in samples:
            write(fname)


def file2lines(file: str) -> Iterator[str]:
    with open(file) as f:
        yield from (l.rstrip('\n') for l in f)


# NB: groups are retained but not enforced
def filter_dataset(X: Sequence[str], y: Array, groups: Array, cond: Callable[[str, Sequence[int]], bool]) \
        -> Tuple[List[str], Array, Array]:
    new_X = []
    new_y = []
    new_groups = []
    for Xi, yi, gi in zip_strict(X, y, groups):
        if cond(Xi, yi):
            new_X.append(Xi)
            new_y.append(yi)
            new_groups.append(gi)
    return new_X, np.asarray(new_y), np.asarray(new_groups)


def shuffle_dataset(X: Sequence[str], y: Array, groups: Array) -> Tuple[List[str], Array, Array]:
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    return [X[i] for i in indices], y[indices], groups[indices]


def train_test_split(
    X: Sequence[str], y: Array, groups: Array, n_labels: int,
    test_size: Optional[float] = None, n_splits: Optional[int] = None,
) -> Iterator[Tuple[List[str], List[str], Array, Array, Array, Array]]:
    if (test_size is None) == (n_splits is None):
        raise ValueError('expected one of test_size or n_splits')
    stratifier = MLStratifiedGroupKFold(
        n_labels=n_labels,
        n_splits=2 if n_splits is None else n_splits,
        fold_ratios=None if test_size is None else [test_size, 1 - test_size],
    )
    splits = stratifier.split(X, y, groups)
    for train_indices, test_indices in splits:
        X_train = [X[i] for i in train_indices]
        X_test = [X[i] for i in test_indices]
        yield X_train, X_test, y[train_indices], y[test_indices], groups[train_indices], groups[test_indices]


def apply_weights(X: Sequence[str], y: Array, weights: Dict[str, float]) \
        -> Tuple[List[str], Array, Array]:
    weighted_X = []
    weighted_y = []
    groups = []
    def write(i: int, Xi: str, yi: Array) -> None:
        weighted_X.append(Xi)
        weighted_y.append(yi)
        groups.append(i)
    for i, (Xi, yi) in enumerate(zip_strict(X, y)):
        weight = weights.get(Xi, 1)

        if weight < 0 or np.isclose(weight, 0):
            raise ValueError('Got negative weight: {}'.format(weight))
        if np.isclose(weight, round(weight)):
            weight_int, weight_float = round(weight), None
        else:
            weight_int = int(weight)
            weight_float = weight - weight_int

        for _ in range(weight_int):  # Repeat N times if weight is N
            write(i, Xi, yi)
        if weight_float is not None:
            if np.random.binomial(1, weight_float):
                write(i, Xi, yi)

    return weighted_X, np.stack(weighted_y), np.asarray(groups)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--splits', type=int, metavar='N', default=5)
    opts = parser.parse_args()

    seed_all(24)

    with os.scandir(CLASS_DIR) as it:
        labels = {entry.name: dir_to_file_set(entry) for entry in it if entry.is_dir()}

    X = list(file2lines('all.txt'))
    tags = [
        [l for l, membs in labels.items() if fname in membs]
        for fname in X
    ]
    mlb = MultiLabelBinarizer()
    y = mlb.fit_transform(tags)

    weights: Dict[str, float] = {}
    if os.path.isdir(WEIGHT_DIR):
        with os.scandir(WEIGHT_DIR) as wdit:
            for wdir in dirs(wdit):
                with os.scandir(wdir) as wfit:
                    set_fweights(wfit, weights, float(wdir.name))

    X, y, groups = apply_weights(X, y, weights)
    for _ in range(5):
        X, y, groups = shuffle_dataset(X, y, groups)

    split = partial(train_test_split, n_labels=len(mlb.classes_))

    trainable_X = X
    trainable_y = y
    trainable_groups = groups
    test_only_X: List[str] = []
    test_only_y: Array = np.array([])
    test_only_groups: Array = np.array([])
    if os.path.isdir('test_only'):
        exclude = dir_to_file_set('test_only')
        test_only_X, test_only_y, test_only_groups = filter_dataset(X, y, groups, lambda Xi, _: Xi in exclude)
        trainable_X, trainable_y, trainable_groups = filter_dataset(X, y, groups, lambda Xi, _: Xi not in exclude)

    datafiles: Dict[str, List[str]] = {}
    for n, fold in enumerate(split(trainable_X, trainable_y, trainable_groups, n_splits=opts.splits)):
        train_X, not_train_X, train_y, not_train_y, train_groups, not_train_groups = fold

        datafiles.clear()
        datafiles['train'] = train_X

        not_train_X.extend(test_only_X)
        not_train_y      = np.concatenate((not_train_y,      test_only_y))  # type: ignore[no-untyped-call]
        not_train_groups = np.concatenate((not_train_groups, test_only_groups))  # type: ignore[no-untyped-call]
        not_train_X, not_train_y, not_train_groups = shuffle_dataset(not_train_X, not_train_y, not_train_groups)

        opt_X, test_X, opt_y, test_y, _, _ = next(split(not_train_X, not_train_y, not_train_groups, test_size=.5))
        datafiles['opt'] = opt_X
        datafiles['test'] = test_X

        for name in ('train', 'opt', 'test'):
            write_file(n, name, datafiles[name], labels)


if __name__ == '__main__':
    main()
