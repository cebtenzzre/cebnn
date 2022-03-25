# -*- coding: utf-8 -*-

from __future__ import annotations

import operator
import os
import re
from abc import ABCMeta, abstractmethod
from functools import reduce
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torch import Tensor
from torch.utils.data.dataset import Dataset, Subset

Element = tuple[Any, Tensor]

if TYPE_CHECKING:
    from typing import Callable, Sequence, Set, Tuple, Union
    from pandas._typing import FilePathOrBuffer
    from util import Array
    StrPath = Union[str, os.PathLike[str]]


class LabeledDataset(Dataset[Element], metaclass=ABCMeta):
    @abstractmethod
    def __len__(self) -> int:
        raise TypeError

    @property
    @abstractmethod
    def labelset(self) -> Sequence[Set[str]]:
        raise NotImplementedError

    @property
    @abstractmethod
    def targets(self) -> Array:
        raise NotImplementedError

    @property
    @abstractmethod
    def groups(self) -> Array:
        raise NotImplementedError


class MultiLabelCSVDataset(LabeledDataset):
    """Dataset wrapping images and target labels.

    Arguments:
        csv_path: CSV file path
        img_path: Image folder path
        classes: Optional list of class names
    """

    def __init__(self, classes: Sequence[str], csv_path: FilePathOrBuffer, img_path: StrPath) -> None:
        self.data_frame = pd.read_csv(csv_path, keep_default_na=False)
        self.data_frame['image_name'] = self.data_frame['image_name'] \
            .apply(lambda x: re.sub(r'\.[^.]+$', '.png', x))
        notfound = [x for x in self.data_frame['image_name']
                    if not os.path.isfile(os.path.join(img_path, x))]
        if notfound:
            raise RuntimeError('Some images referenced in the CSV file were not found:\n{}'
                               .format(notfound))

        self._label_encoder = MultiLabelBinarizer(classes=classes)
        self.img_path = img_path

        tags = self.data_frame['tags'].str.split() \
            .apply(lambda ls: tuple(filter(classes.__contains__, ls)))

        self._X = self.data_frame['image_name']
        self._y = self._label_encoder.fit_transform(tags).astype(np.float32)
        _, self._groups = np.unique(self._X, return_inverse=True)

        self._labelset: Tuple[Set[str], ...] = tuple(map(set, tags))

    def __getitem__(self, index: int) -> Element:
        with Image.open(os.path.join(self.img_path, self._X[index])) as img:
            img = img.convert('RGB')
        return img, torch.from_numpy(self._y[index])

    def __len__(self) -> int:
        return len(self.data_frame.index)

    @property
    def classes(self) -> Array: return self._label_encoder.classes_
    @property
    def labelset(self) -> Sequence[Set[str]]: return self._labelset
    @property
    def targets(self) -> Array: return self._y
    @property
    def groups(self) -> Array: return self._groups


class TransformedDataset(LabeledDataset):
    def __init__(self, dataset: LabeledDataset, transform: Callable[[Image.Image], Tensor]) -> None:
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, index: int) -> Element:
        img, labels = self.dataset[index]
        return self.transform(img), labels

    def __len__(self) -> int:
        return len(self.dataset)

    @property
    def labelset(self) -> Sequence[Set[str]]: return self.dataset.labelset
    @property
    def targets(self) -> Array: return self.dataset.targets
    @property
    def groups(self) -> Array: return self.dataset.groups


# Cannot use base directly, see https://github.com/google/pytype/issues/1151
@LabeledDataset.register
class LabeledSubset(Subset[Element]):
    dataset: LabeledDataset

    @property
    def labelset(self) -> Sequence[Set[str]]:
        lset = self.dataset.labelset
        return tuple(lset[i] for i in self.indices)

    @property
    def targets(self) -> Array:
        return self.dataset.targets[list(self.indices)]

    @property
    def groups(self) -> Array:
        return self.dataset.groups[list(self.indices)]


class CatDataset(LabeledDataset):
    def __init__(self, *datasets: LabeledDataset) -> None:
        self.datasets = datasets

    def __getitem__(self, index: int) -> Element:
        it = iter(self.datasets)
        while True:
            try:
                ds = next(it)
            except StopIteration:
                raise IndexError('CatDataset index out of range') from None
            if index < (l := len(ds)):
                break
            index -= l
        return ds

    def __len__(self) -> int:
        return sum(len(ds) for ds in self.datasets)

    @property
    def labelset(self) -> Sequence[Set[str]]:
        return reduce(operator.add, (list(ds.labelset) for ds in self.datasets))

    @property
    def targets(self) -> Array:
        return np.concatenate([ds.targets for ds in self.datasets])

    @property
    def groups(self) -> Array:
        return np.concatenate([ds.groups for ds in self.datasets])
