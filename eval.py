#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

from __future__ import annotations

import itertools
import os
import pickle
import sys
from pickle import UnpicklingError
from timeit import default_timer as timer
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from cebnn_common import CatModel, ModelLoader, pos_proba, pred_uncertainty
from scale import load_and_scale
from util import zip_strict

if TYPE_CHECKING:
    from typing import Callable, Iterator, List, Sequence, Tuple
    from torch import Tensor
    from cebnn_common import Module
    from util import Array


BATCH_SIZE = 64
WRITE_SIZE = 200  # In batches
NUM_WORKERS = 12
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_SAMPLE_DIM = 32 * 1024
CROP_FACTOR = .7  # Max crop of longer dimension


class ImageDataset(Dataset):
    sample_paths: Sequence[str]

    def __init__(self, sample_paths: Sequence[str],
                 transform: Callable[[Image.Image], Tensor]) -> None:
        self.sample_paths = sample_paths
        self.transform = transform

    def __getitem__(self, index: int) -> Image.Image:
        sample_path = self.sample_paths[index]
        sample = load_and_scale(sample_path)
        return self.transform(sample)

    def __len__(self) -> int:
        return len(self.sample_paths)


@torch.no_grad()  # type: ignore[misc]
def evaluate(model: Module, dataloader: DataLoader, data_cnames: Sequence[str], model_cnames: Sequence[str],
             samples_found: int) -> Iterator[Tuple[Array, Array]]:
    assert set(data_cnames).issuperset(set(model_cnames))
    model.eval()
    print('==> Evaluating...')
    start_time = timer()

    nan = float('NaN')  # Placeholder value
    y_pred: List[Tuple[float, ...]] = []
    y_u: List[Tuple[float, ...]] = []
    offset = samples_found // BATCH_SIZE
    input_it = (inputs.to(DEVICE, non_blocking=True)
                for inputs in tqdm(dataloader, initial=offset, total=len(dataloader) + offset))
    inputs = next(input_it)

    for batch in itertools.count():
        raw_pred = model(inputs)

        # Get inputs moving early
        have_more_data = True
        try:
            inputs = next(input_it)
        except StopIteration:
            have_more_data = False  # Last loop

        pred = pos_proba(raw_pred)
        uncertainty = pred_uncertainty(raw_pred)
        for sample_preds, sample_uncertainties in zip_strict(pred.cpu().numpy(), uncertainty.cpu().numpy()):
            lbl_to_pred = dict(zip_strict(model_cnames, sample_preds))
            lbl_to_uncertainty = dict(zip_strict(model_cnames, sample_uncertainties))
            # Substitute with NaN if label not predicted
            y_pred.append(tuple(lbl_to_pred.get(lbl, nan) for lbl in data_cnames))
            y_u.append(tuple(lbl_to_uncertainty.get(lbl, nan) for lbl in data_cnames))

        if not have_more_data:
            break

        if batch % WRITE_SIZE == WRITE_SIZE - 1:
            yield np.asarray(y_pred), np.asarray(y_u)
            y_pred.clear()
            y_u.clear()

    # Final chunk
    yield np.asarray(y_pred), np.asarray(y_u)

    end_time = timer()
    print('==> Done in {:.2f}s'.format(end_time - start_time))


if __name__ == '__main__':
    if len(sys.argv) != 5:
        raise ValueError('Expected 4 arguments, got {}'.format(len(sys.argv) - 1))

    checkpoint_path, data_cnames_str, sample_paths_file, output_file = sys.argv[1:]
    data_cnames = data_cnames_str.split(',')
    del data_cnames_str

    checkpoint = torch.load(checkpoint_path)
    print("==> Loaded checkpoint from '{}'".format(checkpoint_path))
    print('  Base: {}'.format(checkpoint['base_model']))
    print('  Epochs: {}'.format(checkpoint['epoch'] + 1))
    print('  Test loss: {:.4f}'.format(checkpoint.get('test_loss')))

    with open(sample_paths_file) as spf:
        sample_paths = [line.rstrip('\n') for line in spf]

    try:
        model_cnames = checkpoint['out_classes']
    except KeyError:
        model_cnames = data_cnames  # Model provides all classes
        assert checkpoint['out_features'] == len(model_cnames)
    else:
        assert model_cnames

    model = ModelLoader(checkpoint=checkpoint, tta_mode='mean').create_model().to(DEVICE)
    del checkpoint

    if isinstance(model, CatModel):
        # CatModel has built-in scaling and normalization
        data_transform = transforms.Compose([
            # Scale to a common size
            transforms.Resize((
                max(m.default_cfg['input_size'][-2] for m in model.models),
                max(m.default_cfg['input_size'][-1] for m in model.models),
            )),
            transforms.ToTensor(),
        ])
    else:
        # Use timm-style cfg to get correct normalization parameters
        normalize = transforms.Normalize(model.default_cfg['mean'], model.default_cfg['std'])
        data_transform = transforms.Compose([
            transforms.Resize(model.default_cfg['input_size'][-2:]),  # Scaling for non-CatModel
            transforms.ToTensor(),
            normalize,
        ])

    fd = os.open(output_file, os.O_RDWR | os.O_CREAT, 0o644)
    with open(fd, 'r+b') as predfile:
        # Resume support
        header_found = False
        samples_found = 0
        good_pos = 0
        try:
            good_pos = predfile.tell()
            header = pickle.load(predfile)
            if 'class_names' not in header or 'sample_paths' not in header:
                raise ValueError('Bad header: {}'.format(header))
            if header['class_names'] != model_cnames or header['sample_paths'] != sample_paths:
                raise ValueError('Header does not match ours')
            header_found = True

            while True:
                good_pos = predfile.tell()
                chunk = pickle.load(predfile)
                samples_found += len(chunk['preds'])
        except EOFError:
            pass
        except UnpicklingError:
            predfile.seek(good_pos)
            predfile.truncate()

        dataloader = DataLoader(
            ImageDataset(sample_paths[samples_found:], data_transform),
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            pin_memory=True,
        )

        # Header
        if not header_found:
            pickle.dump({'class_names': model_cnames, 'sample_paths': sample_paths}, predfile)

        # Incremental appends
        for preds, uncertainties in evaluate(model, dataloader, data_cnames, model_cnames, samples_found):
            pickle.dump({'preds': preds, 'uncertainties': uncertainties}, predfile)
