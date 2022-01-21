#!/usr/bin/env python3.8
# -*- coding: utf-8 -*-

from __future__ import annotations

import multiprocessing as mp
import pickle
import struct
import sys
import threading
from timeit import default_timer as timer
from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torchvision import transforms

from cebnn_common import ModelLoader, pos_proba
from util import zip_strict

if TYPE_CHECKING:
    from typing import Callable, Iterator, List, TypeVar
    from torch import Tensor
    from cebnn_common import Module
    from util import Array
    T = TypeVar('T')


BATCH_SIZE = 128
NUM_WORKERS = 4
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAX_SAMPLE_DIM = 32 * 1024


class ImageDataset(IterableDataset):  # pylint: disable=abstract-method
    sample_path_it: Iterator[str]

    def __init__(self, sample_path_it: Iterator[str],
                 transform: Callable[[Image.Image], Tensor]) -> None:
        self.sample_path_it = sample_path_it
        self.transform = transform

    def __iter__(self) -> Iterator[Tensor]:
        return iter(map(self._load, self.sample_path_it))

    def _load(self, sample_path: str) -> Tensor:
        try:
            with Image.open(sample_path) as sample:
                if getattr(sample, 'is_animated', False):
                    sample.seek(sample.n_frames - 1)  # Last frame is a safe choice  # pytype: disable=attribute-error
                if sample is not None and (sample.width > MAX_SAMPLE_DIM or sample.height > MAX_SAMPLE_DIM):
                    print('Image dimensions too large! {}x{} > {msd}x{msd}, file: {}'.format(
                        sample.width, sample.height, sample_path, msd=MAX_SAMPLE_DIM))
                    sample = None
                else:
                    sample = sample.convert('RGB')
        except (OSError, SyntaxError, Image.DecompressionBombError, struct.error) as e:
            print('Caught error loading {}: {}'.format(sample_path, e))
            sample = None

        if sample is None:
            print('Generating blank sample image due to unusable file')
            sample = Image.new('RGB', (224, 224))  # Black replacement image

        return self.transform(sample)


# From https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p_metrics.py
def evaluate(model: Module, dataloader: DataLoader, log: bool = True) -> Array:
    model.eval()
    predictions = []

    start_time = None
    if log:
        print('==> Evaluating...')
        start_time = timer()

    with torch.no_grad():
        for inputs in dataloader:
            raw_pred = model(inputs.to(DEVICE, non_blocking=True))
            pred = pos_proba(raw_pred)
            predictions.append(pred.data.cpu().numpy())

    if start_time is not None:
        end_time = timer()
        print('==> Done in {:.2f}s'.format(end_time - start_time))

    return np.vstack(predictions)


class get_sample_paths:
    def __init__(self, quit_event: mp.synchronize.Event, queue: mp.Queue[str], out_list: List[str]) -> None:
        self.lock = mp.Lock()
        self.quit_event = quit_event
        self.queue = queue
        self.out_list = out_list

    def __iter__(self: T) -> T:
        return self

    def __next__(self) -> str:
        with self.lock:
            if self.quit_event.is_set() and not self.queue.qsize():
                raise StopIteration('queue is empty')
            path = self.queue.get()
        self.out_list.append(path)
        return path


if __name__ == '__main__':
    raise RuntimeError('This script tends to stop reading input before the end. Prefer eval.py.')
    mp.set_start_method('forkserver')  # Fastest safe option

    if len(sys.argv) != 4:
        raise ValueError('Wrong number of arguments')

    checkpoint_path, sample_paths_file, output_file = sys.argv[1:]

    checkpoint = torch.load(checkpoint_path)
    print("==> Loaded checkpoint from '{}'".format(checkpoint_path))
    print('  Base: {}'.format(checkpoint['base_model']))
    print('  Epoch: {}'.format(checkpoint['epoch']))
    print('  Test loss: {:.4f}'.format(checkpoint.get('test_loss')))

    normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])

    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    sample_paths: mp.Queue[str] = mp.Queue(maxsize=1024)
    sample_paths_done = mp.Event()

    def read_paths():
        with open(sample_paths_file) as spf:
            for line in spf:
                sample_paths.put(line.rstrip('\n'))
        sample_paths_done.set()

    gsp_thread = threading.Thread(target=read_paths)
    gsp_thread.start()

    if NUM_WORKERS <= 1:
        used_sample_paths: List[str] = []
    else:
        # Proxy list that is appended to from worker threads
        manager = mp.Manager()
        used_sample_paths = manager.list()

    dataset = ImageDataset(
        get_sample_paths(sample_paths_done, sample_paths, used_sample_paths),
        data_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=False)

    model = ModelLoader(checkpoint=checkpoint).create_model().to(DEVICE)

    with open(output_file, 'wb') as pf:
        preds = evaluate(model, dataloader)
        gsp_thread.join()
        pickle.dump(tuple(zip_strict(used_sample_paths, preds)), pf)
