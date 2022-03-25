# -*- coding: utf-8 -*-

from __future__ import annotations

import copy
import io
import math
import random
from functools import partial
from typing import TYPE_CHECKING, List, Tuple

import torch
from Augmentor.Operations import Distort, Operation, Skew
from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from PIL import Image, ImageChops
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from cebnn_common import Module
from dataset import LabeledDataset, LabeledSubset, TransformedDataset

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, Optional, Tuple, TypeVar, Union

    T = TypeVar('T')


def rand_jpeg_compression(image: Image, quality: Tuple[int, int]) -> Image:
    stream = io.BytesIO()
    image.save(stream, format='jpeg', subsampling=0, quality=random.randint(*quality))
    stream.seek(0)
    with Image.open(stream) as img:
        return img.convert('RGB')


def pil_translate(img: Image, p: float, mode: str, max_: int) -> Image:
    if random.uniform(0, 1) >= p:
        return img  # Skip the operation
    xoff = random.randint(-max_, max_)
    yoff = random.randint(-max_, max_)
    if mode == 'paste':
        imgcp = img.copy()
        imgcp.paste(img, (xoff, yoff))
        return imgcp
    elif mode == 'chop':
        return ImageChops.offset(img, xoff, yoff)
    else:
        raise ValueError('Invalid mode!')


def weighted_mix(a: Any, b: Any, b_weight: float) -> Any:
    return a * (1 - b_weight) + b * b_weight


def apply_n(f: Callable[[T], T], x: T, n: int) -> T:
    val = x
    for _ in range(n):
        val = f(val)
    return val


class AugmentorTransform:
    def __init__(self, op: Operation) -> None:
        self.op = op

    def __call__(self, img: Image) -> Image:
        if random.uniform(0, 1) < self.op.probability:
            img, = self.op.perform_operation((img,))
        return img


class AugmentorTransformPickOne(AugmentorTransform):
    def __init__(self, op: Operation, op2: Operation) -> None:
        self.op = op
        self.op2 = op2

    def __call__(self, img: Image) -> Image:
        if random.uniform(0, 1) < self.op.probability:
            img, = self.op.perform_operation((img,))
        else:
            img, = self.op2.perform_operation((img,))
        return img


class DistortTransform(AugmentorTransform):
    def __init__(self, p: float, gridsz: int, mag: float) -> None:
        super().__init__(Distort(p, gridsz, gridsz, 0))
        self.gridsz = gridsz
        self.mag = mag

    def __call__(self, img: Image) -> Image:
        self.op.magnitude = round(self.mag * self.gridsz / img.size[0])
        return super().__call__(img)


# From https://stackoverflow.com/a/16778797
def rotation_crop_dims(w: float, h: float, angle: float) -> Tuple[float, float]:
    """
    Compute the size of the largest axis-aligned rectangle within a rotated rectangle.

    Arguments:
        w (float): Width of the rotated rectangle
        h (float): Height of the ratated rectangle
        angle (float): The angle the rectangle is rotated by, in radians
    """
    if w <= 0 or h <= 0:
        return 0, 0

    width_is_longer = w >= h
    side_long, side_short = (w, h) if width_is_longer else (h, w)

    # Since the solutions for angle, -angle and 180-angle are all the same,
    # if suffices to look at the first quadrant and the absolute values of sin,cos:
    sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
    if side_short <= 2 * sin_a * cos_a * side_long or abs(sin_a - cos_a) < 1e-10:
        # Half constrained case: two crop corners touch the longer side,
        # the other two corners are on the mid-line parallel to the longer line
        x = side_short / 2
        wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
    else:
        # Fully constrained case: crop touches all 4 sides
        cos_2a = cos_a * cos_a - sin_a * sin_a
        wr = (w * cos_a - h * sin_a) / cos_2a
        hr = (h * cos_a - w * sin_a) / cos_2a

    return wr, hr


class RotateRange(Operation):
    def __init__(self, probability: float, max_rotation: float, interpolation: int = Image.BILINEAR) -> None:
        assert max_rotation >= 0
        super().__init__(probability)
        self.max_rotation = max_rotation
        self.interpolation = interpolation

    def perform_operation(self, images: List[Image]) -> List[Image]:
        if self.max_rotation == 0:
            return images

        rotation = random.uniform(-self.max_rotation, self.max_rotation)

        def rotate(image: Image) -> Image:
            w_old, h_old = image.size

            # Do the rotation
            image = image.rotate(rotation, expand=True, resample=self.interpolation)
            w_bb, h_bb = image.size

            # Get the largest possible crop
            w_crop, h_crop = rotation_crop_dims(w_old, h_old, math.radians(rotation))
            w_crop = h_crop = min(w_crop, h_crop)  # Make it square
            h_inset, v_inset = (w_bb - w_crop) / 2, (h_bb - h_crop) / 2

            # Perform the crop, rounding inwards to avoid black pixels
            return image.crop((math.ceil(h_inset), math.ceil(v_inset),
                               math.floor(w_bb - h_inset), math.floor(h_bb - v_inset)))

        return list(map(rotate, images))


class RandomCrop(transforms.RandomCrop, Operation):
    def __init__(self, crop: float, pad: float) -> None:
        transforms.RandomCrop.__init__(self, (None, None))
        self.crop = crop
        self.pad = pad

    def __call__(self, img: Image) -> Image:
        if self.crop == 1 and self.pad == 0:
            return img
        self.size = (round(img.size[0] * self.crop), round(img.size[1] * self.crop))
        self.padding = (round(img.size[0] * self.pad), round(img.size[1] * self.pad))
        return super().__call__(img)

    def perform_operation(self, images: List[Image]) -> List[Image]:
        return list(map(self.__call__, images))


class MayResize:
    def __init__(
        self,
        size: Optional[tuple[int, int]] = None,
        interpolation: InterpolationMode = InterpolationMode.BILINEAR
    ) -> None:
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img: Image) -> Image:
        if self.size is None or img.size == self.size:
            return img
        return transforms.functional.resize(img, self.size, self.interpolation)


def make_data_transform(distort: Optional[Tuple[float, int, float]], skew: Optional[Tuple[float, float]],
                        rotate: Optional[Tuple[float, float]], crop: Optional[Tuple[float, float]],
                        translate: Optional[Tuple[float, str, int]], erasing: Optional[Tuple[float, float]],
                        brjitter: Union[float, Tuple[float, float]], ctjitter: float, huejitter: float,
                        noise_factors: Tuple[float, float], pepper_factor: float, jpeg_iterations: int,
                        jpeg_quality: Tuple[float, float]) -> transforms.Compose:
    ops = [transforms.RandomHorizontalFlip()]
    if distort is not None:
        ops.append(DistortTransform(*distort))
    if skew is not None:
        ops.append(AugmentorTransform(Skew(skew[0], 'RANDOM', skew[1])))
    if rotate is not None and crop is not None:
        ops.append(AugmentorTransformPickOne(RotateRange(*rotate), RandomCrop(*crop)))
    else:
        if rotate is not None:
            ops.append(AugmentorTransform(RotateRange(*rotate)))
        if crop is not None:
            ops.append(RandomCrop(*crop))
    ops.append(MayResize())
    if translate is not None:
        ops.append(transforms.Lambda(lambda x: pil_translate(x, *translate)))
    ops += [
        transforms.ColorJitter(brjitter, ctjitter, 0, huejitter),
        transforms.ToTensor(),
    ]
    if erasing is not None:
        ops.append(transforms.RandomErasing(erasing[0], (.02, erasing[1])))
    if noise_factors[0] > 0.:
        ops.append(transforms.Lambda(lambda x: weighted_mix(x, torch.randn_like(x), noise_factors[0])))
    if pepper_factor > 0.:
        ops.append(transforms.Lambda(
            lambda x: x * torch.multinomial(
                torch.tensor((pepper_factor, 1. - pepper_factor)),
                x.numel(),
                replacement=True,
            ).view_as(x),
        ))
    ops.append(transforms.ToPILImage())
    if jpeg_iterations > 0:
        ops.append(transforms.Lambda(
            lambda x: apply_n(partial(rand_jpeg_compression, quality=jpeg_quality), x, jpeg_iterations)))
    ops.append(transforms.ToTensor())
    if noise_factors[1] > 0.:
        ops.append(transforms.Lambda(lambda x: x + torch.randn_like(x) * noise_factors[1]))
    return transforms.Compose(ops)


def find_best_augment_params(train_dataset: LabeledDataset, opt_dataset_tformed: LabeledDataset,
                             optimizer: Optimizer, model: Module, device: torch.device, criterion: Module,
                             make_dataloader: Callable[..., DataLoader]) -> Dict[str, Any]:
    max_evals = 200
    # Cycle the train dataset a few times to simulate multiple epochs
    find_aug_train_ds = LabeledSubset(train_dataset, list(range(len(train_dataset))) * 3)

    def objective(args: Dict[str, Any]) -> Dict[str, Any]:
        train_transform = make_data_transform(
            None if args['distort'] is None else
            (float(args['distort']['p']), int(args['distort']['gridsz']), float(args['distort']['mag'])),
            None if args['skew'] is None else
            (float(args['skew']['p']), float(args['skew']['mag'])),
            None if args['rotate'] is None else
            (float(args['rotate']['p']), float(args['rotate']['max'])),
            None if args['crop_factor'] is None else (float(args['crop_factor']), .1),
            None if args['translate'] is None else
            (float(args['translate']['p']), args['translate']['mode'], int(args['translate']['max'])),
            None if args['erasing'] is None else (float(args['erasing']['p']), float(args['erasing']['maxsc'])),
            (.01, 1), 0, .45,
            (float(args['noise_factor_1']), float(args['noise_factor_2'])),
            float(args['pepper_factor']),
            int(args['jpeg_iterations']),
            (
                int(args['jpeg_quality_min']),
                round(100 - args['jpeg_quality_maxf'] * (100 - args['jpeg_quality_min'])),
            ),
        )
        train_iter = make_dataloader(TransformedDataset(find_aug_train_ds, train_transform))

        # Save the model and optimizer
        model_state_dict = copy.deepcopy(model.state_dict())
        optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

        # Train
        model.train()
        for inputs, labels in train_iter:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            for param in model.parameters():
                param.grad = None  # Zero the gradient
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels, 0)
            # Backward pass
            loss.backward()
            optimizer.step()
        del train_iter

        # Evaluate
        opt_iter = make_dataloader(opt_dataset_tformed)
        model.eval()
        running_loss = 0.
        with torch.no_grad():
            for inputs, labels in opt_iter:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                outputs = model(inputs)
                running_loss += criterion(outputs, labels, 0).item() * len(labels)

        # Restore the model and optimizer
        model.load_state_dict(model_state_dict)
        optimizer.load_state_dict(optimizer_state_dict)

        return {'loss': running_loss / len(opt_iter), 'status': STATUS_OK}

    space = {
        'distort': hp.choice('distort', [
            None,
            {
                'p': hp.uniform('ds_p', 0, 1),
                'gridsz': hp.quniform('ds_gridsz', 2, 10, 1),
                'mag': hp.uniform('ds_mag', 0, 1),
            },
        ]),
        'skew': hp.choice('skew', [
            None,
            {
                'p': hp.uniform('sk_p', 0, 1),
                'mag': hp.uniform('sk_mag', 0, 1),
            },
        ]),
        'rotate': hp.choice('rotate', [
            None,
            {
                'p': hp.uniform('rt_p', 0, 1),
                'max': hp.uniform('rt_max', 0, 5),
            },
        ]),
        'crop_factor': hp.choice('crop_factor', [None, hp.uniform('cf_float', 0, 1)]),
        'translate': hp.choice('translate', [
            None,
            {
                'p': hp.uniform('tr_p', 0, 1),
                'mode': hp.choice('tr_mode', ['paste', 'chop']),
                'max': hp.quniform('tr_max', 1, 25, 1),
            },
        ]),
        'erasing': hp.choice('erasing', [
            None,
            {
                'p': hp.uniform('er_p', 0, 1),
                'maxsc': hp.uniform('er_maxsc', .1, .33),
            },
        ]),
        'noise_factor_1': hp.choice('noise_factor_1', [0, hp.uniform('nf1_float', .02, .08)]),
        'noise_factor_2': hp.choice('noise_factor_2', [0, hp.uniform('nf2_float', .02, .08)]),
        'pepper_factor': hp.choice('pepper_factor', [0, hp.uniform('pf_float', .005, .05)]),
        'jpeg_iterations': hp.quniform('jpeg_iterations', 1, 10, 1),
        'jpeg_quality_min': hp.quniform('jpeg_quality_min', 1, 90, 1),
        'jpeg_quality_maxf': hp.uniform('jpeg_quality_maxf', 0., 1.),
    }
    return fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=max_evals, trials=Trials(),
                return_argmin=False)
