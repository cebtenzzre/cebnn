#!/usr/bin/env python3.10
# -*- coding: utf-8 -*-

from __future__ import annotations

# For deterministic behavior
__import__('os').environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import gc
import io
import math
import multiprocessing
import os
import pickle
import random
import re
import subprocess
import sys
import termios
import time
import warnings
from collections import defaultdict
from functools import reduce
from itertools import combinations
from numbers import Real
from operator import itemgetter
from timeit import default_timer as timer
from typing import TYPE_CHECKING, Any, Container, List, Literal, Optional, Tuple, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from adabelief_pytorch import AdaBelief  # type: ignore[attr-defined]
from apex.optimizers import FusedLAMB
from pytorch_warmup import UntunedLinearWarmup
from scipy.optimize import basinhopping
from sklearn.metrics import matthews_corrcoef, roc_auc_score, roc_curve
from skorch import NeuralNetClassifier
from skorch.exceptions import SkorchWarning
from skorch.callbacks import Callback, EpochScoring, PrintLog, ProgressBar
from skorch.dataset import get_len, unpack_data
from skorch.utils import TeeGenerator, to_tensor
from tap import Tap
from torch import Tensor, autograd, nn
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import mypickle
from aadamw01 import AAdamW01
from adamwr.adamw import AdamW
from adamwr.cyclic_scheduler import CyclicLRWithRestarts
from algorithm import fbeta_like_geo_youden, multilabel_confusion_matrix
from augment import MayResize, find_best_augment_params, make_data_transform
from cebnn_common import ModelLoader, model_supports_resnet_d, parse_sublayer_ratio, pos_proba, pred_uncertainty
from datamunge import ImbalancedDatasetSampler
from dataset import CatDataset, LabeledDataset, LabeledSubset, MultiLabelCSVDataset, TransformedDataset
from losses import EDL_Digamma_Loss, EDL_Log_Loss, EDL_MSE_Loss
from sgdw import SGDW
from util import zip_strict

if TYPE_CHECKING:
    from typing import Callable, Dict, Iterable, Iterator, NoReturn, Sequence, Set, Type, Union

    from sklearn.base import BaseEstimator
    from torch.optim.lr_scheduler import _LRScheduler
    from torch.utils.data.dataset import Dataset

    from cebnn_common import Module
    from util import Array

    StrPath = Union[str, 'os.PathLike[str]']
    AnyEstimator = TypeVar('AnyEstimator', bound=BaseEstimator)
    AnyResampleArgs = TypeVar('AnyResampleArgs', bound='ResampleArgs')
    AnyRLS = TypeVar('AnyRLS', bound='ResampledLabeledSubset')
    AnyDataset = Union[Dataset[Any]]
    AnyNNC = TypeVar('AnyNNC', bound='MyNeuralNetClassifier')

T = TypeVar('T')

DEFAULT_SUBLAYERS = (0, .5)
DEFAULT_EPOCHS = 20
DEFAULT_BATCH_SIZE = 16
DEFAULT_ANNEALING_STEP = 10
DEFAULT_OPTIMIZER = 'adam'
DEFAULT_SCHEDULER = 'linear'
DEFAULT_SCH_PERIOD = 1
DEFAULT_SEED = 42
DEFAULT_NUM_WORKERS = 4
UNTYPED_NONE: Any = None

cfg: 'Config' = UNTYPED_NONE
transformed_datasets: Dict[str, LabeledDataset] = {}
device: torch.device = UNTYPED_NONE
options: 'MainArgParser' = UNTYPED_NONE


def disable_echo() -> None:
    fd = sys.stdin.fileno()
    if not os.isatty(fd):
        return
    attr = termios.tcgetattr(fd)
    attr[3] &= ~termios.ECHO  # type: ignore[operator]
    termios.tcsetattr(fd, termios.TCSADRAIN, attr)


def seed_all(seed: int) -> None:
    # See https://pytorch.org/docs/stable/notes/randomness.html
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def imshow(inp: Tensor, title: Optional[str] = None) -> None:
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.show()


class GradScale(autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: Tensor, factor: Tensor) -> Tensor:
        ctx.save_for_backward(factor)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        factor, = ctx.saved_tensors
        return grad_output * factor, None


def grad_scale(x: Tensor, factor: Tensor) -> Tensor:
    return GradScale.apply(x, factor)


class ExtraTrainLoss(autograd.Function):
    @staticmethod
    def forward(ctx: Any, base_loss: Tensor, extra_loss: Tensor) -> Tensor:
        with torch.enable_grad():
            detached_base_loss = base_loss.detach()
            detached_base_loss.requires_grad_()
            detached_extra_loss = extra_loss.detach()
            detached_extra_loss.requires_grad_()
            total_loss = detached_base_loss + detached_extra_loss
        ctx.saved_losses = detached_base_loss, detached_extra_loss
        ctx.save_for_backward(total_loss)
        return base_loss.clone()

    @staticmethod
    def backward(ctx: Any, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        total_loss, = ctx.saved_tensors
        base_loss, extra_loss = ctx.saved_losses
        with torch.enable_grad():
            total_loss.backward(grad_output)
        return base_loss.grad, extra_loss.grad


def extra_train_loss(base_loss: Tensor, extra_loss: Tensor) -> Tensor:
    return ExtraTrainLoss.apply(base_loss, extra_loss)


class ExtraTrainLossMixin(nn.Module):
    def __init__(self, norm_params: Sequence[nn.Parameter], model: Module, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._norm_params = norm_params
        self._model = model
        self.register_buffer('zero', torch.zeros((), device=device))
        if options.l1reg is not None:
            self.register_buffer('l1reg', torch.tensor(options.l1reg, device=device))
        if options.l2reg is not None:
            self.register_buffer('l2reg', torch.tensor(options.l2reg, device=device))
        if options.corrreg is not None:
            self.register_buffer('corrreg', torch.tensor(options.corrreg, device=device))

    def forward(self, pred: Tensor, target: Tensor, epoch: int) -> Tensor:
        base_loss: Tensor = super().forward(pred, target, epoch)

        if not torch.is_grad_enabled():
            return base_loss  # Only apply extra loss while training
        if options.l1reg is None and options.l2reg is None and options.corrreg is None:
            return base_loss  # No extra loss needed

        def lp_loss(p: int, reg: Tensor) -> Tensor:
            return sum((grad_scale(np, reg).norm(p=p) for np in self._norm_params),
                       start=self.zero)

        l1_loss = l2_loss = corr_loss = self.zero

        if options.l1reg is not None:
            l1_loss = lp_loss(1, self.l1reg)
        if options.l2reg is not None:
            l2_loss = lp_loss(2, self.l2reg)

        # Correlation loss (see https://doi.org/10.1007/978-3-319-68612-7_6)
        def layer_corr_loss(layer_weights: Tensor, reg: Tensor) -> float:
            lw = grad_scale(layer_weights, reg)  # Gradient scale factor
            assert len(lw.size()) == 4  # Conv2d expected
            kernels = lw.view(-1, *lw.size()[:-2])

            return sum(self._pearsonr(si.flatten(), sj.flatten()) ** 2
                       for si, sj in combinations(kernels, 2))

        if options.corrreg is not None:
            corr_loss_layers = (
                l for l in self._model.modules()
                if isinstance(l, nn.Conv2d) and l.weight.requires_grad and any(s > 1 for s in l.kernel_size)
            )
            corr_loss = sum(layer_corr_loss(l.weight, self.corrreg) for l in corr_loss_layers)

        extra_loss = l1_loss + l2_loss + corr_loss
        assert base_loss.shape == extra_loss.shape
        return extra_train_loss(base_loss, extra_loss)

    # From https://github.com/pytorch/pytorch/issues/1254
    @staticmethod
    def _pearsonr(x: Tensor, y: Tensor) -> float:
        xm = x.sub(x.mean())
        ym = y.sub(y.mean())
        r_num = xm.dot(ym)
        r_den = xm.norm(2) * ym.norm(2)
        return r_num / r_den


class EDL_MSE_Loss_Extra(ExtraTrainLossMixin, EDL_MSE_Loss):
    pass


class EDL_Log_Loss_Extra(ExtraTrainLossMixin, EDL_Log_Loss):
    pass


class EDL_Digamma_Loss_Extra(ExtraTrainLossMixin, EDL_Digamma_Loss):
    pass


def prestep_clipping_optimizer(base_type: Type[Optimizer]) -> Type[Any]:
    assert issubclass(base_type, Optimizer)

    class PrestepClippingOptimizer(base_type):  # type: ignore[misc, valid-type]
        def __init__(self, params: Union[Sequence[nn.Parameter], Sequence[Dict[str, Any]]],
                     norm_params: Sequence[nn.Parameter], **kwargs: Any) -> None:
            super().__init__(params, **kwargs)
            self._norm_params = norm_params

        def step(self, *args: Any, **kwargs: Any) -> Any:
            # Only clip if there are gradients already and this is not resnet18
            if any(p.grad is not None for p in self._norm_params) and cfg.base_model != 'resnet18':
                # Clip gradient
                nn.utils.clip_grad_norm_(self._norm_params, .1)

            return super().step(*args, **kwargs)

    return PrestepClippingOptimizer


def print_predictions(net: NeuralNetClassifier, checkpoint: Dict[str, Any]) -> None:
    def format_labels(labels: Iterable[str]) -> str:
        labels = frozenset(labels)
        def lify(l: str) -> str: return '{},'.format(l) if l in labels else ' ' * (len(l) + 1)
        return ''.join(map(lify, cfg.model_classes))[:-1]

    metrics = Metrics()
    metrics.f_score_eval(net, checkpoint, log=False)

    for sample_targets, sample_preds in zip_strict(metrics.true_labels, metrics.predictions):
        in_labels = [cn for tl, cn in zip_strict(sample_targets, cfg.model_classes) if tl > .99]
        out_labels = [cn for pr, cn, th in zip_strict(sample_preds, cfg.model_classes, metrics.thresholds) if pr > th]

        if not in_labels and not out_labels:
            continue  # Not interesting

        print('actual: [{}], predicted: [{}]'.format(
            format_labels(in_labels), format_labels(out_labels)))


def geoy(true_labels: Array, predictions: Array, thresholds: Array,
         average: Optional[str] = None, i: Optional[int] = None) -> Any:
    assert true_labels.shape == predictions.shape

    y_true = true_labels if i is None else true_labels[:, i, None]
    y_pred = (predictions > thresholds) if i is None else (predictions[:, i, None] > thresholds.item())
    MCM = multilabel_confusion_matrix(y_true, y_pred, binary=options.optmode == 2)
    with np.errstate(invalid='ignore'):
        scores = list(map(fbeta_like_geo_youden, MCM))
    del MCM
    if average is not None:
        score: Any = np.mean(scores)
    elif i is None:
        score = np.stack(scores)
    else:
        score, = scores

    # Don't use the score if all predictions are true, all predictions are false, there are no true positives, or there
    # are no true negatives, for _any_ label. All of those are degenerate cases where the threshold is unusable.
    def badlabel(truei: Array, predi: Array) -> bool:
        return bool(
            np.all(predi > 0.) or np.all(predi < .99)
            or np.all(predi[truei > 0.] < .99) or np.all(predi[truei < .99] > 0.)
        )

    if i is None:
        for j in range(true_labels.shape[1]):
            if not badlabel(true_labels[:, j], (predictions[:, j] > thresholds[j])):
                continue
            if average is not None:
                return np.zeros(score.shape, score.dtype) if isinstance(score, np.ndarray) else 0.
            score[j] = 0.
    elif badlabel(y_true, y_pred):
        return np.zeros(score.shape, score.dtype) if isinstance(score, np.ndarray) else 0.

    return score.tolist() if isinstance(score, np.ndarray) else score


# Scipy tries to minimize the function, so we must get its inverse
def geoy_neg(true: Array, pred: Array, i: Optional[int] = None) -> Callable[[Array], float]:
    average = 'macro' if i is None and len(cfg.model_classes) > 1 else 'binary'
    return lambda th: - geoy(true, pred, th, average=average, i=i)


def minimizer_bounds(**kwargs: Any) -> bool:
    x: float = kwargs['x_new']
    tmax = bool(np.all(x <= 1.))
    tmin = bool(np.all(x >= 0.))
    return tmax and tmin


def minimize_global(true: Array, pred: Array, minimizer_kwargs: Dict[str, Any],
                    bounds: Callable[..., bool], st_point: float) -> List[Tuple[float, float]]:
    # We combine SLSQP with Basinhopping for stochastic search with random steps
    thr_0 = np.array([st_point for _ in cfg.model_classes])
    opt_output = basinhopping(geoy_neg(true, pred), thr_0, stepsize=.1, niter=150 if options.optmode == 1 else 20,
                              minimizer_kwargs=minimizer_kwargs, accept_test=bounds, seed=88)
    if len(cfg.model_classes) > 1:
        scores: List[float] = geoy(true, pred, opt_output.x, average=None)
    else:
        scores = [-opt_output.fun]

    return list(zip_strict(opt_output.x.tolist(), scores))


def refine(true: Array, pred: Array, minimizer_kwargs: Dict[str, Any],
           bounds: Callable[..., bool], st_point: float, i: int) -> Tuple[float, float]:
    # We combine SLSQP with Basinhopping for stochastic search with random steps
    f = geoy_neg(true, pred, i)
    thr0 = st_point
    fx0 = -f(np.array(thr0))
    steps = (((.1, 100), (.01, 50), (.005, 50)) if options.optmode == 1 else
             ((.1,  30), (.05, 30), (.02,  30)))
    for step, niter in steps:
        opt_output = basinhopping(f, [thr0], stepsize=step, niter=niter,
                                  minimizer_kwargs=minimizer_kwargs, accept_test=bounds, seed=30)
        fx1 = -opt_output.fun
        if fx1 > fx0 or options.optmode == 2:
            thr0 = opt_output.x.item()
            fx0  = fx1
    return thr0, fx0


# From https://github.com/mratsim/Amazon-Forest-Computer-Vision/blob/master/src/p2_validation.py
def best_f_score(true_labels: Array, predictions: Array, log: bool = False) -> List[Tuple[float, float]]:
    # Initialization of best threshold search
    constraints = [(0., 1.) for _ in cfg.model_classes]

    # Search using SLSQP, the epsilon step must be big otherwise there is no gradient
    minimizer_kwargs = {
        'method': 'SLSQP',
        'bounds': constraints,
        'options': {'eps': .01 if options.optmode == 1 else .05},
    }

    if log:
        print('==> Searching for optimal threshold for each label...', end='')

    start_time = timer()

    st_points = [.00736893 * math.e ** (.421019 * x) for x in range(11)]
    st_points.extend(1. - p for p in reversed(st_points[:-1]))

    max_threads = max(len(st_points), len(cfg.model_classes))
    cpus = os.cpu_count()
    if cpus is not None and max_threads > cpus:
        max_threads = cpus

    def best(a: List[Tuple[float, float]], b: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        return list(map(
            lambda t: max(t, key=itemgetter(1)),
            zip_strict(a, b)
        ))

    with multiprocessing.Pool(max_threads) as p:
        best_scores = reduce(
            best,
            p.starmap(
                minimize_global,
                ((true_labels, predictions, minimizer_kwargs, minimizer_bounds, st_point)
                 for st_point in st_points)
            )
        )

        minimizer_kwargs['bounds'] = [constraints[0]]

        assert len(best_scores) == len(cfg.model_classes)

        if len(cfg.model_classes) > 1:
            new_scores = p.starmap(
                refine,
                ((true_labels, predictions, minimizer_kwargs, minimizer_bounds, best_score[0], i)
                 for i, best_score in enumerate(best_scores)))

    end_time = timer()

    if log:
        print(' found in {:.2f}s'.format(end_time - start_time))

    if len(cfg.model_classes) > 1:
        for i, (best_score, new_score) in enumerate(zip_strict(best_scores, new_scores)):
            if new_score[1] > best_score[1]:
                if log:
                    print(
                        'Warning: Found a better threshold locally!'
                        ' Difference in threshold was {:.4f}'.format(
                            abs(best_score[0] - new_score[0])),
                        file=sys.stderr)
                best_scores[i] = new_score

    return best_scores


def plot_roc(true_labels: Array, predictions: Array, fig_dir: StrPath) -> None:
    # One for each label
    rocs = [roc_curve(truth, pred)[:-1] for truth, pred in zip_strict(true_labels.T, predictions.T)]

    assert options.load is not None and len(options.load) == 1
    bname = os.path.basename(*options.load)

    plt.ioff()

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    for roc, name in zip_strict(rocs, cfg.model_classes):
        plt.plot(*roc, label=name)
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.savefig(os.path.join(fig_dir, '{}_roc.svg'.format(bname)), bbox_inches='tight')


def worker_init_fn(worker_id: int) -> None:
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    np.random.seed((torch_seed + worker_id) % 2**32)

def make_dataloader(dataset: LabeledDataset) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )


def evaluate(net: NeuralNetClassifier, checkpoint: Dict[str, Any], log: bool = True) \
        -> Tuple[Array, Array, Optional[Array], Array, Optional[Array]]:
    # Use precomputed data if available
    if options.from_cpickle is not None:
        with open(options.from_cpickle, 'rb') as pf:
            pkl = pickle.load(pf)
            def load(what: str) -> Array:
                return (np.fromiter((y for y in pkl[what] if y is not None), dtype=np.float32)
                        .reshape(-1, len(cfg.model_classes)))
            return load('y_pred'), load('y_u'), None, load('y_true'), pkl['thresh']
    elif (
        options.test_with_cpickle_thr is None
        and not options.ignore_cpeval
        and checkpoint.get('opt_eval_dataset') in (None, cfg.data_cv_dir)
    ):
        try:
            return (checkpoint['opt_eval_y_pred'], checkpoint['opt_eval_y_u'], checkpoint['opt_eval_tot_ev'],
                    checkpoint['opt_eval_y_true'], None)
        except KeyError:
            pass

    start_time = None
    if log:
        print('==> Evaluating...')
        start_time = timer()

    dataset = transformed_datasets['opt' if options.test_with_cpickle_thr is None else 'test']
    preds, us, tot_ev, targets, avg_loss = eval_inner(net, dataset, checkpoint['epoch'] + 1, progress=True)

    if start_time is not None:
        end_time = timer()
        print('==> Done in {:.2f}s.'.format(end_time - start_time))
        print('==> Avg. loss: {:.4f}'.format(avg_loss))

    return preds, us, tot_ev, targets, None


def eval_inner(net: NeuralNetClassifier, dataset: LabeledDataset, epoch: int, progress: bool = False,
               leave: bool = True) -> tuple[Array, Array, Array, Array, float]:
    it: Iterator[Tensor] = net.forward_iter(dataset)
    it = tqdm(it, total=math.ceil(len(dataset) / cfg.batch_size), leave=leave) if progress else it

    alpha = torch.cat(list(it))
    predictions = pos_proba(alpha).cpu().numpy()
    uncertainties = pred_uncertainty(alpha).cpu().numpy()
    total_evidence = torch.sum(alpha - 1, dim=-1).cpu().numpy()

    y_true = torch.as_tensor(dataset.targets, device=alpha.device)
    avg_loss = net.criterion_(alpha, y_true, epoch)

    return predictions, uncertainties, total_evidence, dataset.targets, avg_loss


class Metrics:
    predictions: Array
    uncertainties: Array
    total_evidence: Optional[Array]
    true_labels: Array
    thresholds: List[float]

    def f_score_eval(self, net: NeuralNetClassifier, checkpoint: Dict[str, Any], log: bool = True) \
            -> None:
        self.predictions, self.uncertainties, self.total_evidence, self.true_labels, cpickle_thresh = \
            evaluate(net, checkpoint, log)

        if options.threshold_override is not None:
            if len(options.threshold_override) == 1:
                self.thresholds = [options.threshold_override[0] for _ in cfg.model_classes]
            else:
                self.thresholds = list(options.threshold_override)
            best_f_scores = [(th, 0.) for th in self.thresholds]
        elif options.from_cpickle is not None:
            self.thresholds = [.5 for _ in cfg.model_classes]
            best_f_scores = [(th, 0.) for th in self.thresholds]
        elif options.test_with_cpickle_thr is not None:
            with open(options.test_with_cpickle_thr, 'rb') as pf:
                self.thresholds = pickle.load(pf)['thresh']
            best_f_scores = [(th, 0.) for th in self.thresholds]
        else:
            best_f_scores = best_f_score(self.true_labels, self.predictions, log)
            # Softened thresholds to improve generalization
            self.thresholds = [(th + .5) / 2. for th, fs in best_f_scores]

        if log:
            def format_float(vals: Iterable[float]) -> Dict[str, str]:
                return dict(zip_strict(cfg.model_classes, map('{:.4f}'.format, vals)))
            def format_scores(idx: int) -> Dict[str, str]:
                return format_float(tup[idx] for tup in best_f_scores)
            thresholds, f_scores = format_scores(0), format_scores(1)
            if cpickle_thresh is not None:
                thresholds = format_float(cpickle_thresh)  # For display only

            print()
            print('==> Optimal threshold for each label:\n  {}'.format(thresholds))
            print('==> F-Score for each label:\n  {}'.format(f_scores))

            def zeros() -> List[int]: return [0 for _ in cfg.model_classes]
            true_positives, false_negatives, false_positives, true_negatives = \
                zeros(), zeros(), zeros(), zeros()

            for sample_targets, sample_preds in zip_strict(self.true_labels, self.predictions):
                for j, (label_true, label_pred) in enumerate(zip_strict(sample_targets, sample_preds)):
                    is_true = label_true > .99
                    is_pred = label_pred > best_f_scores[j][0]

                    cat = {
                        (True,  True ): true_positives,
                        (True,  False): false_negatives,
                        (False, True ): false_positives,
                        (False, False): true_negatives,
                    }[(is_true, is_pred)]

                    cat[j] += 1

            def fmt(cat: Iterable[int]) -> Dict[str, str]:
                return dict(zip_strict(cfg.model_classes, ('{:3d}'.format(l) for l in cat)))
            print('==> True positives for each label:\n  {}'.format(fmt(true_positives)))
            print('==> False negatives for each label:\n  {}'.format(fmt(false_negatives)))
            print('==> False positives for each label:\n  {}'.format(fmt(false_positives)))
            print('==> True negatives for each label:\n  {}'.format(fmt(true_negatives)))

            if self.total_evidence is not None:
                thresholds_ = np.asarray([t for t, s in best_f_scores])
                match = np.equal(self.predictions > .99, self.true_labels > thresholds_[None, :]).astype(np.float32)
                mean_evidence: Array = np.mean(self.total_evidence, axis=0)  # type: ignore[assignment]
                mean_evidence_succ = (np.sum(self.total_evidence * match, axis=0)
                                      / np.sum(match + 1e-20, axis=0))
                mean_evidence_fail = (np.sum(self.total_evidence * (1 - match), axis=0)
                                      / np.sum(1 - match + 1e-20, axis=0))
                print('==> Mean evidence for each label:\n  {}'.format(format_float(mean_evidence)))
                print('==> Mean success evidence for each label:\n  {}'.format(format_float(mean_evidence_succ)))
                print('==> Mean failure evidence for each label:\n  {}'.format(format_float(mean_evidence_fail)))

    def metrics_eval(self, net: NeuralNetClassifier, checkpoint: Dict[str, Any]) -> None:
        self.f_score_eval(net, checkpoint)

        # [(label1_true..., label1_pred...), (label2_true..., label2_pred...), ...]
        pred_data = list(zip_strict(self.true_labels.T, self.predictions.T))

        def fmt(g: Iterable[float]) -> Dict[str, str]:
            return dict(zip_strict(cfg.model_classes, ('{:.4f}'.format(fs) for fs in g)))

        mccs = (matthews_corrcoef(truth, [p > th for p in pred])
                for (truth, pred), th in zip_strict(pred_data, self.thresholds))
        print()
        print('==> MCC for each label:\n  {}'.format(fmt(mccs)))

        if options.from_cpickle is None:  # cpickle means binary y_pred, ROC AUC is not meaningful
            roc_aucs = (roc_auc_score(truth, pred) for truth, pred in pred_data)
            print('==> ROC AUC for each label:\n  {}'.format(fmt(roc_aucs)))

    def test_eval(self, net: NeuralNetClassifier, checkpoint: Dict[str, Any], log: bool = True) -> None:
        assert options.eval_dir is not None
        assert options.load is not None and len(options.load) == 1
        path = os.path.join(options.eval_dir, os.path.basename(*options.load) + '_eval.pkl')
        if os.path.exists(path):
            return  # Don't clobber it

        self.predictions, self.uncertainties, self.total_evidence, self.true_labels, _ = \
            evaluate(net, checkpoint, log)

        teval: Dict[str, Tuple[Any, Any]] = defaultdict(lambda: ([], []))
        for sample_targets, sample_preds in zip_strict(self.true_labels, self.predictions):
            for label_name, label_true, label_pred in zip_strict(cfg.model_classes, sample_targets, sample_preds):
                pred, true = teval[label_name]
                pred.append(label_pred)
                true.append(label_true)

        teval = {k: (np.asarray(p), np.asarray(t)) for k, (p, t) in teval.items()}

        with open(path, 'wb') as pf:
            pickle.dump(teval, pf)

        if log:
            print('\n==> Evaluation results saved to {!r}.'.format(path))

    def correct_eval(self, net: NeuralNetClassifier, checkpoint: Dict[str, Any], log: bool = True) -> None:
        assert options.correct_dir is not None
        assert options.load is not None and len(options.load) == 1
        path = os.path.join(options.correct_dir, os.path.basename(*options.load) + '_correct.pkl')
        if os.path.exists(path):
            return  # Don't clobber it

        self.f_score_eval(net, checkpoint)

        y_true: List[Optional[bool]] = []
        y_pred: List[Optional[bool]] = []
        y_u: List[Optional[float]] = []
        it1 = zip_strict(self.true_labels, self.uncertainties, self.predictions)
        for sample_targets, sample_uncertainties, sample_preds in it1:
            lbl_to_true: Dict[str, bool] = {}
            lbl_to_pred: Dict[str, bool] = {}
            lbl_to_uncertainty: Dict[str, float] = {}
            it2 = enumerate(zip_strict(cfg.model_classes, sample_targets, sample_preds, sample_uncertainties))
            for j, (lblname, lbltrue, lblpred, lblu) in it2:
                lbl_to_true[lblname] = lbltrue > .99
                lbl_to_pred[lblname] = lblpred > self.thresholds[j]
                lbl_to_uncertainty[lblname] = lblu
            del it2
            # Substitute with None if label not predicted
            y_true.extend(map(lbl_to_true.get, cfg.data_classes))
            y_pred.extend(map(lbl_to_pred.get, cfg.data_classes))
            y_u.extend(map(lbl_to_uncertainty.get, cfg.data_classes))
        del it1

        with open(path, 'wb') as pf:
            data = {
                'label_count': len(cfg.data_classes), 'thresh': self.thresholds,
                'y_true': y_true, 'y_pred': y_pred, 'y_u': y_u,
            }
            pickle.dump(data, pf)

        if log:
            print('\n==> Correctness pickle saved to {!r}.'.format(path))


def roc_eval(net: NeuralNetClassifier, fig_dir: StrPath, checkpoint: Dict[str, Any], log: bool = True) -> None:
    predictions, _, _, true_labels, _ = evaluate(net, checkpoint, log)

    plot_roc(true_labels, predictions, fig_dir)

    if log:
        print('\n==> ROC curve plots saved to {!r}.'.format(os.fspath(fig_dir)))


def save_random_state() -> Dict[str, Any]:
    return copy.deepcopy({
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'cuda': torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None,
    })


class TrainerCallback(Callback):
    def __init__(
        self, scheduler: Optional[_LRScheduler], warmup_scheduler: Optional[UntunedLinearWarmup], pbar: ProgressBar,
    ) -> None:
        self.scheduler = scheduler
        self.warmup_scheduler = warmup_scheduler
        self.pbar = pbar
        self.best_loss = np.inf
        self.best_state: Optional[io.BytesIO] = None
        self.start_time: Optional[float] = None

    def on_epoch_begin(self, net: NeuralNetClassifier, dataset_train: Optional[AnyDataset] = None,  # noqa: U100
                       dataset_valid: Optional[AnyDataset] = None, **kwargs: Any) -> None:
        net.history[-1]['epoch'] -= 1  # offset for pre-epoch

        self.pbar.batches_per_epoch = (
            (0 if dataset_train is None else math.ceil(get_len(dataset_train) / cfg.batch_size)) +
            math.ceil(get_len(dataset_valid) / cfg.batch_size)
        )

        if isinstance(self.scheduler, CyclicLRWithRestarts):
            assert self.warmup_scheduler is None
            self.scheduler.step()

    def on_batch_begin(self, net: NeuralNetClassifier, batch: object = None,  # noqa: U100
                       training: Optional[bool] = None, **kwargs: object) -> None:  # noqa: U100
        if training and self.warmup_scheduler is not None:
            with warnings.catch_warnings():
                # Deprecated step() usage is recommended by the warmup scheduler
                warnings.filterwarnings('ignore', '.*call them in the opposite order.*', UserWarning)
                warnings.filterwarnings('ignore', 'The epoch parameter.*', UserWarning)
                self.scheduler.step(net.history[-1]['epoch'] - 1)
            self.warmup_scheduler.dampen()

    def on_batch_end(self, net: NeuralNetClassifier, batch: object = None,  # noqa: U100
                     training: Optional[bool] = None, **kwargs: object) -> None:  # noqa: U100
        # CosineAnnealingLR-based scheduler
        if training and isinstance(self.scheduler, CyclicLRWithRestarts):
            assert self.warmup_scheduler is None
            self.scheduler.batch_step()

    def on_epoch_end(self, net: NeuralNetClassifier, dataset_train: object = None, *args: object,  # noqa: U100
                     **kwargs: object) -> None:  # noqa: U100
        if (
            dataset_train is not None
            and isinstance(self.scheduler, (StepLR, CosineAnnealingLR))
            and self.warmup_scheduler is None
        ):
            self.scheduler.step()

        if dataset_train is None:
            net.history[-1]['train_loss'] = None

        # Save the trainer state if the model is still making progress.
        if dataset_train is None or net.history[-1, 'train_loss_best'] or net.history[-1, 'valid_loss_best']:
            self.best_loss = net.history[-1, 'valid_loss']
            self._save_state_dict(net)

    def on_train_begin(self, net: NeuralNetClassifier, *args: object, **kwargs: object) -> None:  # noqa: U100
        self.start_time = timer()
        self._save_state_dict(net)
        self.best_loss = np.inf

    def on_train_end(self, net: NeuralNetClassifier, *args: object, **kwargs: object) -> None:  # noqa: U100
        assert self.start_time is not None
        time_elapsed = timer() - self.start_time
        print('==> Training complete in {:.0f}m {:.2f}s.'.format(
            time_elapsed // 60, time_elapsed % 60))
        print('==> Best validation loss: {:4f}'.format(self.best_loss))

    def _save_state_dict(self, net: NeuralNetClassifier) -> None:
        self.best_state = io.BytesIO()
        save_dict = get_save_state_dict(
            cfg=cfg,
            options=options,
            model_state_dict=net.module_.state_dict(),
            optimizer_state_dict=net.optimizer_.state_dict(),
            scheduler=self.scheduler,
            warmup_scheduler=self.warmup_scheduler,
            random_state=save_random_state(),
            net=net,
        )
        torch.save(save_dict, self.best_state, pickle_module=mypickle)
        self.best_state.seek(0)


class MyNeuralNetClassifier(NeuralNetClassifier):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.initial_epoch = kwargs.pop('initial_epoch')
        super().__init__(*args, **kwargs)

    # skorch: Don't set up the optimizer if we don't intend to use one
    def initialize_optimizer(self: AnyNNC, triggered_directly: bool = True) -> AnyNNC:
        if self.optimizer is None:
            return self
        return super().initialize_optimizer(triggered_directly=triggered_directly)

    # SciKit-Learn: Don't clone me, I'm fragile (and big)
    def __deepcopy__(self, _memo: Dict[int, object]) -> NoReturn:
        raise RuntimeError('plz no')

    # Passes epoch to get_loss
    def validation_step(self, batch: Any, **fit_params: Any) -> Dict[str, Any]:
        epoch: int = fit_params.pop('epoch')
        self.module_.eval()
        Xi, yi = unpack_data(batch)
        with torch.no_grad():
            y_pred = self.infer(Xi, **fit_params)
            loss = self._get_loss(y_pred, yi, epoch)
        return {'loss': loss, 'y_pred': y_pred}

    # Passes epoch to get_loss
    def train_step_single(self, batch: Any, **fit_params: Any) -> Dict[str, Any]:
        epoch: int = fit_params.pop('epoch')
        self.module_.train()
        Xi, yi = unpack_data(batch)
        y_pred = self.infer(Xi, **fit_params)
        loss = self._get_loss(y_pred, yi, epoch)
        loss.backward()

        return {'loss': loss, 'y_pred': y_pred}

    # Modified train_step that avoids classic zero_grad
    # See https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
    def train_step(self, batch: Any, **fit_params: Any) -> Dict[str, Any]:
        step_accumulator = self.get_train_step_accumulator()
        def step_fn() -> Tensor:
            step = self.train_step_single(batch, **fit_params)
            step_accumulator.store_step(step)
            self.notify(
                'on_grad_computed',
                named_parameters=TeeGenerator(self.module_.named_parameters()),
                batch=batch,
            )
            return step['loss']
        self.optimizer_.step(step_fn)
        for param in self.module_.parameters():
            param.grad = None
        return step_accumulator.get_step()

    # Passes the accurate epoch to run_single_epoch so loss can use it
    # Gets validation data via ds_valid
    def fit_loop(self, X: AnyDataset, y: Optional[Array] = None, epochs: Optional[int] = None, **fit_params: Any) \
            -> NeuralNetClassifier:
        self.check_data(X, y)
        epochs = epochs if epochs is not None else self.max_epochs

        dataset_train = self.get_dataset(X, y)
        dataset_valid = self.get_dataset(fit_params.pop('valid'))
        on_epoch_kwargs = {
            'dataset_train': dataset_train,
            'dataset_valid': dataset_valid,
        }

        for epoch in range(self.initial_epoch, self.initial_epoch + epochs):
            fit_params['epoch'] = epoch

            self.notify('on_epoch_begin', **on_epoch_kwargs)

            self.run_single_epoch(dataset_train, training=True, prefix='train',
                                  step_fn=self.train_step, **fit_params)

            if dataset_valid is not None:
                self.run_single_epoch(dataset_valid, training=False, prefix='valid',
                                      step_fn=self.validation_step, **fit_params)

            self.notify('on_epoch_end', **on_epoch_kwargs)
        return self

    def get_loss(self, y_pred: Tensor, y_true: Tensor, *args: object, **kwargs: object) -> NoReturn:  # noqa: U100
        raise NotImplementedError('not used by MyNeuralNetClassifier')

    # Passes epoch to criterion
    def _get_loss(self, y_pred: Tensor, y_true: Tensor, epoch: int) -> Tensor:
        y_true = to_tensor(y_true, device=self.device)
        return self.criterion_(y_pred, y_true, epoch)


class MyPrintLog(PrintLog):
    KEY_ORDER = ('epoch', 'train_loss', 'valid_loss', 'valid_mcc', 'valid_acc', 'dur')

    # User-defined key order
    def _sorted_keys(self, keys: object) -> list[str]:
        skeys = super()._sorted_keys(keys)
        if set(skeys) != set(self.KEY_ORDER):
            raise ValueError(f'Expected keys: {self.KEY_ORDER}\nGot keys: {skeys}')
        return list(self.KEY_ORDER)


class MyAdaBelief(AdaBelief):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault('eps', 1e-8)
        kwargs.setdefault('rectify', False)
        super().__init__(*args, **kwargs)


class MySGDW(SGDW):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.setdefault('momentum', .9)
        super().__init__(*args, **kwargs)


class MyReduceMaxLROnRestart:
    def __init__(self, ratio: float, period: int = 1) -> None:
        self.ratio = ratio
        self.period = period
        self.iter = 0

    def __call__(self, eta_min: float, eta_max: float) -> tuple[float, float]:
        self.iter += 1
        if self.iter >= self.period:
            eta_max *= self.ratio
            self.iter = 0
        return eta_min, eta_max


def isclose_nested(a: Union[Tuple[Any, ...], float], b: Union[Tuple[Any, ...], float]) -> bool:
    if isinstance(a, tuple):
        assert isinstance(b, tuple)
        assert len(a) == len(b)
        return all(isclose_nested(aa, bb) for aa, bb in zip_strict(a, b))
    assert isinstance(a, Real) and isinstance(b, Real)
    return math.isclose(a, b)


class ResampleArgs:
    def __init__(self, algorithm: Optional[str], kwargs: Dict[str, Any]) -> None:
        self.algorithm = algorithm
        self.kwargs = kwargs

    @classmethod
    def parse(cls: Type[AnyResampleArgs], s: str) -> AnyResampleArgs:
        if s == 'none':
            return cls(None, {})
        m, s = s[0], s[1:]
        if m not in '+-':
            raise ValueError("Resample: Expected mode of '+' or '-', got {!r}".format(m))
        alg = 'ML-ROS' if m == '+' else 'ML-RUS'
        kwargs: Dict[str, Any] = {}
        if not s:
            return cls(alg, kwargs)
        args = s.split(':')
        if len(args) > 3:
            raise ValueError('Resample: Expected at most 3 arguments, got {}'.format(len(args)))
        it = iter(args)
        try:
            if arg := next(it):
                kwargs['resample_limit'] = float(arg) / 100
            if arg := next(it):
                kwargs['imbalance_target'] = float(arg)
            if arg := next(it):
                if arg not in ('pos', 'all'):
                    raise ValueError("Resample: Expected imbalance mode of 'pos' or 'all', got {!r}".format(arg))
                kwargs['mode'] = arg
        except StopIteration:
            pass
        return cls(alg, kwargs)


class ResampledLabeledSubset(LabeledDataset):
    def __init__(self, dataset: LabeledDataset, indices: Optional[Sequence[int]],  # pytype: disable=module-attr
                 rand: np.random.RandomState) -> None:
        self.dataset = dataset
        self.indices = indices
        self.rand = rand

    @classmethod
    def new(cls: Type[AnyRLS], dataset: LabeledDataset) -> AnyRLS:
        inst = cls(
            dataset, None,
            np.random.RandomState(np.random.randint(0, 2**32)),  # pytype: disable=module-attr
        )
        inst._init()
        return inst

    @classmethod
    def load(cls: Type[AnyRLS], dataset: LabeledDataset,  # pytype: disable=module-attr
             indices: Optional[Sequence[int]], rand: np.random.RandomState) -> AnyRLS:
        return cls(dataset, indices, rand)

    def __getitem__(self, index: int) -> Tuple[Any, Tensor]:
        if self.indices is None:
            return self.dataset[index]
        return self.dataset[self.indices[index]]

    def __len__(self) -> int:
        return len(self.dataset) if self.indices is None else len(self.indices)

    @property
    def labelset(self) -> Sequence[Set[str]]:
        if self.indices is None:
            return self.dataset.labelset
        lset = self.dataset.labelset
        return tuple(lset[i] for i in self.indices)

    @property
    def targets(self) -> Array:
        if self.indices is None:
            return self.dataset.targets
        return self.dataset.targets[list(self.indices)]

    @property
    def groups(self) -> Array:
        if self.indices is None:
            return self.dataset.groups
        return self.dataset.groups[list(self.indices)]

    def _init(self) -> None:
        if cfg.resample.algorithm is not None:
            self.indices = tuple(ImbalancedDatasetSampler(
                self.dataset.labelset, ds_labels=cfg.model_classes,
                algorithm=cfg.resample.algorithm,
                alg_kwargs=cfg.resample.kwargs,
                rand=self.rand,
            ))


class Config:
    def __init__(self, seed_value: int) -> None:
        self.annealing_step:      int             = DEFAULT_ANNEALING_STEP
        self.base_model:          Optional[Union[str, Tuple[str, ...]]] = None
        self.batch_size:          int             = DEFAULT_BATCH_SIZE
        self.building_ensemble:   bool            = False
        self.building_model:      bool            = False
        self.classifier_dropout:  float           = .5
        self.cont_opt:            bool            = False
        self.cont_sch:            bool            = False
        self.cpload:              bool            = False
        self.criterion:           str             = ''
        self.data_classes:        tuple[str, ...] = ()
        self.data_cv_dir:         str             = ''
        self.epochs:              int             = DEFAULT_EPOCHS
        self.initial_epoch:       Any             = None
        self.inner_dropout:       Union[float, Tuple[float, ...]] = 0.
        self.jpeg_iterations:     int             = 3
        self.jpeg_quality:        Tuple[int, int] = (90, 98)
        self.load_sublayer_ratio: Optional[Union[float, Tuple[float, float], Sequence[Tuple[float, float]]]] = None
        self.lr_warmup:           bool            = False
        self.model_classes:       tuple[str, ...] = ()
        self.model_features:      Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]] = None
        self.need_train_data:     bool            = False
        self.noise_factor_1:      float           = 1 / 56
        self.noise_factor_2:      float           = 1 / 35
        self.num_workers:         int             = DEFAULT_NUM_WORKERS
        self.optimizer_kwargs:    Dict[str, Any]  = {}
        self.optimizer_type:      Type[Optimizer] = UNTYPED_NONE
        self.resample:            ResampleArgs    = ResampleArgs.parse('+50')
        self.pepper_factor:       float           = 0.
        self.seed_value = seed_value
        self.scheduler_kwargs:    Optional[Dict[str, Any]] = None
        self.scheduler_name:      str             = ''
        self.scheduler_type:      Optional[Type[_LRScheduler]] = None
        self.sublayer_ratio:      Optional[Union[float, Tuple[float, float], Tuple[Tuple[float, float], ...]]] = None
        self.training:            bool            = False
        self.tta_mode:            str             = ''
        self.virtual_batch_size:  Optional[int]   = None
        self.weight_decay:        Optional[float] = None


def get_save_state_dict(
    *,
    cfg: 'Config',
    options: 'MainArgParser',
    model_state_dict: Dict[str, Any],
    optimizer_state_dict: Dict[str, Any],
    scheduler: _LRScheduler,
    warmup_scheduler: Optional[UntunedLinearWarmup],
    random_state: Dict[str, Any],
    net: NeuralNetClassifier,
) -> Dict[str, Any]:
    model = net.module_
    old_training = model.training
    model.train()
    save_dict = {
        'task': options.task,
        'base_model': cfg.base_model,
        'data_classes': cfg.data_classes,
        'out_classes': cfg.model_classes,
        'model_features': cfg.model_features,
        'batch_size': cfg.batch_size,
        'virtual_batch_size': cfg.virtual_batch_size,
        'criterion': cfg.criterion,
        'annealing_step': cfg.annealing_step,
        'resample': vars(cfg.resample),
        'epoch': cfg.initial_epoch + cfg.epochs - 1,
        'inner_dropout': cfg.inner_dropout,
        'classifier_dropout': cfg.classifier_dropout,
        'noise_factors': (cfg.noise_factor_1, cfg.noise_factor_2),
        'jpeg_quality': cfg.jpeg_quality,
        'jpeg_iterations': cfg.jpeg_iterations,
        'optimizer_type': cfg.optimizer_type,
        'optimizer_wd': cfg.weight_decay,
        'optimizer_kwargs': cfg.optimizer_kwargs,
        'scheduler_type': cfg.scheduler_type,
        'scheduler_kwargs': None if cfg.scheduler_type is None else cfg.scheduler_kwargs,
        'lr_warmup': cfg.lr_warmup,
        'seed_value': cfg.seed_value,
        'tta_mode': cfg.tta_mode,
        'load_sublayer_ratio': cfg.load_sublayer_ratio,
        'sublayer_ratio': cfg.sublayer_ratio,
        'params_trained': {n: p.requires_grad for n, p in model.named_parameters()},
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer_state_dict,
        'scheduler': scheduler,
        'warmup_state': None if warmup_scheduler is None else warmup_scheduler.state_dict(),
        'random_state': random_state,
        'history': net.history_,
        'virtual_params': net.virtual_params_,
    }
    model.train(old_training)
    return save_dict


def get_save_stats_dict(
    *,
    cfg: 'Config',
    model: Module,
    data_transforms: Dict[str, transforms.Compose],
    opt_eval_y_true: Array,
    opt_eval_y_pred: Array,
    opt_eval_y_u: Array,
    opt_eval_tot_ev: Array,
    test_loss: float
) -> Dict[str, Any]:
    script_dir = os.path.dirname(os.path.realpath(__file__))

    def gitcmd(*args: Any) -> str:
        p = subprocess.run(('git', '-C', script_dir, *args), check=True, capture_output=True, text=True)
        return p.stdout.rstrip('\n')

    old_training = model.training
    model.train()
    save_dict = {
        'cmdline': tuple(sys.argv),
        'modules': tuple((nm, getattr(mod, '__version__', None), getattr(mod, '__path__', None))
                         for nm, mod in sorted(sys.modules.items(), key=lambda x: x[0])),
        'timestamp': time.time(),
        'revision': 'r{}.{}'.format(gitcmd('rev-list', '--count', 'HEAD'),
                                    gitcmd('rev-parse', '--short', 'HEAD')),
        'data_transforms': data_transforms,
        'opt_eval_dataset': cfg.data_cv_dir,
        'opt_eval_y_true': opt_eval_y_true,
        'opt_eval_y_pred': opt_eval_y_pred,
        'opt_eval_y_u': opt_eval_y_u,
        'opt_eval_tot_ev': opt_eval_tot_ev,
        'test_loss': test_loss,
    }
    model.train(old_training)
    return save_dict


# Adapted from timm
def add_weight_decay(model: Module, weight_decay: float, skip_list: Container[str]) -> List[Dict[str, Any]]:
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith('.bias') or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}
    ]


def get_valid_mcc(net: NeuralNetClassifier, X: Tensor, y_true: Array) -> float:
    with torch.no_grad():
        predictions = pos_proba(net.forward(X)).cpu().numpy()

    best_f_scores = best_f_score(y_true, predictions, log=False)
    # Softened thresholds to improve generalization
    thresholds = [(th + .5) / 2. for th, fs in best_f_scores]

    # [(label1_true..., label1_pred...), (label2_true..., label2_pred...), ...]
    pred_data = list(zip_strict(y_true.T, predictions.T))

    return np.mean([
        matthews_corrcoef(truth, [p > th for p in pred])
        for (truth, pred), th in zip_strict(pred_data, thresholds)
    ])


def check_cond(typ: Callable[[str], T], cond: Callable[[T], bool], name: str) -> Callable[[str], T]:
    def parse(value: str) -> Any:
        tv = typ(value)
        if not cond(tv):
            raise ValueError('Value must be {}'.format(name))
        return tv
    return parse


def tuple_arg(elt_typ: Callable[[str], T], sep: str = ',') -> Callable[[str], Tuple[T, ...]]:
    def parse(value: str) -> Tuple[Any, ...]:
        return tuple(elt_typ(e) for e in value.split(sep))
    return parse


positive_int = check_cond(int, lambda v: v > 0, 'positive')
nonnegative_int = check_cond(int, lambda v: v >= 0, 'nonnegative')
positive_float = check_cond(float, lambda v: v > 0, 'positive')
nonnegative_float = check_cond(float, lambda v: v >= 0, 'nonnegative')
nonnegative_floatexpr = check_cond(lambda s: float(eval(s)), lambda v: v >= 0, 'nonnegative')  # noqa: S307
ratio_float = check_cond(float, lambda v: 0 <= v <= 1, 'between zero and one, inclusive')


class MainArgParser(Tap):
    task: Literal['preview_input', 'print_numels', 'find_aug', 'train', 'visualize', 'metrics', 'eval_test',
                  'get_correct', 'roc']
    """What to do with the neural network"""

    data_dir: Optional[str] = None  # Dataset location
    cv_fold: Optional[int] = None  # Fold of cross validator to use
    load: Optional[Tuple[str, ...]] = None  # type: ignore[assignment]
    save: Optional[str] = None  # type: ignore[assignment]
    resample: Optional[str] = None  # Pre-split resample arguments
    noise_factors: Optional[Tuple[float, ...]] = None  # Data augmentation: Noise factors
    pepper_factor: Optional[float] = None  # Data augmentation: Pepper factor
    jpeg_quality: Optional[Tuple[int, int]] = None  # Data augmentation: JPEG quality range
    jpeg_iterations: Optional[int] = None  # Data augmentation: JPEG iterations

    forget_state: List[Literal['model', 'frozen', 'optimizer', 'scheduler']] = []
    """Ignore one or more parts of a checkpoint"""

    base_model: Optional[Tuple[str, ...]] = None  # Name of pretrained base model
    batch_size: Optional[int] = None  # Size of data batches
    class_filter: Optional[Tuple[str, ...]] = None  # Learn the specified subset of classes
    criterion: Optional[Literal['mse', 'digamma', 'log']] = None  # Which evidential loss function to use
    annealing_step: Optional[int] = None  # Annealing step for loss function
    sublayers: Optional[Tuple[Tuple[float, float], ...]] = None  # Ratio of sublayers to unfreeze, 0=none, 1=all
    freeze_fc: bool = False  # Freeze the final classifier layer
    inner_dropout: Optional[Tuple[float, ...]] = None  # Apply inner dropout with probability P
    classifier_dropout: Optional[float] = None  # Apply classifier dropout with probability P
    epochs: Optional[int] = None  # Train for N epochs
    scheduler: Optional[Literal['cyclic', 'cosine', 'linear', 'NONE']] = None  # Which type of LR scheduler to use

    schpolicy: Optional[Literal['cosine', 'arccosine', 'triangular', 'triangular2', 'exp_range']] = None
    """Policy for cyclic scheduler"""

    lr_warmup: bool = False  # Use LR warmup

    optimizer: Optional[Literal['adam', 'aadamw01', 'adabelief', 'sgdw', 'lamb']] = None
    """Which type of optimizer to use"""

    lr: Optional[float] = None  # Learning rate
    gamma: Optional[float] = None  # Gamma parameter of scheduler
    gamma_override: Optional[float] = None  # Override restart gamma
    sch_period: Optional[int] = None  # Scheduler period in epochs
    wd: Optional[float] = None  # Weight decay
    resnet_d: bool = False  # Apply Resnet-D to trained sublayers
    l1reg: Optional[float] = None  # L1 regularization factor
    l2reg: Optional[float] = None  # L2 regularization factor
    corrreg: Optional[float] = None  # Correlation regularization factor
    virtual_batch_size: Optional[int] = None  # Size of virtual "ghost" batches
    optmode: Literal[1, 2] = 1  # Use a different mode for threshold searching
    threshold_override: Optional[Tuple[float, ...]] = None  # Use given thresholds instead of searching
    from_cpickle: Optional[str] = None  # Load correctness pickle from FILE and use it instead of evaluating
    test_with_cpickle_thr: Optional[str] = None  # Load thresholds from cpickle FILE and do test eval with them
    fig_dir: Optional[str] = None  # Where to store plotted figures
    eval_dir: Optional[str] = None  # Where to store eval pickles
    correct_dir: Optional[str] = None  # Where to store correctness pickles
    quick_find: bool = False  # (find_aug) Validate on only 500 samples, and train on 250

    tta: Optional[Literal['none', 'mean', 'local_certainty', 'global_certainty']] = None
    """Test time augmentation mode"""

    ignore_cpeval: bool = False  # Ignore eval results stored in checkpoint
    seed: int = DEFAULT_SEED  # Value to seed random state with

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.use_cuda = True

    def configure(self) -> None:
        self.add_argument('--data-dir', metavar='DIR')
        self.add_argument('--cv-fold', metavar='N', type=nonnegative_int)
        self.add_argument('--load', type=tuple_arg(str, sep=';'), metavar='FILE', help='Load a checkpoint from FILE')
        self.add_argument('--save', metavar='FILE', help='Save a checkpoint to FILE')
        self.add_argument('--resample', metavar='ARGS')
        self.add_argument('--noise-factors', type=tuple_arg(nonnegative_floatexpr), metavar='FACTORS')
        self.add_argument('--pepper-factor', type=nonnegative_floatexpr, metavar='FACTOR')
        self.add_argument('--jpeg-quality', type=tuple_arg(nonnegative_int), metavar='RANGE')
        self.add_argument('--jpeg-iterations', type=nonnegative_int, metavar='N')
        self.add_argument('--forget-state', action='append')
        self.add_argument('--base-model', type=tuple_arg(str, sep=';'), metavar='MODEL')
        self.add_argument('--batch-size', type=positive_int, metavar='N')
        self.add_argument('--class-filter', type=tuple_arg(str), metavar='CLASSES')
        self.add_argument('--annealing-step', metavar='N')
        self.add_argument('--sublayers', type=tuple_arg(tuple_arg(nonnegative_float), sep=';'), metavar='RATIO')
        self.add_argument('--inner-dropout', type=tuple_arg(ratio_float, sep=';'), metavar='P')
        self.add_argument('--classifier-dropout', type=ratio_float, metavar='P')
        self.add_argument('--epochs', type=nonnegative_int, metavar='N')
        self.add_argument('--lr', type=positive_float)
        self.add_argument('--gamma', type=positive_float, metavar='γ')
        self.add_argument('--gamma-override', type=positive_float, metavar='γ')
        self.add_argument('--sch-period', type=positive_int, metavar='N')
        self.add_argument('--wd', type=ratio_float)
        self.add_argument('--l1reg', type=positive_float, metavar='FACTOR')
        self.add_argument('--l2reg', type=positive_float, metavar='FACTOR')
        self.add_argument('--corrreg', type=positive_float, metavar='FACTOR')
        self.add_argument('--virtual-batch-size', metavar='N')
        self.add_argument('--threshold-override', type=tuple_arg(positive_float), metavar='THRESHOLDS')
        self.add_argument('--from-cpickle', metavar='FILE')
        self.add_argument('--test-with-cpickle-thr', metavar='FILE')
        self.add_argument('--fig-dir', metavar='DIR')
        self.add_argument('--eval-dir', metavar='DIR')
        self.add_argument('--correct-dir', metavar='DIR')
        self.add_argument('--no-cuda', action='store_false', dest='use_cuda', help='Disable CUDA')


def main() -> None:
    global cfg, device, options, transformed_datasets

    # Filter a SkorchWarning about on_batch_{begin,end} signature
    warnings.filterwarnings('ignore', category=SkorchWarning)

    disable_echo()

    parser = MainArgParser(underscores_to_dashes=True)
    options = parser.parse_args()

    cfg = Config(seed_value=options.seed)
    seed_all(cfg.seed_value)

    # The number of requested models. Note: This may be one model which has submodels.
    if options.load is not None:
        if options.base_model is not None:
            parser.error('--base-model conflicts with --load')
        req_models = len(options.load)
    elif options.base_model is not None:
        req_models = len(options.base_model)
    else:
        req_models = 1
    assert options.sublayers is None or len(options.sublayers) == req_models
    assert options.inner_dropout is None or len(options.inner_dropout) == req_models

    train_tasks = ('find_aug', 'train')
    cfg.training = options.task in train_tasks
    cfg.need_train_data = options.task in train_tasks + ('preview_input',)
    cfg.cpload = options.load is not None and req_models == 1
    noweight_tasks = ('preview_input', 'print_numels')
    cfg.building_model = not (cfg.cpload or options.task in noweight_tasks)
    cfg.building_ensemble = options.load is not None and not cfg.cpload

    # Training-only options
    if options.task == 'train':
        cfg.epochs = DEFAULT_EPOCHS if options.epochs is None else options.epochs
    else:
        if options.epochs is not None:
            parser.error('--epochs is only valid if training')
        if options.save is not None:
            parser.error('--save is only valid if task is train')

    # Other restrictions
    if options.criterion is None and cfg.building_model:
        parser.error('--criterion must be specified if building a model')
    if options.annealing_step is not None and not cfg.building_model:
        parser.error('--annealing_step is only valid if building a model')

    required_cpload_tasks = ('visualize', 'metrics', 'roc', 'eval_test', 'get_correct')
    valid_cpload_tasks = required_cpload_tasks + train_tasks
    valid_ensemble_tasks = train_tasks
    valid_load_tasks = valid_cpload_tasks + valid_ensemble_tasks
    if options.load is not None and options.task not in valid_load_tasks:
        parser.error('--load is only valid for tasks: {}'.format(', '.join(valid_load_tasks)))
    if cfg.cpload and options.task not in valid_cpload_tasks:
        parser.error('Checkpoint loading is only valid for tasks: {}'.format(', '.join(valid_cpload_tasks)))
    if not cfg.cpload and options.task in required_cpload_tasks:
        parser.error('Checkpoint loading (--load) is required for tasks: {}'.format(', '.join(required_cpload_tasks)))
    if cfg.building_ensemble and options.task not in valid_ensemble_tasks:
        parser.error('Ensemble building is only valid for tasks: {}'.format(', '.join(valid_ensemble_tasks)))
    assert (options.load is not None) == cfg.cpload or cfg.building_ensemble

    if options.quick_find and options.task != 'find_aug':
        parser.error('--quick-find is only valid for --task find_aug')
    if options.forget_state and not cfg.cpload:
        parser.error('--forget-state is only valid if loading a checkpoint')
    if options.forget_state is None:
        options.forget_state = []
    if (options.fig_dir is not None) != (options.task == 'roc'):
        parser.error('--fig-dir must be specified iff task=roc')
    if (options.eval_dir is not None) != (options.task == 'eval_test'):
        parser.error('--eval-dir must be specified iff task=eval_test')
    if (options.correct_dir is not None) != (options.task == 'get_correct'):
        parser.error('--correct-dir must be specified iff task=get_correct')
    if cfg.cpload and options.class_filter is not None:
        parser.error('--class-filter must not be specified if loading a checkpoint')
    if sum(1 for opt in ('optmode', 'threshold_override', 'from_cpickle', 'test_with_cpickle_thr')
           if getattr(options, opt) != parser.get_default(opt)) > 1:
        parser.error(
            'Only one of --optmode, --threshold-override, --from-cpickle, or --test-with-cpickle-thr may be given')
    valid_cpickle_tasks = ('visualize', 'metrics', 'roc')
    if options.from_cpickle is not None and options.task not in valid_cpickle_tasks:
        parser.error('--from-cpickle is only valid for tasks: {}'.format(valid_cpickle_tasks))
    valid_cpickle_thr_tasks = valid_cpickle_tasks + ('get_correct',)
    if options.test_with_cpickle_thr is not None and options.task not in valid_cpickle_thr_tasks:
        parser.error('--test-with-cpickle-thr is only valid for tasks: {}'.format(valid_cpickle_thr_tasks))

    # Whether to continue with same optimizer or scheduler
    cfg.cont_opt = cfg.cpload and cfg.training
    cfg.cont_sch = cfg.cpload and options.task == 'train'
    if 'optimizer' in options.forget_state:
        if not cfg.cont_opt:
            parser.error('--forget-state=optimizer passed but optimizer state would not be loaded anyway')
        cfg.cont_opt = False
    if 'scheduler' in options.forget_state:
        if not cfg.cont_sch:
            parser.error('--forget-state=scheduler passed but scheduler state would not be loaded anyway')
        cfg.cont_sch = False

    # Checkpoint-related argument checks
    if options.base_model is not None and options.load is not None:
        parser.error('--base-model must not be specified if a checkpoint will be loaded')

    # TODO: loop
    if options.batch_size is not None and cfg.cpload:
        parser.error('--batch-size must not be specified if a checkpoint will be loaded')
    if options.virtual_batch_size is not None and cfg.cpload:
        parser.error('--virtual-batch-size must not be specified if a checkpoint will be loaded')

    if options.optimizer is not None and (not cfg.training or cfg.cont_opt):
        parser.error('--optimizer is only valid if training and optimizer state will not be loaded')
    if (options.task != 'train' or cfg.cont_sch):
        for opt in ('scheduler', 'gamma', 'sch-period'):
            if getattr(options, opt.replace('-', '_')) is not None:
                parser.error('--{} is only valid if training and scheduler state will not be loaded'.format(opt))
    if options.lr_warmup:
        if options.task != 'train' or cfg.cpload:
            parser.error('--lr-warmup is only valid if training and a checkpoint will not be loaded')
        assert DEFAULT_SCHEDULER != 'NONE'
        if not cfg.cont_sch and options.scheduler == 'NONE':
            parser.error('--lr-warmup requires an LR scheduler')
    if (options.lr is not None) != (options.task in ('train', 'find_aug') and not cfg.cont_sch):
        parser.error('--lr must be specified iff training and scheduler state will not be loaded')
    if options.task == 'train':
        # (default scheduler is linear)
        if (options.gamma is not None) != (options.scheduler in (None, 'linear', 'cyclic') and not cfg.cont_sch):
            parser.error('--gamma must be specified iff creating a new linear or cyclic scheduler')
    if options.schpolicy is not None and options.scheduler != 'cyclic':
        parser.error('--schpolicy is only valid if creating a new cyclic scheduler')
    if options.gamma_override is not None and not cfg.cont_sch:
        parser.error('--gamma-override is only valid if scheduler state will be loaded')
    if options.sublayers is not None and not cfg.training:
        parser.error('--sublayers is only valid if training')
    if options.base_model is None and (options.load is None and options.task != 'preview_input'):
        parser.error('--base-model is required if training a new model')

    if options.sublayers is not None:
        def endify(sr: Tuple[float, ...]) -> Tuple[float, float]:
            if len(sr) > 2:
                raise ValueError('Bad value for --sublayers: {}'.format(options.sublayers))
            return sr if len(sr) == 2 else (0, *sr)  # type: ignore[return-value]
        options.sublayers = tuple(map(endify, options.sublayers))

    if cfg.cpload:
        assert options.load is not None
        checkpoint = torch.load(*options.load)
        sub_cps = None

        cpbm: Union[str, Tuple[str, ...]] = checkpoint['base_model']
        cfg.base_model = cpbm if isinstance(cpbm, tuple) else (cpbm,)
        assert cfg.base_model

        cfg.criterion = checkpoint['criterion']
        cfg.annealing_step = checkpoint.get('annealing_step', DEFAULT_ANNEALING_STEP)

        cfg.sublayer_ratio = parse_sublayer_ratio(checkpoint['sublayer_ratio'], cfg.base_model)
        assert cfg.sublayer_ratio is not None
        if options.sublayers is not None:
            if isclose_nested(options.sublayers, cfg.sublayer_ratio):
                print('\x1b[93;1mWarning: --sublayers unnecessarily specified\x1b[0m', file=sys.stderr)
            elif 'frozen' not in options.forget_state:
                parser.error('Overriding sublayer ratio requires --forget-state frozen')
            else:
                print('==> Overriding sublayer ratio: {} -> {}'.format(cfg.sublayer_ratio, options.sublayers))
                cfg.sublayer_ratio = options.sublayers
    else:
        checkpoint = None

        if not cfg.training:
            cfg.sublayer_ratio = None
        elif options.sublayers is not None:
            cfg.sublayer_ratio = options.sublayers
        elif req_models > 1:
            cfg.sublayer_ratio = tuple(DEFAULT_SUBLAYERS for _ in range(req_models))
        else:
            cfg.sublayer_ratio = DEFAULT_SUBLAYERS

        if cfg.building_ensemble:
            assert req_models > 1
            assert isinstance(options.load, tuple) and len(options.load) == req_models
            sub_cps = tuple(torch.load(p) for p in options.load)
            cfg.base_model = None
        else:
            sub_cps = None
            cfg.base_model = options.base_model
        if options.task not in ('preview_input', 'print_numels'):
            assert options.criterion is not None
            cfg.criterion = options.criterion
            cfg.annealing_step = DEFAULT_ANNEALING_STEP if options.annealing_step is None else options.annealing_step

    if options.data_dir is not None and options.cv_fold is not None:
        opts_cv_dir = os.path.join(options.data_dir, 'fold{}'.format(options.cv_fold))
    elif options.data_dir is not None or options.cv_fold is not None:
        parser.error('--data-dir and --cv-fold must be specified together')
    else:
        opts_cv_dir = None

    if checkpoint is None:
        if opts_cv_dir is None:
            parser.error('--data-dir and --cv-fold must be specified unless loading a checkpoint')
        assert opts_cv_dir is not None
        cfg.data_cv_dir = opts_cv_dir
    else:
        cfg.data_cv_dir = checkpoint['opt_eval_dataset']
        if opts_cv_dir is not None:
            if opts_cv_dir == cfg.data_cv_dir:
                print('\x1b[93;1mWarning: --data-dir and --cv-fold unnecessarily specified\x1b[0m', file=sys.stderr)
            else:
                print('==> Overriding dataset: {!r} -> {!r}'.format(cfg.data_cv_dir, opts_cv_dir))
                cfg.data_cv_dir = opts_cv_dir

    if checkpoint is not None:
        cfg.batch_size = checkpoint['batch_size']
        cfg.virtual_batch_size = checkpoint.get('virtual_batch_size')
    else:
        cfg.batch_size = DEFAULT_BATCH_SIZE if options.batch_size is None else options.batch_size
        cfg.virtual_batch_size = options.virtual_batch_size

        if cfg.virtual_batch_size is not None and cfg.batch_size % cfg.virtual_batch_size != 0:
            parser.error('Batch size ({}) must be evenly divisible by --virtual-batch-size ({})'.format(
                cfg.batch_size, cfg.virtual_batch_size,
            ))

    if options.load is not None:
        if checkpoint is not None:
            cfg.inner_dropout = checkpoint['inner_dropout']
        else:
            assert sub_cps is not None
            cfg.inner_dropout = tuple(cp['inner_dropout'] for cp in sub_cps)

        if options.inner_dropout is not None:
            if isclose_nested(options.inner_dropout, cfg.inner_dropout):
                print('\x1b[93;1mWarning: --inner-dropout unnecessarily specified\x1b[0m', file=sys.stderr)
            else:
                print('==> Overriding inner dropout: {} -> {}'.format(cfg.inner_dropout, options.inner_dropout))
                cfg.inner_dropout = options.inner_dropout
    elif options.inner_dropout is not None:
        cfg.inner_dropout = options.inner_dropout
        if len(cfg.inner_dropout) == 1:
            cfg.inner_dropout, = cfg.inner_dropout
    else:
        cfg.inner_dropout = 0. if req_models == 1 else tuple(0. for _ in range(req_models))

    if cfg.cpload:
        cfg.classifier_dropout = checkpoint['classifier_dropout']

        if options.classifier_dropout is not None:
            if math.isclose(options.classifier_dropout, cfg.classifier_dropout):
                print('\x1b[93;1mWarning: --classifier-dropout unnecessarily specified\x1b[0m', file=sys.stderr)
            else:
                print('==> Overriding classifier dropout: {} -> {}'.format(
                    cfg.classifier_dropout, options.classifier_dropout,
                ))
                cfg.classifier_dropout = options.classifier_dropout
    else:
        cfg.classifier_dropout = 0. if options.classifier_dropout is None else options.classifier_dropout

    if not cfg.need_train_data:
        for opt in ('resample', 'noise-factors', 'jpeg-quality', 'jpeg-iterations', 'pepper-factor'):
            if getattr(options, opt.replace('-', '_')) is not None:
                parser.error('--{} is only valid if training data will be used'.format(opt))
    elif checkpoint is not None:
        # TODO: Support override of these parameters
        cfg.resample = ResampleArgs(**checkpoint['resample'])
        if checkpoint.get('infl_pct') is not None:
            print("\x1b[93;1mWarning: Ignoring non-None 'infl_pct' in checkpoint\x1b[0m", file=sys.stderr)
        noise_factors: Tuple[float, float] = checkpoint['noise_factors']
        cfg.noise_factor_1, cfg.noise_factor_2 = noise_factors
        del noise_factors
        cfg.jpeg_quality = checkpoint.get('jpeg_quality', (90, 98))
        cfg.jpeg_iterations = checkpoint.get('jpeg_iterations', 3)
        cfg.pepper_factor = checkpoint.get('pepper_factor', 0)
    else:
        if options.resample is not None:
            cfg.resample = ResampleArgs.parse(options.resample)
        cfg.noise_factor_1, cfg.noise_factor_2 = \
            (1 / 56, 1 / 35) if options.noise_factors is None else options.noise_factors
        cfg.jpeg_quality = (90, 98) if options.jpeg_quality is None else options.jpeg_quality
        cfg.jpeg_iterations = 3 if options.jpeg_iterations is None else options.jpeg_iterations
        cfg.pepper_factor = 0 if options.pepper_factor is None else options.pepper_factor

    # Model features that cannot be overridden
    if (options.load or not cfg.training) and options.resnet_d not in (None, False):
        parser.error('--resnet-d is only valid if training a new model')
    if options.load:
        cfg.model_features = None if checkpoint is None else checkpoint['model_features']
    else:
        cfg.model_features = {}
        if options.resnet_d:
            cfg.model_features['resnet-d'] = True
        if req_models > 1 and cfg.model_features is not None:
            assert cfg.base_model is not None

            def feat_supp(feat: str, model: str) -> bool:
                if feat == 'resnet-d':
                    return model_supports_resnet_d(model)
                return True

            cfg.model_features = tuple({k: v for k, v in cfg.model_features.items() if feat_supp(k, model)}
                                       for model in cfg.base_model)

    if cfg.cpload and cfg.training and cfg.cont_opt:
        cfg.weight_decay = checkpoint['optimizer_wd']
        if options.wd is not None:
            if cfg.weight_decay is not None and math.isclose(options.wd, cfg.weight_decay):
                print('\x1b[93;1mWarning: --wd unnecessarily specified\x1b[0m', file=sys.stderr)
            else:
                print('==> Overriding weight decay: {} -> {}'.format(cfg.weight_decay, options.wd))
                cfg.weight_decay = options.wd
    else:
        cfg.weight_decay = options.wd

    if cfg.cont_sch and options.gamma_override is not None \
            and not issubclass(checkpoint['scheduler_type'], CyclicLRWithRestarts):
        parser.error('--gamma-override requires a cyclic scheduler')

    if options.task == 'train':
        cfg.initial_epoch = 0 if checkpoint is None else checkpoint['epoch'] + 1

    if checkpoint is not None:
        cfg.tta_mode = checkpoint.get('tta_mode', 'none')
        if options.tta is not None:
            if options.tta == cfg.tta_mode:
                print('\x1b[93;1mWarning: --tta unnecessarily specified\x1b[0m', file=sys.stderr)
            else:
                print('==> Overriding TTA mode: {} -> {}'.format(cfg.tta_mode, options.tta))
                cfg.tta_mode = options.tta
    else:
        cfg.tta_mode = 'none' if options.tta is None else options.tta

    print("==> Dataset: {!r}".format(cfg.data_cv_dir))

    data_transforms = {}
    if checkpoint is not None:
        data_transforms = checkpoint['data_transforms']
        # Remove any stray normalize transforms; we don't use these anymore
        for _, dt in data_transforms.items():
            while isinstance(dt.transforms[-1], transforms.Normalize):
                dt.transforms.pop()
    else:
        if cfg.need_train_data:
            data_transforms['train'] = make_data_transform(
                distort=(.5, 5, .57), skew=None, rotate=(.5, 10), crop=(.85, .1), translate=None, erasing=None,
                brjitter=(.8, 1), ctjitter=0, huejitter=.05,
                noise_factors=(cfg.noise_factor_1, cfg.noise_factor_2), pepper_factor=cfg.pepper_factor,
                jpeg_iterations=cfg.jpeg_iterations, jpeg_quality=cfg.jpeg_quality,
            )
        data_transforms['test'] = transforms.Compose([MayResize(), transforms.ToTensor()])

    with open(os.path.join(cfg.data_cv_dir, '..', 'classes.txt')) as f:
        cfg.data_classes = tuple(next(iter(f)).rstrip('\r\n').split(','))

    if cfg.cpload:
        assert options.class_filter is None
        cfg.model_classes = checkpoint['out_classes']
        assert checkpoint['data_classes'] == cfg.data_classes
    elif options.class_filter is not None:
        cfg.model_classes = options.class_filter
    else:
        cfg.model_classes = cfg.data_classes
    assert set(cfg.model_classes).issubset(set(cfg.data_classes))

    image_datasets: Dict[str, LabeledDataset] = {
        x: MultiLabelCSVDataset(
            cfg.model_classes,
            os.path.join(cfg.data_cv_dir, '{}_tagged.csv'.format(x)),
            os.path.join(cfg.data_cv_dir, '..', 'images'),
        ) for x in ('train', 'opt', 'test')}
    image_datasets['valid'] = CatDataset(image_datasets['opt'], image_datasets['test'])

    if options.quick_find:
        image_datasets['train'] = LabeledSubset(image_datasets['train'], range(250))
        image_datasets['opt'] = LabeledSubset(image_datasets['opt'], range(500))

    if options.task == 'preview_input':
        cfg.num_workers = 0  # Just need a little data, grab in the main thread

    transformed_datasets = {}
    bal_train_dataset: LabeledDataset = UNTYPED_NONE
    if options.task != 'print_numels':
        if cfg.need_train_data:
            if checkpoint is None:
                bal_train_dataset = ResampledLabeledSubset.new(image_datasets['train'])
            else:
                bal_train_dataset = ResampledLabeledSubset.load(
                    image_datasets['train'], checkpoint['dataset_indices'], checkpoint['dataset_rand'],
                )
            # prevent single-item batch at end of training phase - breaks batchnorm
            if len(bal_train_dataset) % cfg.batch_size == 1:
                bal_train_dataset = LabeledSubset(bal_train_dataset, np.arange(get_len(bal_train_dataset) - 1))
            bal_train_dataset_tform = TransformedDataset(bal_train_dataset, data_transforms['train'])
        for name in ('opt', 'test', 'valid'):
            transformed_datasets[name] = TransformedDataset(image_datasets[name], data_transforms['test'])

    del image_datasets

    def set_input_size(size: Tuple[int, int]) -> None:
        ops = (next(o for o in dt.transforms if type(o).__name__ == 'MayResize') for dt in data_transforms.values())
        for op in ops:
            op.size = size

    if options.task == 'preview_input':
        set_input_size((224, 224))

        # Get a batch of training data
        train_iter = make_dataloader(bal_train_dataset_tform)
        inputs, classes = next(iter(train_iter))

        # Make a grid from batch
        out = torchvision.utils.make_grid(inputs)

        imshow(out, title=str([
            ','.join(cn for c, cn in zip_strict(cl, cfg.model_classes) if c > .99)
            for cl in classes]))

        sys.exit(0)

    model_loader = ModelLoader(
        base_model=cfg.base_model if checkpoint is None else None,
        num_labels=len(cfg.model_classes),
        features=cfg.model_features,
        checkpoint=checkpoint,
        sub_cps=sub_cps,
        training=cfg.training,
        load_state='model' not in options.forget_state,
        sublayer_ratio=cfg.sublayer_ratio,  # type: ignore[arg-type]
        inner_dropout=cfg.inner_dropout,
        classifier_dropout=cfg.classifier_dropout,
        freeze_fc=options.freeze_fc,
        merge_fc=True,
        tta_mode=cfg.tta_mode,
        virtual_batch_size=cfg.virtual_batch_size,
    )
    del sub_cps
    model = model_loader.create_model()
    gc.collect()  # After unreferencing several checkpoints

    cfg.base_model = model_loader.base_model
    cfg.model_features = model_loader.features
    cfg.load_sublayer_ratio = model_loader.load_sublayer_ratio
    cfg.sublayer_ratio = model_loader.sublayer_ratio

    # Resize samples in dataloader worker threads
    set_input_size(model.max_insize)

    if checkpoint is not None:
        assert options.load is not None and len(options.load) == 1
        print('==> Loaded checkpoint from {!r}.'.format(*options.load))
        print('  Epoch: {}'.format(checkpoint.get('epoch', 'N/A')))
        print('  Test loss: {:.4f}'.format(checkpoint.get('test_loss', None) or checkpoint['val_loss']))

    if options.task == 'print_numels':
        def printn(loader: ModelLoader) -> None:
            total_numels = sum(loader.sublayer_numels)
            lines = []
            seen_numels = 0
            for i, numel in enumerate(reversed(loader.sublayer_numels), start=1):
                seen_numels += numel
                lines.append((i, seen_numels))
            for i, numels in reversed(lines):
                print('{}: {}'.format(i, numels / total_numels))

        if model_loader.sublayer_numels:
            printn(model_loader)
        elif model_loader.subloaders:
            for loader in model_loader.subloaders:
                print('-- {}'.format(loader.base_model))
                printn(loader)
        sys.exit(0)

    if options.task not in ('find_aug', 'train'):
        model_loader = UNTYPED_NONE

    device = torch.device('cuda:0' if options.use_cuda and torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    def get_norm_params() -> List[nn.Parameter]:
        return [p for name, p in model.named_parameters()
                if p.requires_grad and not re.search(r'\bbn[0-9]+\.bias$', name)]

    norm_params = get_norm_params()

    # Criterion is the loss function of our model.
    criterion_types: Dict[str, Type[ExtraTrainLoss]] = {
        'mse': EDL_MSE_Loss_Extra, 'log': EDL_Log_Loss_Extra, 'digamma': EDL_Digamma_Loss_Extra,
    }
    criterion_type = criterion_types[cfg.criterion]

    opt_params: Optional[List[Dict[str, Any]]] = None
    if cfg.training:
        cfg.optimizer_type = checkpoint['optimizer_type'] if cfg.cont_opt else {
            'adam': AdamW,
            'aadamw01': AAdamW01,
            'adabelief': MyAdaBelief,
            'sgdw': MySGDW,
            'lamb': FusedLAMB,
        }[DEFAULT_OPTIMIZER if options.optimizer is None else options.optimizer]

        skip = {}
        if hasattr(model, 'no_weight_decay'):
            skip = model.no_weight_decay()
        opt_params = add_weight_decay(model, cfg.weight_decay or 0., skip)

    nontrain_criterion: ExtraTrainLoss = UNTYPED_NONE
    if options.task != 'train':
        nontrain_criterion = criterion_type(norm_params, model, annealing_step=cfg.annealing_step)

    if options.task == 'find_aug':
        model_loader.postload()
        del model_loader
        optimizer = cfg.optimizer_type(opt_params, lr=options.lr)
        best_params = find_best_augment_params(
            bal_train_dataset, transformed_datasets['opt'],
            optimizer, model, device, nontrain_criterion,
            make_dataloader,
        )
        print('Best augmentation parameters:\n{}'.format(best_params))
        sys.exit(0)

    if options.task == 'train':
        cfg.optimizer_kwargs = checkpoint['optimizer_kwargs'] if cfg.cont_opt else {'lr': options.lr}
        clip_optimizer_type = prestep_clipping_optimizer(cfg.optimizer_type)
        optimizer = clip_optimizer_type(opt_params, norm_params, **cfg.optimizer_kwargs)
        del opt_params
        if cfg.cont_opt:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except ValueError as e:
                print("ERROR: Failed to load optimizer state. Try '--forget-state optimizer'.", file=sys.stderr)
                raise e
            print('==> Loaded optimizer state from checkpoint.')

        # Modify model AFTER model.load_state_dict() and optimizer.load_state_dict()
        model_loader.postload()
        if model_loader.sublayer_ratio != model_loader.load_sublayer_ratio:
            # Recompute parameters
            norm_params = get_norm_params()
            old_opt_params = {p for g in optimizer.param_groups for p in g['params']}
            skip = {}
            if hasattr(model, 'no_weight_decay'):
                skip = model.no_weight_decay()
            optimizer.param_groups = add_weight_decay(model, cfg.weight_decay or 0., skip)
            if cfg.cont_opt:
                new_opt_params = (p for g in optimizer.param_groups for p in g['params'])
                for param in old_opt_params.difference(new_opt_params):
                    del optimizer.state[param]  # The state dict will not serialize if we don't do this
        del model_loader

        cfg.scheduler_kwargs = None
        if cfg.cont_sch:
            cfg.scheduler_type = checkpoint['scheduler_type']
            if cfg.scheduler_type is not None:
                cfg.scheduler_kwargs = checkpoint['scheduler_kwargs'].copy()
                assert cfg.scheduler_kwargs is not None
                if cfg.scheduler_type is CosineAnnealingLR:
                    # New schedule
                    cfg.scheduler_kwargs['T_max'] = cfg.epochs
                elif cfg.cont_opt:
                    # Continue where schedule left off
                    cfg.scheduler_kwargs['last_epoch'] = checkpoint['epoch']
                print('==> Loaded scheduler state from checkpoint.')
        else:
            cfg.scheduler_name = DEFAULT_SCHEDULER if options.scheduler is None else options.scheduler
            sch_period = options.sch_period or DEFAULT_SCH_PERIOD
            if cfg.scheduler_name == 'cyclic':
                assert options.gamma is not None
                cfg.scheduler_type = CyclicLRWithRestarts
                cfg.scheduler_kwargs = {'batch_size': cfg.batch_size, 'epoch_size': get_len(bal_train_dataset),
                                        'restart_period': 1, 't_mult': 1,
                                        'eta_on_restart_cb': MyReduceMaxLROnRestart(
                                            ratio=options.gamma, period=sch_period
                                        ),
                                        'policy': 'cosine' if options.schpolicy is None else options.schpolicy}
            elif cfg.scheduler_name == 'cosine':
                cfg.scheduler_type = CosineAnnealingLR
                cfg.scheduler_kwargs = {'T_max': cfg.epochs}
            elif cfg.scheduler_name == 'linear':
                assert options.gamma is not None
                cfg.scheduler_type = StepLR
                cfg.scheduler_kwargs = {'step_size': sch_period, 'gamma': options.gamma}
            elif cfg.scheduler_name == 'NONE':
                cfg.scheduler_type = None
                cfg.scheduler_kwargs = None
            else:
                raise AssertionError('Invalid scheduler!')

        if cfg.scheduler_type is None:
            scheduler = None
        elif cfg.cont_sch and cfg.scheduler_type is not CosineAnnealingLR:
            scheduler = checkpoint['scheduler']
            scheduler.optimizer = optimizer

            if options.gamma_override is not None:
                if not cfg.cont_sch:
                    parser.error('--gamma-override is only valid if continuing a scheduler')  # What?
                elif not isinstance(scheduler, CyclicLRWithRestarts):
                    parser.error('--gamma-override requires a cyclic scheduler')
                elif options.gamma_override == scheduler.eta_on_restart_cb.ratio:
                    print('\x1b[93;1mWarning: --gamma-override unnecessarily specified\x1b[0m', file=sys.stderr)
                else:
                    scheduler.eta_on_restart_cb.ratio = options.gamma_override
        else:
            assert cfg.scheduler_kwargs is not None
            scheduler = cfg.scheduler_type(optimizer, **cfg.scheduler_kwargs)

        cfg.lr_warmup = options.lr_warmup if checkpoint is None else checkpoint.get('lr_warmup', False)
        if cfg.lr_warmup:
            if cfg.scheduler_type is CyclicLRWithRestarts:
                raise NotImplementedError('LR warmup cannot be used with cyclic scheduler')
            ws = UntunedLinearWarmup(optimizer)
            ws.last_step = -1  # Initialize the step counter
            if checkpoint is not None:
                ws.load_state_dict(checkpoint['warmup_state'])
            warmup_scheduler = ws
            del ws
        else:
            warmup_scheduler = None

        if cfg.cont_opt and not cfg.cont_sch and options.lr is not None:
            # LR override
            for group in optimizer.param_groups:
                group['initial_lr'] = group['lr'] = options.lr

        pbar = ProgressBar()
        trainer_cb = TrainerCallback(scheduler, warmup_scheduler, pbar)

        net = MyNeuralNetClassifier(
            lambda model: model,
            module__model=model,
            criterion=criterion_type,
            criterion__norm_params=norm_params,
            criterion__model=model,
            criterion__annealing_step=cfg.annealing_step,
            optimizer=lambda params, lr, optimizer: optimizer,  # noqa: U100
            optimizer__optimizer=optimizer,
            initial_epoch=cfg.initial_epoch,
            max_epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            iterator_train__num_workers=cfg.num_workers,
            iterator_train__pin_memory=True,
            iterator_valid__num_workers=cfg.num_workers,
            iterator_valid__pin_memory=True,
            callbacks=[
                EpochScoring(
                    get_valid_mcc,
                    name='valid_mcc',
                    lower_is_better=False,
                ),
                trainer_cb,
                pbar,
            ],
            callbacks__print_log=MyPrintLog(),
            warm_start=True,
            device=device,
            **({} if checkpoint is None else {
                'history': checkpoint['history'],
                'virtual_params_': checkpoint['virtual_params'],
            }),
        )
        net.initialize()

        if checkpoint is not None:
            state: Dict[str, Any] = checkpoint['random_state']
            random.setstate(state['python'])
            np.random.set_state(state['numpy'])
            torch.set_rng_state(state['torch'])
            if torch.cuda.is_available() and state['cuda'] is not None:
                torch.cuda.set_rng_state(state['cuda'])

        del checkpoint  # Not used past this point

        # Record initial state
        ds_valid = net.get_dataset(transformed_datasets['test'])
        net.notify('on_epoch_begin', dataset_train=None, dataset_valid=ds_valid)
        net.run_single_epoch(ds_valid, training=False, prefix='valid', step_fn=net.validation_step, epoch=0)
        net.notify('on_epoch_end', dataset_train=None, dataset_valid=ds_valid)

        # Train the model
        net.fit(bal_train_dataset_tform, y=None, valid=transformed_datasets['valid'])

        model_state_dict = copy.deepcopy(model.state_dict())
        optimizer_state_dict = copy.deepcopy(optimizer.state_dict())

        random_state = save_random_state()

        model.eval()
        gc.collect()
        best_state_dict = None
        if trainer_cb is not None:
            assert trainer_cb.best_state is not None  # Assume training began
            best_state_dict = torch.load(trainer_cb.best_state)
            model.load_state_dict(best_state_dict['model_state_dict'])

        print("==> Evaluating on 'opt' data...")
        opt_eval_y_pred, opt_eval_y_u, opt_eval_tot_ev, opt_eval_y_true, test_loss = \
            eval_inner(net, transformed_datasets['opt'], cfg.initial_epoch + cfg.epochs, progress=True, leave=False)

        if scheduler is not None:
            scheduler.optimizer = None

        if options.save is not None:
            if savedir := os.path.dirname(options.save):
                os.makedirs(savedir, exist_ok=True)
            if best_state_dict is not None:
                save_dict = best_state_dict
                assert save_dict is not None  # Assume training began
            else:
                save_dict = get_save_state_dict(
                    cfg=cfg,
                    options=options,
                    model_state_dict=model_state_dict,
                    optimizer_state_dict=optimizer_state_dict,
                    scheduler=scheduler,
                    warmup_scheduler=warmup_scheduler,
                    random_state=random_state,
                    net=net,
                )
            save_dict.update(get_save_stats_dict(
                cfg=cfg,
                model=model,
                data_transforms=data_transforms,
                opt_eval_y_true=opt_eval_y_true,
                opt_eval_y_pred=opt_eval_y_pred,
                opt_eval_y_u=opt_eval_y_u,
                opt_eval_tot_ev=opt_eval_tot_ev,
                test_loss=test_loss,
            ))
            torch.save(save_dict, options.save, pickle_module=mypickle)
            print('==> Saved checkpoint to {!r}.'.format(options.save))
        sys.exit(0)

    # visualize, metrics, or roc

    assert checkpoint is not None
    eval_checkpoint: Dict[str, Any] = {'epoch': checkpoint['epoch']}
    if cfg.tta_mode == checkpoint.get('tta_mode', 'none'):  # Only use cached data if it's computed the same way
        eval_checkpoint.update((k, v) for k, v in checkpoint.items() if k.startswith('opt_eval_'))
    del checkpoint

    model.eval()
    net = MyNeuralNetClassifier(
        lambda model: model,
        module__model=model,
        criterion=criterion_type,
        criterion__norm_params=norm_params,
        criterion__model=model,
        criterion__annealing_step=cfg.annealing_step,
        optimizer=None,
        initial_epoch=cfg.initial_epoch,
        batch_size=cfg.batch_size,
        iterator_train=None,
        iterator_valid__num_workers=cfg.num_workers,
        iterator_valid__pin_memory=True,
        callbacks='disable',
        device=device,
    )
    net.initialize()

    if options.task == 'visualize':
        print_predictions(model, eval_checkpoint)
    elif options.task == 'metrics':
        Metrics().metrics_eval(model, eval_checkpoint)
    elif options.task == 'eval_test':
        assert options.eval_dir is not None
        try:
            os.mkdir(options.eval_dir)
        except FileExistsError:
            pass

        Metrics().test_eval(net, eval_checkpoint)
    elif options.task == 'get_correct':
        assert options.correct_dir is not None
        try:
            os.mkdir(options.correct_dir)
        except FileExistsError:
            pass

        Metrics().correct_eval(net, eval_checkpoint)
    elif options.task == 'roc':
        assert options.fig_dir is not None
        try:
            os.mkdir(options.fig_dir)
        except FileExistsError:
            pass

        roc_eval(net, options.fig_dir, eval_checkpoint)
    else:
        raise AssertionError('Invalid task!')


if __name__ == '__main__':
    main()
