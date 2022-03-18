# -*- coding: utf-8 -*-

from __future__ import annotations

import abc
import dataclasses
import sys
from dataclasses import dataclass
from itertools import count, islice, zip_longest
from math import ceil, isclose
from numbers import Real
from typing import TYPE_CHECKING, Iterable, Iterator, Sequence, Tuple, Union, cast

import pretrainedmodels
import timm
import torch
from pretrainedmodels.models import dpn
from torch import nn
from torch.nn import functional as F

from util import zip_strict, zipstar_strict

if TYPE_CHECKING:
    from typing import Any, Callable, Dict, List, Literal, Optional
    from torch import Tensor


Module = Union['nn.Module[Any]']


def get_parameters(x: Union[nn.Parameter, Module]) -> Iterator[nn.Parameter]:
    if isinstance(x, nn.Parameter):
        yield x
    else:
        yield from x.parameters()


def set_req_grad(x: Union[nn.Parameter, Module], req_grad: bool) -> None:
    for param in get_parameters(x):
        param.requires_grad = req_grad


def total_elem(seq: Iterable[Tensor]) -> int:
    return sum(p.numel() for p in seq)


def union(intervals: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    u: List[Tuple[int, int]] = []
    for begin, end in sorted(intervals):
        if end < begin:
            raise ValueError('Degenerate interval: {}'.format((begin, end)))
        if u and (begin < u[-1][1] or isclose(begin, u[-1][1])):
            u[-1] = (u[-1][0], max(u[-1][1], end))  # Overlap -> merge
        else:
            u.append((begin, end))  # No overlap -> new interval
    return u


def get_sublayers_to_unfreeze(sublayer_numels: Sequence[int], sublayer_ratio: Tuple[float, float]) -> Tuple[int, int]:
    if not sublayer_numels:
        if sublayer_ratio != (0, 0):
            raise ValueError('Cannot unfreeze sublayers on this model!')
        return 0, 0

    seen_numel = sublayers_to_skip = sublayers_to_unfreeze = 0
    total_sublayer_numel = sum(sublayer_numels)
    for numel in reversed(sublayer_numels):
        seen_numel += numel
        if seen_numel <= int(total_sublayer_numel * sublayer_ratio[0]):
            sublayers_to_skip += 1
            continue
        if seen_numel <= int(total_sublayer_numel * sublayer_ratio[1]):
            sublayers_to_unfreeze += 1
            continue
        break
    return sublayers_to_skip, sublayers_to_unfreeze


def refreeze_model(model: Module, sublayers: Sequence[Module], sublayers_to_skip: int, sublayers_to_unfreeze: int,
                   fc: Module, freeze_fc: bool = False) -> int:
    if not isinstance(model, CatModel):
        # Freeze the model
        set_req_grad(model, req_grad=False)

    # Unfreeze a number of sublayers
    unfrozen_params = 0
    slskip = islice(reversed(sublayers), sublayers_to_skip, None)
    for layer in islice(slskip, sublayers_to_unfreeze):
        set_req_grad(layer, req_grad=True)
        unfrozen_params += total_elem(get_parameters(layer))

    if not freeze_fc:
        # Unfreeze the classifier
        set_req_grad(fc, req_grad=True)
        unfrozen_params += total_elem(fc.parameters())
    return unfrozen_params


def model_supports_resnet_d(base_model: str) -> bool:
    return base_model.startswith('resnet') or base_model.startswith('se_resnext')


# NB: Modifies parameters, but does not update sublayer_numels
def configure_resnet_d(base_model: str, sublayers: Sequence[Module], sublayers_to_skip: int,
                       sublayers_to_unfreeze: int) -> None:
    assert model_supports_resnet_d(base_model)
    for i, layer in enumerate(reversed(sublayers)):
        if i < sublayers_to_skip:
            continue
        if i >= sublayers_to_unfreeze:
            break
        if layer.downsample is None:
            continue
        # ResNet-D trick from https://arxiv.org/pdf/1812.01187.pdf
        assert isinstance(layer.downsample, nn.Sequential)
        if len(layer.downsample) == 3:
            avg, ds = layer.downsample[:2]
            assert isinstance(avg, nn.AvgPool2d)
            assert isinstance(avg.kernel_size, tuple) and (avg.kernel_size[0] > 1 or avg.kernel_size[1] > 1)
            assert isinstance(ds, nn.Conv2d) and ds.kernel_size == (1, 1)
            continue  # Already applied
        assert len(layer.downsample) == 2
        old_ds = layer.downsample[0]
        assert isinstance(old_ds, nn.Conv2d) and old_ds.kernel_size == (1, 1)
        if not (old_ds.stride[0] > 1 or old_ds.stride[1] > 1):
            continue  # Not applicable
        layer.downsample = nn.Sequential(
            nn.AvgPool2d(kernel_size=cast(Tuple[int, int], old_ds.stride)),
            nn.Conv2d(old_ds.in_channels, old_ds.out_channels,
                      kernel_size=1, stride=1, bias=False),
            *layer.downsample[1:])
        nn.init.kaiming_normal_(layer.downsample[1].weight,
                                mode='fan_out', nonlinearity='relu')


class FCInfo:
    def __init__(self, model: Module) -> None:
        try:
            fc_name = next(c for c in ('fc', '_fc', 'last_linear', 'classifier', 'classif', 'head')
                           if getattr(model, c, None) is not None)
        except StopIteration as e:
            raise RuntimeError('Could not find classifier layer') from e

        self._model = model
        self._fc_name = fc_name

    def get(self) -> Module:
        fc = getattr(self._model, self._fc_name)
        if isinstance(fc, timm.models.layers.ClassifierHead):
            fc = fc.fc
        if isinstance(fc, nn.Sequential) and len(fc) == 2 and isinstance(fc[0], nn.Dropout):
            fc = fc[1]
        return fc

    def set(self, fc: Module) -> None:
        mfc = getattr(self._model, self._fc_name)
        if isinstance(mfc, timm.models.layers.ClassifierHead):
            mfc.fc = fc
        else:
            setattr(self._model, self._fc_name, fc)


def parse_sublayer_ratio(sublayer_ratio: Union[float, Tuple[float, float], Sequence[Tuple[float, float]]],
                         base_model: Union[str, Tuple[str, ...]]) -> Tuple[Any, ...]:
    def endify(sr: Any) -> Tuple[float, float]:
        if isinstance(sr, Sequence):
            assert len(sr) == 2
            return tuple(sr)  # type: ignore[return-value]
        assert isinstance(sublayer_ratio, Real)
        return (0, sr)

    if isinstance(base_model, tuple) and len(base_model) > 1:
        # Must have a matching models dimension
        assert isinstance(sublayer_ratio, Sequence) and len(sublayer_ratio) == len(base_model)
        return tuple(map(endify, sublayer_ratio))
    if isinstance(sublayer_ratio, Sequence) and len(sublayer_ratio) == 1:
        sublayer_ratio, = sublayer_ratio  # Unwrap redundant models dimension
    return endify(sublayer_ratio)


def apply_tta(eval_fun: Callable[[Tensor], Tensor], x: Tensor, mode: str, training: bool) -> Tensor:
    alpha1 = eval_fun(x)
    if training or mode == 'none':
        return alpha1

    alpha2 = eval_fun(x.flip(dims=(-1,)))
    if mode == 'mean':  # Simple mean of model outputs
        return torch.mean(torch.stack((alpha1, alpha2)), dim=0)
    if mode == 'local_certainty':  # Pick most certain model per label
        return torch.max(alpha1, alpha2)
    if mode == 'global_certainty':  # Pick most certain model per sample
        def select(ab: Tensor, i: Tensor) -> Tensor:
            return ab.gather(0, i.unsqueeze(dim=0)).squeeze(dim=0)
        alphas = torch.stack((alpha1, alpha2))
        sample_ev = alphas.sum(dim=-1, keepdim=True)
        return select(alphas, sample_ev.argmax(dim=0).expand(alpha1.shape))

    raise AssertionError('Invalid TTA mode!')


# Subclasses must use dirichlet() in forward()
class BaseDirichletModel(nn.Module, metaclass=abc.ABCMeta):
    @staticmethod
    def dirichlet(x: Tensor) -> Tensor:
        evidence = F.relu(x)
        alpha = evidence + 1
        # N labels -> N/2 pairs of positive and negative probabilities
        return alpha.view(alpha.size(0), -1, 2)

    @staticmethod
    def normalize_batch(batch: Tensor, mean: Tuple[float, float, float], std: Tuple[float, float, float]) -> Tensor:
        with torch.no_grad():
            mean_ = torch.as_tensor(mean, dtype=batch.dtype, device=batch.device)[None, :, None, None]
            std_  = torch.as_tensor(std,  dtype=batch.dtype, device=batch.device)[None, :, None, None]
            return batch.sub(mean_).div_(std_)


class DirichletModel(BaseDirichletModel):
    def __init__(self, model: Any, tta_mode: str) -> None:
        super().__init__()
        self.model = model
        self.tta_mode = tta_mode
        self._fc_info = FCInfo(model)

    @property
    def default_cfg(self) -> Dict[str, Any]:
        return self.model.default_cfg

    @property
    def fc(self) -> Module:
        return self._fc_info.get()

    @fc.setter
    def fc(self, fc: Module) -> None:
        self._fc_info.set(fc)

    def forward_(self, x: Tensor) -> Tensor:
        x = self.normalize_batch(x, self.model.default_cfg['mean'], self.model.default_cfg['std'])
        x = self.model(x)
        x = self.dirichlet(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return apply_tta(self.forward_, x, mode=self.tta_mode, training=self.training)


class CatModel(BaseDirichletModel):
    def __init__(self, models: Iterable[Module], out_features: Optional[int], classifier_dropout: Optional[float],
                 tta_mode: str) -> None:
        super().__init__()
        self.classifier_dropout = classifier_dropout
        self.tta_mode = tta_mode

        # Unwrap models first if we're going to bypass their classifiers
        if out_features is not None:
            models = ((m.model if isinstance(m, DirichletModel) else m) for m in models)

        self.models = nn.ModuleList(models)
        self.input_sizes = []
        self.fc_infeatures = []
        for model in self.models:
            fc_info = FCInfo(model)
            fc = fc_info.get()

            inx, iny = model.default_cfg['input_size'][-2:]
            assert inx == iny
            self.input_sizes.append(inx)

            if out_features is not None:
                if hasattr(fc, 'in_features'):
                    self.fc_infeatures.append(fc.in_features)
                elif hasattr(fc, 'in_channels'):
                    self.fc_infeatures.append(fc.in_channels)
                else:
                    raise RuntimeError('Could not find in_features/in_channels')

                fc_info.set(nn.Identity())

        self.insize_uniq = sorted(set(self.input_sizes), reverse=True)
        print('==> CatModel input sizes: {}'.format(self.insize_uniq))

        # NB: Placeholder, will be replaced and have dropout added later
        self.fc = nn.Identity() if out_features is None else nn.Linear(sum(self.fc_infeatures), out_features)

    def forward_(self, x: Tensor) -> Tensor:
        # One scale at a time
        outputs: List[torch.Tensor] = []
        for insize in self.insize_uniq:
            scaled = self.scale_batch(x, insize)
            for model, m_insize in zip_strict(self.models, self.input_sizes):
                if m_insize == insize:
                    norm = self.normalize_batch(scaled, model.default_cfg['mean'], model.default_cfg['std'])
                    outputs.append(model(norm))
            del scaled

        x = torch.cat(outputs, dim=1)
        if self.classifier_dropout is not None:
            x = F.dropout(x, p=self.classifier_dropout, training=self.training)
        x = self.fc(x)
        x = self.dirichlet(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return apply_tta(self.forward_, x, mode=self.tta_mode, training=self.training)

    @staticmethod
    def scale_batch(batch: Tensor, insize: int) -> Tensor:
        assert batch.ndim == 4
        _, c, h, w = batch.shape
        assert c == 3
        assert h == w
        if h == insize:
            return batch  # Same -> keep
        with torch.no_grad():
            if abs(h - insize) < insize * .05:
                # Small difference -> no interpolation
                return F.interpolate(batch, size=insize, mode='nearest')
            # Large difference -> bilinear interpolation
            return F.interpolate(batch, size=insize, mode='bilinear', align_corners=False)


# Converts alpha output of model to the positive class probability for each label.
def pos_proba(alpha: Tensor) -> Tensor:
    assert alpha.shape[-1] == 2
    probas = alpha / alpha.sum(dim=-1, keepdim=True)
    return probas[..., 1]  # Positive class


def pred_uncertainty(alpha: Tensor) -> Tensor:
    return alpha.shape[-1] / alpha.sum(dim=-1)


class GhostBatchNorm(nn.BatchNorm2d):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.virtual_batch_size = kwargs.pop('virtual_batch_size')
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        chunks = x.chunk(ceil(x.size(0) / self.virtual_batch_size))
        return torch.cat(list(map(super().forward, chunks)))


@dataclass(init=False)
class ModelLoader:
    base_model: Any  # Optional[Union[str, Tuple[str, ...]]]
    num_labels: Union[int, Literal[False]]
    features: Any  # Optional[Union[Dict[str, Any], Tuple[Dict[str, Any], ...]]]
    checkpoint: Optional[Dict[str, Any]]
    sub_cps: Optional[Tuple[Dict[str, Any], ...]]
    training: bool
    load_state: bool
    load_sublayer_ratio: Any  # Optional[Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]]
    sublayer_ratio: Any  # Union[Tuple[float, float], Tuple[Tuple[float, float], ...]]
    inner_dropout: Union[float, Tuple[float, ...]]
    classifier_dropout: Optional[float]
    freeze_fc: bool
    tta_mode: str
    virtual_batch_size: Optional[int]

    def __init__(
        self,
        base_model: Optional[Union[str, Sequence[str]]] = None,
        num_labels: Optional[Union[int, Literal[False]]] = None,
        features: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]] = None,
        checkpoint: Optional[Dict[str, Any]] = None,
        sub_cps: Optional[Sequence[Dict[str, Any]]] = None,
        training: bool = False,
        load_state: bool = True,
        load_sublayer_ratio: Optional[Union[float, Tuple[float, float], Sequence[Tuple[float, float]]]] = None,
        sublayer_ratio: Optional[Union[float, Tuple[float, float], Sequence[Tuple[float, float]]]] = None,
        inner_dropout: Optional[Union[float, Sequence[float]]] = None,
        classifier_dropout: Optional[float] = None,
        freeze_fc: bool = False,
        merge_fc: bool = True,
        tta_mode: str = 'none',
        virtual_batch_size: Optional[int] = None,
    ) -> None:
        if checkpoint is not None:
            if base_model is None:
                base_model = checkpoint['base_model']
                assert base_model is not None
            else:
                assert isinstance(base_model, str)  # No tuples or lists
        elif sub_cps is not None:
            assert base_model is None
            assert len(sub_cps) > 1
            base_model = []
            for cp in sub_cps:
                bm = cp['base_model']
                if isinstance(bm, Sequence) and not isinstance(bm, str):
                    bm, = bm  # No submodels of submodels
                base_model.append(bm)
        else:
            assert base_model is not None
        if isinstance(base_model, Sequence) and not isinstance(base_model, str):
            if len(base_model) == 1:
                base_model, = base_model
            else:
                base_model = tuple(base_model)
        self.base_model = base_model
        if num_labels is None:
            assert checkpoint is not None
            try:
                num_labels = len(checkpoint['out_classes'])
            except KeyError:
                num_labels = checkpoint['out_features']
            assert num_labels is not None
        self.num_labels = num_labels
        # Positive and negative evidence for each label
        self.out_features = False if num_labels is False else 2 * num_labels
        features_: Optional[Union[Dict[str, Any], Sequence[Dict[str, Any]]]]
        if features is not None:
            features_ = features
        elif checkpoint is not None:
            features_ = checkpoint['model_features']
        elif sub_cps is not None:
            features_ = None  # Will let submodels decide what the features are
        elif isinstance(self.base_model, tuple):
            features_ = [{} for _ in self.base_model]  # Defaults
        else:
            features_ = {}  # Defaults
        if not isinstance(features_, Sequence):
            self.features = features_
        elif len(features_) == 1:
            self.features, = features_
        else:
            self.features = tuple(features_)
        assert self.features is None or (isinstance(self.features, tuple) == isinstance(self.base_model, tuple))
        if isinstance(self.features, tuple):
            assert len(self.features) == len(self.base_model)
        assert checkpoint is None or sub_cps is None
        assert not isinstance(base_model, tuple) or sub_cps is None or (len(sub_cps) == len(base_model))
        self.checkpoint = checkpoint
        self.sub_cps = None if sub_cps is None else tuple(sub_cps)
        self.training = training
        if checkpoint is None and sub_cps is None:
            load_state = False  # Make sure we load pretrained weights
        self.load_state = load_state
        if sub_cps is not None:
            pass  # Defaulted in subloaders
        elif sublayer_ratio is None and (training or checkpoint is not None):
            # Default to the value in the checkpoint
            assert checkpoint is not None
            sublayer_ratio = checkpoint['sublayer_ratio']
            assert sublayer_ratio is not None
        self.sublayer_ratio = None if sublayer_ratio is None \
            else parse_sublayer_ratio(sublayer_ratio, base_model)
        if sub_cps is not None:
            # Defaulted in subloaders
            assert load_sublayer_ratio is None
        elif checkpoint is not None:
            # Get it from the checkpoint, as intended
            try:
                load_sublayer_ratio = checkpoint['load_sublayer_ratio']
            except KeyError:
                load_sublayer_ratio = checkpoint['sublayer_ratio']
            assert load_sublayer_ratio is not None
        elif load_sublayer_ratio is None and training:
            # No checkpoints, use the same value
            assert sublayer_ratio is not None
            load_sublayer_ratio = sublayer_ratio
        self.load_sublayer_ratio = None if load_sublayer_ratio is None \
            else parse_sublayer_ratio(load_sublayer_ratio, base_model)
        if inner_dropout is None:
            assert checkpoint is not None
            inner_dropout = checkpoint['inner_dropout']
        if not isinstance(inner_dropout, Sequence):
            self.inner_dropout = inner_dropout
        elif len(inner_dropout) == 1:
            self.inner_dropout, = inner_dropout
        else:
            self.inner_dropout = tuple(inner_dropout)
        assert isinstance(self.inner_dropout, tuple) == isinstance(self.base_model, tuple)
        if isinstance(self.inner_dropout, tuple):
            assert len(self.inner_dropout) == len(self.base_model)
        if classifier_dropout is None and num_labels is not False:
            assert checkpoint is not None
            try:
                classifier_dropout = checkpoint['classifier_dropout']
            except KeyError:
                print('\x1b[93;1mWarning: Unknown classifier_dropout, defaulting to 0.5\x1b[0m', file=sys.stderr)
                classifier_dropout = .5
        self.classifier_dropout = classifier_dropout
        self.freeze_fc = freeze_fc
        self.merge_fc = merge_fc and isinstance(self.base_model, tuple)
        self.tta_mode = tta_mode
        self.virtual_batch_size = virtual_batch_size

        self.model: Any = None  # Optional[Module]
        self.sublayers: Any = None  # Optional[List[Module]]
        self.sublayer_numels: Any = None  # Optional[List[int]]
        self.subloaders: Optional[Tuple[ModelLoader, ...]] = None

    def create_model(self, log: bool = True) -> Module:
        if log:
            print('==> Preparing base model.')
            print('  Model: {}'.format(self.base_model))
            if not self.features or (isinstance(self.features, tuple) and not any(self.features)):
                features_str = '(none)'
            else:
                features_str = str(self.features)
            print('  Features: {}'.format(features_str))

        # If load_state is True, we will overwrite the state_dict later, so don't load a pretrained model
        if isinstance(self.base_model, tuple):
            assert len(self.base_model) > 1
            assert self.num_labels is not False
            assert isinstance(self.inner_dropout, tuple)
            # Don't make me type this, please...
            def get_model(name, mf, cp, lsr, sr, ido):  # type: ignore[no-untyped-def]
                subloader = dataclasses.replace(
                    self, base_model=name, features=mf, checkpoint=cp, sub_cps=None, load_sublayer_ratio=lsr,
                    sublayer_ratio=sr, inner_dropout=ido,
                )
                if self.merge_fc:
                    subloader.freeze_fc = True  # Don't bother unfreezing a fc that doesn't exist
                subloader.create_model(log=False)
                return subloader
            self.subloaders = tuple(get_model(*args) for args in zip_longest(
                self.base_model, self.features or (), self.sub_cps or (), self.load_sublayer_ratio or (),
                self.sublayer_ratio or (), self.inner_dropout or (),
            ))
            self.sub_cps = True  # type: ignore  # Save memory
            # Read attributes from subloaders which were either defaulted or loaded from checkpoints
            self.features, self.load_sublayer_ratio = \
                zipstar_strict((s.features, s.load_sublayer_ratio) for s in self.subloaders)
            self.model = CatModel((s.model for s in self.subloaders), self.out_features if self.merge_fc else None,
                                  self.classifier_dropout, self.tta_mode)
        elif (
            self.base_model.startswith('mixnet_')
            or self.base_model.startswith('tresnet_')
            or (self.base_model.startswith('resnet') and self.base_model.endswith('d'))
            or self.base_model.startswith('regnet')
            or self.base_model.startswith('rexnet')
            or self.base_model.startswith('resnest')
            or self.base_model.startswith('ecaresnet')
            or self.base_model.startswith('gluon_resnet')
            or self.base_model.startswith('efficientnet_b')
            or self.base_model.startswith('cspdarknet')
            or self.base_model.startswith('inception_v')
            or self.base_model.startswith('xcit_')
        ):
            kwargs: Dict[str, Any] = {}
            if self.base_model.startswith('xcit_'):
                assert not isinstance(self.inner_dropout, tuple)
                kwargs['drop_rate'] = self.inner_dropout

            self.model = timm.create_model(
                self.base_model, pretrained=(self.training and not self.load_state), **kwargs)
        else:
            # Obtain the desired model from the pretrainedmodels library
            ds = 'imagenet+5k' if self.base_model.startswith('dpn') and self.base_model.endswith('b') else 'imagenet'
            factory = getattr(pretrainedmodels, self.base_model)
            model = factory(num_classes=1000, pretrained=(ds if self.training and not self.load_state else None))
            try:
                mod: Any = sys.modules[factory.__module__]
                cfg = mod.pretrained_settings[self.base_model][ds]
            except (AttributeError, KeyError):
                pass
            else:
                model.default_cfg = {k: cfg[k] for k in ('input_size', 'mean', 'std')}
            self.model = model
            del model

        def apply_post_dropout(*modules: Module) -> nn.Sequential:
            assert not isinstance(self.inner_dropout, tuple)
            return nn.Sequential(*modules, nn.Dropout(self.inner_dropout))

        def make_gbn(bn: nn.BatchNorm2d, virtual_batch_size: int) -> GhostBatchNorm:
            gbn = GhostBatchNorm(bn.num_features, bn.eps, bn.momentum, bn.affine, bn.track_running_stats,
                                 virtual_batch_size=virtual_batch_size)
            with torch.no_grad():
                if bn.affine:
                    gbn.weight.copy_(bn.weight)
                    gbn.bias.copy_(bn.bias)
                if bn.track_running_stats:
                    gbn.running_mean.copy_(bn.running_mean)
                    gbn.running_var.copy_(bn.running_var)
            return gbn

        parents: Iterable[Module]
        if isinstance(self.base_model, tuple):
            assert len(self.base_model) > 1
            self.sublayers = []
        elif self.base_model.startswith('resnet'):
            self.sublayers = [bb for i in range(1, 5)
                                 for bb in getattr(self.model, 'layer{}'.format(i))]

            def reluapd(bb: Module, rlname: str) -> bool:
                try:
                    relu = getattr(bb, rlname)
                except AttributeError:
                    return False
                assert isinstance(relu, nn.Module)
                setattr(bb, rlname, apply_post_dropout(relu))
                return True

            for i in range(2, 5):
                for bb in getattr(self.model, 'layer{}'.format(i)):
                    if reluapd(bb, 'relu'):
                        continue
                    for j in count(start=1):
                        if not reluapd(bb, 'act{}'.format(j)):
                            break
        elif self.base_model.startswith('se_resnext'):
            self.sublayers = [bb for i in range(5)
                                 for bb in getattr(self.model, 'layer{}'.format(i))]

            for i in range(1, 5):
                for bb in getattr(self.model, 'layer{}'.format(i)):
                    assert isinstance(bb.relu, nn.Module)
                    bb.relu = apply_post_dropout(bb.relu)
        elif self.base_model.startswith('regnet'):
            self.sublayers = [bb for i in range(1, 5)
                                 for bb in getattr(self.model, 's{}'.format(i)).children()]

            parents = (p for i in range(2, 5)
                         for bb in getattr(self.model, 's{}'.format(i)).children()
                         for p in bb.modules())
            for parent in parents:
                for name, act in parent.named_children():
                    if name in ('act', 'act3'):
                        setattr(parent, name, apply_post_dropout(act))
        elif self.base_model.startswith('rexnet'):
            self.sublayers = list(self.model.features)

            parents = (p for bb in islice(self.sublayers, 3, None) for p in bb.modules())
            for parent in parents:
                for name, act in parent.named_children():
                    if name in ('act', 'act_dw'):
                        setattr(parent, name, apply_post_dropout(act))
        elif self.base_model.startswith('resnest'):
            self.sublayers = [bb for i in range(1, 5)
                                 for bb in getattr(self.model, 'layer{}'.format(i)).children()]

            parents = (p for i in range(2, 5)
                         for bb in getattr(self.model, 'layer{}'.format(i)).children()
                         for p in bb.modules())
            for parent in parents:
                for name, act in parent.named_children():
                    if name in ('act0', 'act1', 'act3'):
                        setattr(parent, name, apply_post_dropout(act))
        elif self.base_model.startswith('ecaresnet') or self.base_model.startswith('gluon_resnet'):
            self.sublayers = [bb for i in range(1, 5)
                                 for bb in getattr(self.model, 'layer{}'.format(i)).children()]

            parents = (p for i in range(2, 5)
                         for bb in getattr(self.model, 'layer{}'.format(i)).children()
                         for p in bb.modules())
            for parent in parents:
                for name, act in parent.named_children():
                    if name in ('act1', 'act2', 'act3'):
                        setattr(parent, name, apply_post_dropout(act))
        elif self.base_model == 'xception':
            self.sublayers = [getattr(self.model, 'block{}'.format(i))
                              for i in range(1, 13)]

            for block in self.sublayers:
                block.rep = nn.Sequential(*(
                    apply_post_dropout(mod) if isinstance(mod, nn.ReLU)
                    else mod
                    for mod in block.rep))
        elif self.base_model.startswith('dpn'):
            self.sublayers = [mod for mod in self.model.features
                              if isinstance(mod, dpn.DualPathBlock)]

            for conv in self.sublayers:
                for mod in conv.children():
                    if isinstance(mod, dpn.BnActConv2d):
                        mod.act = apply_post_dropout(mod.act)
        elif self.base_model == 'inceptionresnetv2':
            self.sublayers = []  # XXX: Unfreezing sublayers not implemented

            for mod in self.model.modules():
                relu = getattr(mod, 'relu', None)
                if relu is None or isinstance(relu, nn.Sequential):
                    continue
                mod.relu = apply_post_dropout(relu)
        elif self.base_model.startswith('efficientnet_b') or self.base_model.startswith('mixnet_'):
            self.sublayers = [l for b in self.model.blocks for l in b]
            for layer in self.sublayers:
                layer.act1 = apply_post_dropout(layer.act1)
                layer.act2 = apply_post_dropout(layer.act2)
        elif self.base_model.startswith('tresnet_'):
            self.sublayers = [bb for i in range(1, 5)
                                 for bb in getattr(self.model.body, 'layer{}'.format(i))]

            for i in range(2, 5):
                for bb in getattr(self.model.body, 'layer{}'.format(i)):
                    assert isinstance(bb.relu, nn.Module)
                    bb.relu = apply_post_dropout(bb.relu)
        elif self.base_model.startswith('cspdarknet'):
            self.sublayers = [bb for s in self.model.stages for bb in s.children()]

            parents = (p for s in islice(self.model.stages, 1, None)
                         for bb in s.children()
                         for p in bb.modules())
            for parent in parents:
                for name, act in parent.named_children():
                    if name == 'act':
                        setattr(parent, name, apply_post_dropout(act))
        elif self.base_model.startswith('inception_v'):
            self.sublayers = list(self.model.features)

            parents = (p for bb in islice(self.sublayers, 4, None) for p in bb.modules())
            for parent in parents:
                for name, act in parent.named_children():
                    if name == 'relu':
                        setattr(parent, name, apply_post_dropout(act))
        elif self.base_model.startswith('xcit_'):
            self.sublayers = [self.model.patch_embed, self.model.pos_embed, *self.model.blocks, self.model.cls_token,
                              *self.model.cls_attn_blocks, self.model.norm]

            # Dropout is built-in
        else:
            raise NotImplementedError('Unknown network type: {}'.format(self.base_model))

        if not isinstance(self.base_model, tuple):
            self.sublayer_numels = [total_elem(get_parameters(l)) for l in self.sublayers]

        if self.virtual_batch_size is not None and not isinstance(self.base_model, tuple):
            # Only necessary for trained sublayers
            sublayers_to_skip, sublayers_to_unfreeze = \
                get_sublayers_to_unfreeze(self.sublayer_numels, self.sublayer_ratio)
            slskip = islice(reversed(self.sublayers), sublayers_to_skip, None)
            parents = tuple(p for bb in islice(slskip, sublayers_to_unfreeze) for p in bb.modules())
            for parent in parents:
                for name, mod in parent.named_children():
                    if isinstance(mod, nn.BatchNorm2d):
                        setattr(parent, name, make_gbn(mod, self.virtual_batch_size))

        # We replace the fully connected layers of the base model
        # which served as the classifier with our custom trainable classifier.

        fc_info = FCInfo(self.model)
        fc = fc_info.get()

        # NB: This classifier may be replaced by CatModel, but it is always created so we can load the complete
        #     checkpoint.
        if isinstance(self.base_model, tuple) and not self.merge_fc:
            # Create a CatModel that merges the classifier layer outputs.
            # Here we pass the outputs through without modification.
            pass
        else:
            assert self.classifier_dropout is not None
            if hasattr(fc, 'in_features'):
                classifier: Module = nn.Linear(fc.in_features, self.out_features)
            elif hasattr(fc, 'in_channels'):
                classifier = nn.Conv2d(fc.in_channels, self.out_features, kernel_size=1, stride=1)
            else:
                raise RuntimeError('Could not find in_features/in_channels')
            fc = nn.Sequential(nn.Dropout(self.classifier_dropout), classifier)
            del classifier

            fc_info.set(fc)

            if not isinstance(self.model, BaseDirichletModel):
                self.model = DirichletModel(self.model, self.tta_mode)  # Get 2D Dirichlet output
        del fc_info, fc

        if self.merge_fc:
            # Run through refreeze_model to deal with CatModel's merged fc
            self.setfrozen(0, 0, log=log)
        elif not isinstance(self.base_model, tuple):
            if self.load_sublayer_ratio is not None:
                sublayers_to_skip, sublayers_to_unfreeze = \
                    get_sublayers_to_unfreeze(self.sublayer_numels, self.load_sublayer_ratio)

                if self.features.get('resnet-d'):
                    configure_resnet_d(self.base_model, self.sublayers, sublayers_to_skip, sublayers_to_unfreeze)
                self.setfrozen(sublayers_to_skip, sublayers_to_unfreeze, log=log)

        if self.load_state and self.checkpoint is not None:
            param_names = {n for n, p in self.model.named_parameters()}
            state_dict = self.checkpoint['model_state_dict']
            state_dict_names = state_dict.keys()
            if (not param_names.intersection(state_dict_names) and
                all(n.startswith('model.') for n in param_names) and
                not any(n.startswith('model.') for n in state_dict_names)
            ):
                # Edge case: Model saved without DirichletModel wrapper but loaded with one
                state_dict = {'model.' + n: p for n, p in state_dict.items()}

            self.model.load_state_dict(state_dict)
            print('==> Loaded model state from checkpoint.')

        self.checkpoint = True  # type: ignore  # Save memory
        return self.model

    def postload(self) -> None:
        if self.subloaders is None:
            self._postload_inner()
        else:
            # Only refreeze submodels because that's where the sublayers are
            for loader in self.subloaders:
                loader._postload_inner()  # pylint: disable=protected-access

        print('==> Prepared base model.')
        print('  {:,} total parameters'.format(total_elem(self.model.parameters())))

        if self.training:
            total_trainable_params = total_elem(p for p in self.model.parameters() if p.requires_grad)
            print('  {:,} trainable parameters'.format(total_trainable_params))

    def setfrozen(self, sublayers_to_skip: Optional[int], sublayers_to_unfreeze: Optional[int], postload: bool = False,
                  log: bool = True) -> None:
        if not self.training:
            return  # Makes no difference to evaluation
        if postload or self.checkpoint is None:
            assert sublayers_to_skip is not None and sublayers_to_unfreeze is not None
            ratio = self.sublayer_ratio if postload else self.load_sublayer_ratio
            if ratio[0] == 0 and ratio[1] > 1:
                set_req_grad(self.model, req_grad=True)
                unfrozen_params = total_elem(self.model.parameters())
            else:
                fc = FCInfo(self.model).get()
                unfrozen_params = refreeze_model(
                    self.model, self.sublayers, sublayers_to_skip, sublayers_to_unfreeze, fc,
                    freeze_fc=self.freeze_fc,
                )
            if log and unfrozen_params:
                print('==> Set {:,} parameters as trainable.'.format(unfrozen_params))
        else:
            total_trainable = 0
            param_names = {n for n, p in self.model.named_parameters()}
            params_trained = self.checkpoint['params_trained']
            param_trained_names = params_trained.keys()
            if (not param_names.intersection(param_trained_names) and
                all(n.startswith('model.') for n in param_names) and
                not any(n.startswith('model.') for n in param_trained_names)
            ):
                # Edge case: Model saved without DirichletModel wrapper but loaded with one
                params_trained = {'model.' + n: p for n, p in params_trained.items()}
                param_trained_names = params_trained.keys()
            diffstrs = []
            if diff := param_names.difference(param_trained_names):
                diffstrs.append('\tMissing key(s) in params_trained: {}'.format(diff))
            if diff := set(param_trained_names).difference(param_names):
                diffstrs.append('\tUnexpected key(s) in params_trained: {}'.format(diff))
            if diffstrs:
                raise ValueError('Error(s) loading params_trained from checkpoint:\n{}'.format('\n'.join(diffstrs)))
            for name, param in self.model.named_parameters():
                train = params_trained[name]
                param.requires_grad = train
                if train:
                    total_trainable += param.numel()
            if log:
                print('==> Set {:,} parameters as trainable from checkpoint.'.format(total_trainable))

    def _postload_inner(self) -> None:
        if self.sublayer_ratio in (None, self.load_sublayer_ratio):
            return
        if self.features.get('resnet-d'):
            # resnet-d is additive, never taken away, so we have to compute and store the total range.
            new_lsr = union((self.load_sublayer_ratio, self.sublayer_ratio))
            if len(new_lsr) > 1:
                raise ValueError('Non-overlapping sublayer ratios are not implemented')
            self.load_sublayer_ratio, = new_lsr
            sublayers_to_skip, sublayers_to_unfreeze = \
                get_sublayers_to_unfreeze(self.sublayer_numels, self.load_sublayer_ratio)
            configure_resnet_d(self.base_model, self.sublayers, sublayers_to_skip, sublayers_to_unfreeze)
        sublayers_to_skip, sublayers_to_unfreeze = \
            get_sublayers_to_unfreeze(self.sublayer_numels, self.sublayer_ratio)
        self.setfrozen(sublayers_to_skip, sublayers_to_unfreeze, postload=True)
