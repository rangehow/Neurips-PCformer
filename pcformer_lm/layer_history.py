import torch
import torch.nn as nn
import queue
import torch.nn.functional as F
import numpy as np


def CreateLayerHistory(block_depth, block_dim, normalization_before):
    return RKPredictorMultistepCorrectorHistory(block_depth, block_dim, normalization_before)
    # return LearnableDenseLayerHistory(block_depth, block_dim, normalization_before)


class BaseLayerHistory(nn.Module):

    def __init__(self, block_depth, block_dim, normalization_before):
        super(BaseLayerHistory, self).__init__()
        # since Swin-Transformer follows a pre-Norm Transformer backcbone.
        self.normalize_before = normalization_before

        # the first layer (aka. embedding layer) does not have layer normalization
        self.layer_norms = nn.ModuleList(LayerNorm(block_dim) for _ in range(block_depth))

    def add(self, layer):
        raise NotImplemented

    def pop(self):
        raise NotImplemented

    def clean(self):
        raise NotImplemented


class LearnableDenseLayerHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, block_depth, block_dim, normalization_before):
        super(LearnableDenseLayerHistory, self).__init__(block_depth, block_dim, normalization_before)
        self.sum = None
        self.layer_idx = 0
        self.layer_num = 1 + block_depth
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)

    def extra_repr(self):
        return 'n_layers={layer_num}, '.format(**self.__dict__)

    def add(self, layer):

        # assert self.normalize_before is True, "This dlcl only supports the pre-Norm Swin Transformer"
        self.layer_idx += 1

        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        if self.normalize_before:
            layer = self.layer_norms[self.layer_idx - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        ret = (torch.stack(self.layers, 0) * self.weight[self.layer_idx - 1, : self.layer_idx].view(-1, 1, 1, 1)).sum(0)

        if self.layer_idx == 1 or self.normalize_before:
            return ret

        return self.layer_norms[self.layer_idx - 2](ret)

    def clean(self):
        self.sum = None
        self.layer_idx = 0
        self.layers = []


class RKPredictorMultistepCorrectorHistory(BaseLayerHistory):
    """
    x_n = (x_1 + y_1 + y_2 + ... y_{n-1}) / n
    """

    def __init__(self, block_depth, block_dim, normalization_before):
        super(RKPredictorMultistepCorrectorHistory, self).__init__(block_depth, block_dim, normalization_before)
        self.sum = None
        self.layer_idx = 0
        self.layer_num = 1 + block_depth
        self.weight = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight.data = self.weight.data / self.weight.data.sum(1, keepdim=True)

        self.weight_c = nn.Parameter(torch.Tensor(self.layer_num, self.layer_num).fill_(1.0).tril())
        self.weight_c.data = self.weight_c.data / self.weight_c.data.sum(1, keepdim=True)

        self.layer_norms = nn.ModuleList(LayerNorm(block_dim) for _ in range(block_depth))
        self.rouge_predictions_norm = nn.ModuleList(LayerNorm(block_dim) for _ in range(block_depth))

    def extra_repr(self):
        return 'n_layers={layer_num}, '.format(**self.__dict__)

    def add(self, layer):
        self.layer_idx += 1

        # first layer
        if self.sum is None:
            self.sum = layer
            self.layers.append(layer)
            return

        # following layer
        if self.normalize_before:
            layer = self.rouge_predictions_norm[self.layer_idx - 2](layer)

        self.layers.append(layer)

    def pop(self):
        assert len(self.layers) > 0
        ret = (torch.stack(self.layers, 0) * self.weight[self.layer_idx - 1, : self.layer_idx].view(-1, 1, 1, 1)).sum(0)

        if self.layer_idx == 1 or self.normalize_before:
            return ret

        return self.layer_norms[self.layer_idx - 2](ret)

    def update(self, layer):

        # update the layer presentation
        if self.normalize_before:
            self.layers[self.layer_idx - 1] = self.layer_norms[self.layer_idx - 2](layer)

    def refine(self):
        assert len(self.layers) > 0
        ret = (torch.stack(self.layers, 0) * self.weight_c[self.layer_idx - 1, : self.layer_idx].view(-1, 1, 1, 1)).sum(
            0)
        if self.layer_idx == 1 or self.normalize_before:
            return ret
        return self.layer_norms[self.layer_idx - 2](ret)

    def clean(self):
        self.sum = None
        self.layer_idx = 0
        self.layers = []


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True, export=False):
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)
