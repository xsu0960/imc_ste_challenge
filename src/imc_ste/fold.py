import torch
import torch.nn as nn


def _fold_conv_batchnorm(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> None:
    if conv.out_channels != bn.num_features:
        raise ValueError("Conv2d out_channels must match BatchNorm2d num_features")

    weight = conv.weight.detach()
    if conv.bias is None:
        bias = torch.zeros(conv.out_channels, device=weight.device, dtype=weight.dtype)
    else:
        bias = conv.bias.detach()

    gamma = (
        bn.weight.detach()
        if bn.affine
        else torch.ones_like(bn.running_mean)
    ).to(device=weight.device, dtype=weight.dtype)
    beta = (
        bn.bias.detach()
        if bn.affine
        else torch.zeros_like(bn.running_mean)
    ).to(device=weight.device, dtype=weight.dtype)
    running_mean = bn.running_mean.detach().to(device=weight.device, dtype=weight.dtype)
    running_var = bn.running_var.detach().to(device=weight.device, dtype=weight.dtype)

    scale = gamma / torch.sqrt(running_var + bn.eps)
    folded_weight = weight * scale.reshape(-1, 1, 1, 1)
    folded_bias = (bias - running_mean) * scale + beta

    conv.weight = nn.Parameter(folded_weight)
    conv.bias = nn.Parameter(folded_bias)


def fold_batchnorms(module: nn.Module) -> int:
    """Fold Conv2d + BatchNorm2d pairs inside Sequential containers.

    The transformation preserves eval-mode clean outputs while changing the
    hardware mapping so BatchNorm affine/statistical scaling is absorbed into
    the preceding convolution weights and bias.
    """

    folded = 0
    for child in module.children():
        folded += fold_batchnorms(child)

    if not isinstance(module, nn.Sequential):
        return folded

    names = list(module._modules.keys())
    for conv_name, bn_name in zip(names, names[1:]):
        conv = module._modules[conv_name]
        bn = module._modules[bn_name]
        if isinstance(conv, nn.Conv2d) and isinstance(bn, nn.BatchNorm2d):
            _fold_conv_batchnorm(conv, bn)
            module._modules[bn_name] = nn.Identity()
            folded += 1
    return folded
