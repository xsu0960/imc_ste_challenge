import math

import torch
import torch.nn as nn

from imc_ste import (
    NoisyConv2d,
    NoisyLinear,
    OnlineGradientProfile,
    estimate_channel_noise,
)
from test_noise import quiet_config


def test_channel_noise_decomposition_separates_aligned_and_orthogonal_error():
    clean = torch.tensor(
        [[[[1.0, -1.0, 2.0, -2.0]], [[1.0, -1.0, 1.0, -1.0]]]]
    )
    orthogonal = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]], [[1.0, 1.0, -1.0, -1.0]]]])
    noisy = clean.clone()
    noisy[:, 0] = 1.2 * clean[:, 0]
    noisy[:, 1] = clean[:, 1] + 0.5 * orthogonal[:, 1]

    estimate = estimate_channel_noise(clean, noisy, channel_dim=1)

    torch.testing.assert_close(
        estimate.bias_ratio_sq, torch.tensor([0.04, 0.0]), atol=1e-6, rtol=1e-6
    )
    torch.testing.assert_close(
        estimate.stochastic_ratio_sq,
        torch.tensor([0.0, 0.25]),
        atol=1e-6,
        rtol=1e-6,
    )
    assert bool(estimate.valid.all())


def test_online_profile_updates_bounded_channel_scale_after_warmup():
    layer = NoisyLinear(2, 2, bias=False, noise_config=quiet_config(), compute_mode="clean")
    model = nn.Sequential(layer)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
    profile = OnlineGradientProfile(
        model,
        ("0",),
        ema_decay=0.0,
        variance_strength=0.0,
        bias_strength=1.0,
        scale_floor=0.25,
        warmup_updates=2,
    )
    inputs = torch.tensor([[1.0, 2.0], [-2.0, 1.0]])

    profile.begin_clean()
    model(inputs)
    profile.begin_noisy()
    model(2 * inputs)
    profile.finalize_batch()
    torch.testing.assert_close(layer.gradient_channel_scale, torch.ones(2))

    profile.begin_clean()
    model(inputs)
    profile.begin_noisy()
    model(2 * inputs)
    profile.finalize_batch()
    expected = torch.full((2,), 1 / math.sqrt(2))
    torch.testing.assert_close(
        layer.gradient_channel_scale, expected, atol=1e-6, rtol=1e-6
    )
    summary = profile.summary()
    assert summary["matched_calls"] == 2
    assert summary["initialized_layers"] == 1
    profile.close()


def test_online_channel_scale_changes_linear_ste_gradient():
    layer = NoisyLinear(
        2,
        2,
        bias=False,
        noise_config=quiet_config(),
        compute_mode="variance_aware_ste",
    )
    with torch.no_grad():
        layer.weight.copy_(torch.eye(2))
        layer.gradient_channel_scale.copy_(torch.tensor([0.25, 0.75]))
    inputs = torch.tensor([[1.0, 2.0]], requires_grad=True)

    layer(inputs).sum().backward()

    torch.testing.assert_close(inputs.grad, torch.tensor([[0.25, 0.75]]))


def test_online_channel_scale_changes_grouped_conv_ste_gradient():
    layer = NoisyConv2d(
        2,
        2,
        kernel_size=1,
        groups=2,
        bias=False,
        noise_config=quiet_config(),
        compute_mode="variance_aware_ste",
    )
    with torch.no_grad():
        layer.weight.fill_(1.0)
        layer.gradient_channel_scale.copy_(torch.tensor([0.2, 0.8]))
    inputs = torch.ones(1, 2, 2, 2, requires_grad=True)

    layer(inputs).sum().backward()

    expected = torch.empty_like(inputs)
    expected[:, 0].fill_(0.2)
    expected[:, 1].fill_(0.8)
    torch.testing.assert_close(inputs.grad, expected)
