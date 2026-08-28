import math

import pytest
import torch
import torch.nn as nn

from imc_ste import (
    NoiseConfig,
    NoisyConv2d,
    NoisyLinear,
    activation_scale_regularization_loss,
    apply_activation_range_scaling,
    apply_gradient_noise_statistics,
    apply_layerwise_mapping_gains,
    apply_layerwise_mac_tile_sizes,
    apply_layerwise_noise_scales,
    apply_named_layer_read_repeats,
    apply_output_noise_read_compensation,
    convert_model,
    enable_learnable_activation_scales,
    moment_matched_read_config,
    set_read_approximation,
    set_conv_chunk_rows,
    set_conv_weight_noise_scope,
    set_layer_read_repeats,
)
from test_noise import quiet_config


def test_noisy_linear_shape_and_gradient():
    layer = NoisyLinear(4, 3, noise_config=NoiseConfig(), compute_mode="ste")
    input = torch.randn(2, 4, requires_grad=True)
    output = layer(input)
    assert output.shape == (2, 3)
    output.sum().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_noisy_conv_shape_and_gradient():
    layer = NoisyConv2d(
        3, 5, kernel_size=3, padding=1, noise_config=NoiseConfig(), compute_mode="ste"
    )
    input = torch.randn(2, 3, 8, 8, requires_grad=True)
    output = layer(input)
    assert output.shape == (2, 5, 8, 8)
    output.mean().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_grouped_noisy_conv_matches_clean_conv2d():
    reference = nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2)
    layer = NoisyConv2d(
        4,
        6,
        kernel_size=3,
        padding=1,
        groups=2,
        noise_config=NoiseConfig(),
        compute_mode="clean",
    )
    layer.load_state_dict(reference.state_dict())

    input = torch.randn(2, 4, 8, 8)
    torch.testing.assert_close(layer(input), reference(input))


def test_grouped_noisy_conv_ste_matches_conv2d_with_quiet_noise():
    reference = nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2)
    layer = NoisyConv2d(
        4,
        6,
        kernel_size=3,
        padding=1,
        groups=2,
        noise_config=quiet_config(),
        compute_mode="ste",
    )
    layer.load_state_dict(reference.state_dict())

    input = torch.randn(2, 4, 8, 8, requires_grad=True)
    output = layer(input)
    torch.testing.assert_close(output, reference(input))
    output.mean().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_chunked_noisy_conv_matches_conv2d_with_quiet_noise():
    reference = nn.Conv2d(3, 5, kernel_size=3, padding=1, stride=2)
    layer = NoisyConv2d(
        3,
        5,
        kernel_size=3,
        padding=1,
        stride=2,
        noise_config=quiet_config(),
        compute_mode="ste",
    )
    layer.conv_chunk_rows = 2
    layer.load_state_dict(reference.state_dict())

    input = torch.randn(2, 3, 17, 19, requires_grad=True)
    output = layer(input)
    torch.testing.assert_close(output, reference(input))
    output.mean().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_chunked_grouped_noisy_conv_matches_conv2d_with_quiet_noise():
    reference = nn.Conv2d(4, 6, kernel_size=3, padding=1, groups=2)
    layer = NoisyConv2d(
        4,
        6,
        kernel_size=3,
        padding=1,
        groups=2,
        noise_config=quiet_config(),
        compute_mode="sat_aware_ste",
    )
    layer.conv_chunk_rows = 3
    layer.load_state_dict(reference.state_dict())

    input = torch.randn(2, 4, 11, 13, requires_grad=True)
    output = layer(input)
    torch.testing.assert_close(output, reference(input))
    output.mean().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_chunked_noisy_conv_reuses_weight_state_per_read():
    config = quiet_config(
        prog_noise_std=0.2,
        drift_factor=0.05,
        retention_loss=0.03,
        temperature_factor=0.02,
    )
    reference = NoisyConv2d(
        3, 5, kernel_size=3, padding=1, noise_config=config, compute_mode="noise"
    )
    chunked = NoisyConv2d(
        3, 5, kernel_size=3, padding=1, noise_config=config, compute_mode="noise"
    )
    chunked.load_state_dict(reference.state_dict())
    chunked.conv_chunk_rows = 2
    reference.read_repeats = 3
    chunked.read_repeats = 3
    input = torch.randn(2, 3, 9, 11)

    torch.manual_seed(123)
    expected = reference(input)
    torch.manual_seed(123)
    actual = chunked(input)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_chunked_grouped_conv_reuses_weight_state_per_read():
    config = quiet_config(prog_noise_std=0.2, drift_factor=0.05)
    reference = NoisyConv2d(
        4,
        6,
        kernel_size=3,
        padding=1,
        groups=2,
        noise_config=config,
        compute_mode="sat_aware_ste",
    )
    chunked = NoisyConv2d(
        4,
        6,
        kernel_size=3,
        padding=1,
        groups=2,
        noise_config=config,
        compute_mode="sat_aware_ste",
    )
    chunked.load_state_dict(reference.state_dict())
    chunked.conv_chunk_rows = 3
    input = torch.randn(1, 4, 10, 12)

    torch.manual_seed(321)
    expected = reference(input)
    torch.manual_seed(321)
    actual = chunked(input)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_legacy_chunk_scope_resamples_weight_state():
    config = quiet_config(prog_noise_std=0.3)
    shared = NoisyConv2d(
        3, 4, kernel_size=3, padding=1, noise_config=config, compute_mode="noise"
    )
    legacy = NoisyConv2d(
        3, 4, kernel_size=3, padding=1, noise_config=config, compute_mode="noise"
    )
    legacy.load_state_dict(shared.state_dict())
    shared.conv_chunk_rows = 2
    legacy.conv_chunk_rows = 2
    legacy.weight_noise_scope = "chunk"
    input = torch.randn(1, 3, 8, 8)

    torch.manual_seed(99)
    shared_output = shared(input)
    torch.manual_seed(99)
    legacy_output = legacy(input)

    assert not torch.allclose(shared_output, legacy_output)


def test_depthwise_noisy_conv_supports_padding_modes_and_gradient():
    reference = nn.Conv2d(
        4, 4, kernel_size=3, padding=1, groups=4, bias=False, padding_mode="reflect"
    )
    layer = NoisyConv2d(
        4,
        4,
        kernel_size=3,
        padding=1,
        groups=4,
        bias=False,
        padding_mode="reflect",
        noise_config=NoiseConfig(),
        compute_mode="clean",
    )
    layer.load_state_dict(reference.state_dict())

    input = torch.randn(2, 4, 8, 8, requires_grad=True)
    output = layer(input)
    torch.testing.assert_close(output, reference(input))
    output.mean().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_depthwise_clean_hybrid_mode_matches_clean_conv2d():
    reference = nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4)
    layer = NoisyConv2d(
        4,
        4,
        kernel_size=3,
        padding=1,
        groups=4,
        noise_config=NoiseConfig(prog_noise_std=10.0, output_noise_std=10.0),
        compute_mode="dw_clean_sat_aware_ste",
    )
    layer.load_state_dict(reference.state_dict())

    input = torch.randn(2, 4, 8, 8, requires_grad=True)
    output = layer(input)
    torch.testing.assert_close(output, reference(input))
    output.mean().backward()
    assert input.grad is not None
    assert layer.weight.grad is not None


def test_layerwise_noise_scaling_targets_selected_layer_types():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Conv2d(4, 8, kernel_size=1),
        nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 2),
    )
    base = NoiseConfig(prog_noise_std=0.02, output_noise_std=0.04)
    converted = convert_model(model, base, compute_mode="noise")

    counts = apply_layerwise_noise_scales(
        converted,
        base,
        depthwise_noise_scale=0.25,
        pointwise_noise_scale=0.5,
        linear_noise_scale=0.75,
    )

    conv3x3, pointwise, depthwise = [
        module for module in converted.modules() if isinstance(module, NoisyConv2d)
    ]
    linear = next(module for module in converted.modules() if isinstance(module, NoisyLinear))

    assert counts == {"depthwise": 1, "pointwise": 1, "linear": 1}
    assert math.isclose(conv3x3.noise_config.prog_noise_std, 0.02)
    assert math.isclose(pointwise.noise_config.prog_noise_std, 0.01)
    assert math.isclose(depthwise.noise_config.prog_noise_std, 0.005)
    assert math.isclose(linear.noise_config.output_noise_std, 0.03)


def test_layerwise_mapping_gain_preserves_quiet_outputs():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Conv2d(4, 8, kernel_size=1),
        nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 2),
    )
    input = torch.randn(2, 3, 4, 4)
    expected = model(input)
    converted = convert_model(model, quiet_config(), compute_mode="noise")

    counts = apply_layerwise_mapping_gains(
        converted,
        mapping_gain=2.0,
        depthwise_mapping_gain=3.0,
        pointwise_mapping_gain=4.0,
        linear_mapping_gain=5.0,
    )
    actual = converted(input)

    torch.testing.assert_close(actual, expected)
    assert counts == {"conv": 1, "depthwise": 1, "pointwise": 1, "linear": 1}
    conv3x3, pointwise, depthwise = [
        module for module in converted.modules() if isinstance(module, NoisyConv2d)
    ]
    linear = next(module for module in converted.modules() if isinstance(module, NoisyLinear))
    assert math.isclose(conv3x3.mapping_gain, 2.0)
    assert math.isclose(pointwise.mapping_gain, 4.0)
    assert math.isclose(depthwise.mapping_gain, 3.0)
    assert math.isclose(linear.mapping_gain, 5.0)


def test_set_conv_chunk_rows_updates_converted_convolutions():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    )
    converted = convert_model(model, quiet_config(), compute_mode="ste")
    count = set_conv_chunk_rows(converted, 4)
    assert count == 2
    assert [
        module.conv_chunk_rows
        for module in converted.modules()
        if isinstance(module, NoisyConv2d)
    ] == [4, 4]


def test_set_conv_weight_noise_scope_updates_converted_convolutions():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Conv2d(4, 4, kernel_size=1),
        nn.Flatten(),
        nn.Linear(4 * 8 * 8, 2),
    )
    converted = convert_model(model, quiet_config(), compute_mode="noise")

    count = set_conv_weight_noise_scope(converted, "chunk")

    assert count == 2
    assert [
        module.weight_noise_scope
        for module in converted.modules()
        if isinstance(module, NoisyConv2d)
    ] == ["chunk", "chunk"]
    with pytest.raises(ValueError):
        set_conv_weight_noise_scope(converted, "invalid")


def test_set_layer_read_repeats_preserves_quiet_outputs():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Flatten(), nn.Linear(64, 2))
    input = torch.randn(2, 3, 4, 4)
    expected = model(input)
    converted = convert_model(model, quiet_config(), compute_mode="noise")
    count = set_layer_read_repeats(converted, 3)
    actual = converted(input)

    torch.testing.assert_close(actual, expected)
    assert count == 2
    assert [
        module.read_repeats
        for module in converted.modules()
        if isinstance(module, (NoisyConv2d, NoisyLinear))
    ] == [3, 3]


def test_named_layer_read_repeats_uses_longest_matching_prefix():
    class NestedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Module()
            self.backbone.body = nn.Conv2d(3, 4, 3, padding=1)
            self.backbone.fpn = nn.Conv2d(4, 4, 1)
            self.head = nn.Linear(4, 2)

    converted = convert_model(NestedModel(), quiet_config(), compute_mode="noise")
    summary = apply_named_layer_read_repeats(
        converted,
        default_read_repeats=1,
        prefix_read_repeats={"backbone.": 2, "backbone.fpn": 4},
    )

    assert converted.backbone.body.read_repeats == 2
    assert converted.backbone.fpn.read_repeats == 4
    assert converted.head.read_repeats == 1
    assert summary["read_histogram"] == {"2": 1, "4": 1, "1": 1}


def test_layerwise_mac_tiling_preserves_quiet_outputs():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=3, padding=1),
        nn.Conv2d(4, 8, kernel_size=1),
        nn.Conv2d(8, 8, kernel_size=3, padding=1, groups=8),
        nn.Flatten(),
        nn.Linear(8 * 4 * 4, 2),
    )
    input = torch.randn(2, 3, 4, 4)
    expected = model(input)
    converted = convert_model(model, quiet_config(), compute_mode="noise")

    counts = apply_layerwise_mac_tile_sizes(
        converted,
        mac_tile_size=5,
        depthwise_mac_tile_size=2,
        pointwise_mac_tile_size=3,
        linear_mac_tile_size=7,
    )
    actual = converted(input)

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    assert counts == {"conv": 1, "depthwise": 1, "pointwise": 1, "linear": 1}
    conv3x3, pointwise, depthwise = [
        module for module in converted.modules() if isinstance(module, NoisyConv2d)
    ]
    linear = next(module for module in converted.modules() if isinstance(module, NoisyLinear))
    assert conv3x3.mac_tile_size == 5
    assert pointwise.mac_tile_size == 3
    assert depthwise.mac_tile_size == 2
    assert linear.mac_tile_size == 7


def test_convert_model_preserves_clean_output():
    model = nn.Sequential(nn.Conv2d(3, 4, 3, padding=1), nn.Flatten(), nn.Linear(64, 2))
    input = torch.randn(2, 3, 4, 4)
    expected = model(input)
    converted = convert_model(model, NoiseConfig(), compute_mode="clean")
    actual = converted(input)
    torch.testing.assert_close(actual, expected)


def test_gradient_noise_statistics_are_matched_by_layer_name(tmp_path):
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
        nn.Flatten(),
        nn.Linear(4 * 4 * 4, 2),
    )
    converted = convert_model(model, quiet_config(), compute_mode="variance_aware_ste")
    stats_path = tmp_path / "stats.csv"
    stats_path.write_text(
        "name,stochastic_to_signal,bias_to_signal\n"
        "0,0.8,0.4\n"
        "1,1.2,0.6\n"
        "3,0.5,0.2\n"
    )

    counts = apply_gradient_noise_statistics(
        converted,
        stats_path,
        variance_strength=2.0,
        bias_strength=0.5,
        kinds=("pointwise", "depthwise", "linear"),
    )
    pointwise, depthwise = [
        module for module in converted.modules() if isinstance(module, NoisyConv2d)
    ]
    linear = next(module for module in converted.modules() if isinstance(module, NoisyLinear))

    assert counts["pointwise"] == 1
    assert counts["depthwise"] == 1
    assert counts["linear"] == 1
    assert math.isclose(pointwise.gradient_stochastic_to_signal, 0.8)
    assert math.isclose(depthwise.gradient_bias_to_signal, 0.6)
    assert math.isclose(linear.gradient_variance_strength, 2.0)


def test_activation_range_scaling_preserves_quiet_linear_operation(tmp_path):
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
    )
    reference_input = torch.randn(2, 3, 4, 4, requires_grad=True)
    converted_input = reference_input.detach().clone().requires_grad_()
    expected = model(reference_input)
    converted = convert_model(model, quiet_config(), compute_mode="ste")
    stats_path = tmp_path / "activation_stats.csv"
    stats_path.write_text(
        "name,clean_abs_p99_mean\n"
        "0,20\n"
        "1,10\n"
    )

    counts = apply_activation_range_scaling(
        converted,
        stats_path,
        target_abs=4.0,
        kinds=("pointwise", "depthwise"),
    )
    actual = converted(converted_input)
    pointwise, depthwise = [
        module for module in converted.modules() if isinstance(module, NoisyConv2d)
    ]

    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    expected.square().mean().backward()
    actual.square().mean().backward()
    torch.testing.assert_close(
        converted_input.grad, reference_input.grad, atol=1e-6, rtol=1e-5
    )
    for reference_parameter, converted_parameter in zip(
        model.parameters(), converted.parameters()
    ):
        torch.testing.assert_close(
            converted_parameter.grad,
            reference_parameter.grad,
            atol=1e-6,
            rtol=1e-5,
        )
    assert math.isclose(pointwise.activation_scale, 0.2)
    assert math.isclose(depthwise.activation_scale, 0.4)
    assert counts["scaled"] == 2


def test_activation_range_scaling_respects_name_prefixes(tmp_path):
    model = nn.ModuleDict(
        {
            "backbone": nn.Conv2d(3, 4, kernel_size=1),
            "head": nn.Conv2d(3, 4, kernel_size=1),
        }
    )
    converted = convert_model(model, quiet_config(), compute_mode="noise")
    stats_path = tmp_path / "activation_stats.csv"
    stats_path.write_text(
        "name,clean_abs_p99_mean\nbackbone,20\nhead,20\n"
    )

    counts = apply_activation_range_scaling(
        converted,
        stats_path,
        target_abs=4.0,
        kinds=("pointwise",),
        name_prefixes=("backbone",),
    )

    assert math.isclose(converted["backbone"].activation_scale, 0.2)
    assert math.isclose(converted["head"].activation_scale, 1.0)
    assert counts["scaled"] == 1


def test_learnable_activation_scaling_is_bounded_and_checkpointed(tmp_path):
    model = nn.Sequential(nn.Conv2d(3, 4, kernel_size=1, bias=False))
    converted = convert_model(model, quiet_config(), compute_mode="ste")
    stats_path = tmp_path / "activation_stats.csv"
    stats_path.write_text("name,clean_abs_p99_mean\n0,20\n")
    apply_activation_range_scaling(
        converted, stats_path, target_abs=4.0, kinds=("pointwise",)
    )
    counts = enable_learnable_activation_scales(
        converted, scale_min=0.1, scale_max=1.0, kinds=("pointwise",)
    )
    layer = next(module for module in converted.modules() if isinstance(module, NoisyConv2d))

    assert counts["enabled"] == 1
    assert layer.activation_scale_logit is not None
    assert 0.1 <= float(layer.activation_scale.detach()) <= 1.0
    assert any(key.endswith("activation_scale_logit") for key in converted.state_dict())
    penalty = activation_scale_regularization_loss(converted)
    torch.testing.assert_close(penalty, torch.zeros_like(penalty), atol=1e-10, rtol=0)

    reference_input = torch.randn(2, 3, 4, 4)
    torch.testing.assert_close(
        converted(reference_input), model(reference_input), atol=1e-6, rtol=1e-5
    )
    with torch.no_grad():
        layer.activation_scale_logit.add_(0.5)
    assert float(activation_scale_regularization_loss(converted)) > 0


def test_output_noise_read_compensation_follows_inverse_scale_budget():
    model = nn.Sequential(
        nn.Conv2d(3, 4, kernel_size=1),
        nn.Conv2d(4, 4, kernel_size=3, padding=1, groups=4),
    )
    converted = convert_model(model, quiet_config(), compute_mode="ste")
    pointwise, depthwise = [
        module for module in converted.modules() if isinstance(module, NoisyConv2d)
    ]
    pointwise.activation_scale = 0.5
    depthwise.activation_scale = 0.25

    counts = apply_output_noise_read_compensation(
        converted,
        base_repeats=1,
        max_repeats=8,
        kinds=("pointwise", "depthwise"),
    )

    assert pointwise.read_repeats == 4
    assert depthwise.read_repeats == 8
    assert counts["read_histogram"] == {"4": 1, "8": 1}
    assert math.isclose(counts["mean_read_repeats"], 6.0)


def test_moment_matched_reads_preserve_systematic_terms_and_scale_random_terms():
    config = NoiseConfig(
        prog_noise_std=0.08,
        nonlinear_alpha=0.2,
        nonlinear_beta=0.1,
        output_noise_std=0.04,
        quantization_bits=8,
    )
    averaged = moment_matched_read_config(config, 4)

    assert math.isclose(averaged.prog_noise_std, 0.04)
    assert math.isclose(averaged.output_noise_std, 0.02)
    assert math.isclose(averaged.nonlinear_alpha, config.nonlinear_alpha)
    assert math.isclose(averaged.nonlinear_beta, config.nonlinear_beta)
    assert averaged.quantization_bits == config.quantization_bits


def test_read_approximation_is_explicitly_selectable():
    converted = convert_model(
        nn.Sequential(nn.Conv2d(3, 4, kernel_size=1)),
        quiet_config(),
        compute_mode="ste",
    )
    layer = next(module for module in converted.modules() if isinstance(module, NoisyConv2d))
    counts = set_read_approximation(
        converted, "moment_matched", kinds=("pointwise",)
    )

    assert counts["pointwise"] == 1
    assert layer.read_approximation == "moment_matched"
