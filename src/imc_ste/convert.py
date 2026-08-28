import copy
import csv
import math
from pathlib import Path

import torch
import torch.nn as nn

from .layers import WEIGHT_NOISE_SCOPES, NoisyConv2d, NoisyLinear
from .noise import NoiseConfig, scale_noise_config


def _copy_parameters(source: nn.Module, target: nn.Module) -> None:
    target.load_state_dict(source.state_dict())
    target.train(source.training)


def _convert_child(
    child: nn.Module, noise_config: NoiseConfig, compute_mode: str
) -> nn.Module:
    if isinstance(child, NoisyLinear) or isinstance(child, NoisyConv2d):
        child.noise_config = noise_config
        child.compute_mode = compute_mode
        return child

    if isinstance(child, nn.Linear):
        converted = NoisyLinear(
            child.in_features,
            child.out_features,
            bias=child.bias is not None,
            noise_config=noise_config,
            compute_mode=compute_mode,
        )
        _copy_parameters(child, converted)
        return converted

    if isinstance(child, nn.Conv2d):
        converted = NoisyConv2d(
            child.in_channels,
            child.out_channels,
            child.kernel_size,
            stride=child.stride,
            padding=child.padding,
            dilation=child.dilation,
            groups=child.groups,
            bias=child.bias is not None,
            padding_mode=child.padding_mode,
            noise_config=noise_config,
            compute_mode=compute_mode,
        )
        _copy_parameters(child, converted)
        return converted

    for name, grandchild in list(child.named_children()):
        setattr(child, name, _convert_child(grandchild, noise_config, compute_mode))
    return child


def convert_model(
    model: nn.Module,
    noise_config: NoiseConfig,
    compute_mode: str = "ste",
    *,
    inplace: bool = False,
) -> nn.Module:
    """Replace all supported linear and convolution layers with noisy variants."""

    converted = model if inplace else copy.deepcopy(model)
    return _convert_child(converted, noise_config, compute_mode)


def set_compute_mode(model: nn.Module, compute_mode: str) -> None:
    for module in model.modules():
        if isinstance(module, (NoisyLinear, NoisyConv2d)):
            module.compute_mode = compute_mode


def set_conv_chunk_rows(model: nn.Module, chunk_rows: int | None) -> int:
    count = 0
    if chunk_rows is not None and chunk_rows <= 0:
        chunk_rows = None
    for module in model.modules():
        if isinstance(module, NoisyConv2d):
            module.conv_chunk_rows = chunk_rows
            count += 1
    return count


def set_conv_weight_noise_scope(model: nn.Module, scope: str) -> int:
    """Choose whether convolution weight noise is shared per read or per chunk."""

    if scope not in WEIGHT_NOISE_SCOPES:
        raise ValueError(f"unknown convolution weight noise scope: {scope}")
    count = 0
    for module in model.modules():
        if isinstance(module, NoisyConv2d):
            module.weight_noise_scope = scope
            count += 1
    return count


def set_layer_read_repeats(model: nn.Module, read_repeats: int) -> int:
    if read_repeats < 1:
        raise ValueError("read_repeats must be positive")
    count = 0
    for module in model.modules():
        if isinstance(module, (NoisyLinear, NoisyConv2d)):
            module.read_repeats = read_repeats
            count += 1
    return count


def apply_named_layer_read_repeats(
    model: nn.Module,
    *,
    default_read_repeats: int = 1,
    prefix_read_repeats: dict[str, int] | None = None,
) -> dict[str, object]:
    """Assign logical read counts by module-name prefix.

    The longest matching prefix wins, which lets callers define a broad role such
    as ``backbone.`` and then override a nested role such as ``backbone.fpn.``.
    """

    prefix_read_repeats = prefix_read_repeats or {}
    if default_read_repeats < 1:
        raise ValueError("default_read_repeats must be positive")
    for prefix, repeats in prefix_read_repeats.items():
        if not prefix:
            raise ValueError("read-repeat prefixes must be non-empty")
        if repeats < 1:
            raise ValueError(f"read repeats for prefix {prefix!r} must be positive")

    by_prefix = {prefix: 0 for prefix in prefix_read_repeats}
    histogram: dict[str, int] = {}
    default_count = 0
    layer_count = 0
    total_repeats = 0
    for name, module in model.named_modules():
        if not isinstance(module, (NoisyConv2d, NoisyLinear)):
            continue
        matching_prefixes = [
            prefix for prefix in prefix_read_repeats if name.startswith(prefix)
        ]
        if matching_prefixes:
            prefix = max(matching_prefixes, key=len)
            repeats = prefix_read_repeats[prefix]
            by_prefix[prefix] += 1
        else:
            repeats = default_read_repeats
            default_count += 1
        module.read_repeats = repeats
        histogram[str(repeats)] = histogram.get(str(repeats), 0) + 1
        layer_count += 1
        total_repeats += repeats

    return {
        "layers": layer_count,
        "default": default_count,
        "by_prefix": by_prefix,
        "read_histogram": histogram,
        "mean_read_repeats": total_repeats / layer_count if layer_count else 0.0,
        "total_read_repeats": total_repeats,
    }


def apply_layerwise_read_repeats(
    model: nn.Module,
    *,
    read_repeats: int = 1,
    depthwise_read_repeats: int | None = None,
    pointwise_read_repeats: int | None = None,
    linear_read_repeats: int | None = None,
) -> dict[str, int]:
    for name, repeats in {
        "read_repeats": read_repeats,
        "depthwise_read_repeats": depthwise_read_repeats,
        "pointwise_read_repeats": pointwise_read_repeats,
        "linear_read_repeats": linear_read_repeats,
    }.items():
        if repeats is not None and repeats < 1:
            raise ValueError(f"{name} must be positive")

    counts = {
        "conv": 0,
        "depthwise": 0,
        "pointwise": 0,
        "linear": 0,
    }

    for module in model.modules():
        if isinstance(module, NoisyConv2d):
            repeats = read_repeats
            if module.is_depthwise and depthwise_read_repeats is not None:
                repeats = depthwise_read_repeats
                counts["depthwise"] += 1
            elif (
                module.groups == 1
                and _kernel_pair(module.kernel_size) == (1, 1)
                and pointwise_read_repeats is not None
            ):
                repeats = pointwise_read_repeats
                counts["pointwise"] += 1
            else:
                counts["conv"] += 1
            module.read_repeats = repeats
        elif isinstance(module, NoisyLinear):
            module.read_repeats = (
                linear_read_repeats
                if linear_read_repeats is not None
                else read_repeats
            )
            counts["linear"] += 1

    return counts


def apply_layerwise_mac_tile_sizes(
    model: nn.Module,
    *,
    mac_tile_size: int | None = None,
    depthwise_mac_tile_size: int | None = None,
    pointwise_mac_tile_size: int | None = None,
    linear_mac_tile_size: int | None = None,
) -> dict[str, int]:
    for name, tile_size in {
        "mac_tile_size": mac_tile_size,
        "depthwise_mac_tile_size": depthwise_mac_tile_size,
        "pointwise_mac_tile_size": pointwise_mac_tile_size,
        "linear_mac_tile_size": linear_mac_tile_size,
    }.items():
        if tile_size is not None and tile_size <= 0:
            raise ValueError(f"{name} must be positive")

    counts = {
        "conv": 0,
        "depthwise": 0,
        "pointwise": 0,
        "linear": 0,
    }

    for module in model.modules():
        if isinstance(module, NoisyConv2d):
            tile_size = mac_tile_size
            if module.is_depthwise and depthwise_mac_tile_size is not None:
                tile_size = depthwise_mac_tile_size
                counts["depthwise"] += 1
            elif (
                module.groups == 1
                and _kernel_pair(module.kernel_size) == (1, 1)
                and pointwise_mac_tile_size is not None
            ):
                tile_size = pointwise_mac_tile_size
                counts["pointwise"] += 1
            else:
                counts["conv"] += 1
            module.mac_tile_size = tile_size
        elif isinstance(module, NoisyLinear):
            module.mac_tile_size = (
                linear_mac_tile_size
                if linear_mac_tile_size is not None
                else mac_tile_size
            )
            counts["linear"] += 1

    return counts


def _kernel_pair(value) -> tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def _noisy_layer_kind(module: nn.Module) -> str | None:
    if isinstance(module, NoisyConv2d):
        if module.is_depthwise:
            return "depthwise"
        if module.groups == 1 and _kernel_pair(module.kernel_size) == (1, 1):
            return "pointwise"
        return "conv"
    if isinstance(module, NoisyLinear):
        return "linear"
    return None


def _matches_name_prefixes(
    name: str, name_prefixes: tuple[str, ...] | None
) -> bool:
    return name_prefixes is None or any(
        name.startswith(prefix) for prefix in name_prefixes
    )


def _activation_scale_as_float(module: NoisyConv2d | NoisyLinear) -> float:
    value = module.activation_scale
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu())
    return float(value)


def enable_learnable_activation_scales(
    model: nn.Module,
    *,
    scale_min: float = 0.1,
    scale_max: float = 1.0,
    kinds: tuple[str, ...] = ("depthwise", "pointwise"),
    only_preconditioned: bool = True,
    name_prefixes: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Turn fixed activation preconditioners into bounded trainable scalars."""

    valid_kinds = {"conv", "depthwise", "pointwise", "linear"}
    selected_kinds = set(kinds)
    unknown_kinds = selected_kinds - valid_kinds
    if unknown_kinds:
        raise ValueError(f"unknown learnable activation kinds: {sorted(unknown_kinds)}")

    counts = {kind: 0 for kind in sorted(valid_kinds)}
    counts["enabled"] = 0
    for name, module in model.named_modules():
        if not _matches_name_prefixes(name, name_prefixes):
            continue
        kind = _noisy_layer_kind(module)
        if kind not in selected_kinds:
            continue
        if only_preconditioned and _activation_scale_as_float(module) >= 1.0:
            continue
        module.enable_learnable_activation_scale(
            scale_min=scale_min, scale_max=scale_max
        )
        counts[kind] += 1
        counts["enabled"] += 1
    return counts


def activation_scale_regularization_loss(model: nn.Module) -> torch.Tensor | None:
    """Penalize log-scale drift from the statistics-derived initialization."""

    penalties = []
    for module in model.modules():
        if not isinstance(module, (NoisyConv2d, NoisyLinear)):
            continue
        if module.activation_scale_logit is None:
            continue
        scale = module.activation_scale
        reference = scale.new_tensor(module.activation_scale_reference)
        penalties.append((torch.log(scale) - torch.log(reference)).square())
    if not penalties:
        return None
    return torch.stack(penalties).mean()


def activation_scale_summary(model: nn.Module) -> dict[str, float | int]:
    values = []
    references = []
    for module in model.modules():
        if not isinstance(module, (NoisyConv2d, NoisyLinear)):
            continue
        if module.activation_scale_logit is None:
            continue
        values.append(_activation_scale_as_float(module))
        references.append(float(module.activation_scale_reference))
    if not values:
        return {"count": 0}
    return {
        "count": len(values),
        "min": min(values),
        "mean": sum(values) / len(values),
        "max": max(values),
        "reference_mean": sum(references) / len(references),
        "mean_abs_delta": sum(
            abs(value - reference) for value, reference in zip(values, references)
        )
        / len(values),
    }


def set_read_approximation(
    model: nn.Module,
    approximation: str,
    *,
    kinds: tuple[str, ...] = ("depthwise", "pointwise"),
    name_prefixes: tuple[str, ...] | None = None,
) -> dict[str, int]:
    if approximation not in {"exact", "moment_matched"}:
        raise ValueError(f"unknown read approximation: {approximation}")
    valid_kinds = {"conv", "depthwise", "pointwise", "linear"}
    selected_kinds = set(kinds)
    unknown_kinds = selected_kinds - valid_kinds
    if unknown_kinds:
        raise ValueError(f"unknown read approximation kinds: {sorted(unknown_kinds)}")
    counts = {kind: 0 for kind in sorted(valid_kinds)}
    for name, module in model.named_modules():
        if not _matches_name_prefixes(name, name_prefixes):
            continue
        kind = _noisy_layer_kind(module)
        if kind in selected_kinds:
            module.read_approximation = approximation
            counts[kind] += 1
    return counts


def apply_output_noise_read_compensation(
    model: nn.Module,
    *,
    base_repeats: int = 1,
    max_repeats: int = 8,
    exponent: float = 2.0,
    kinds: tuple[str, ...] = ("depthwise", "pointwise"),
    name_prefixes: tuple[str, ...] | None = None,
) -> dict[str, object]:
    """Allocate real reads to offset inverse-scale output-noise amplification.

    For scale ``a``, additive output-noise variance after inverse scaling grows as
    ``1/a^2``. The default policy therefore uses ``ceil(base/a^2)`` reads, capped
    by an explicit hardware budget.
    """

    if base_repeats < 1 or max_repeats < base_repeats:
        raise ValueError("read compensation requires 1 <= base_repeats <= max_repeats")
    if not math.isfinite(exponent) or exponent < 0:
        raise ValueError("read compensation exponent must be finite and non-negative")
    valid_kinds = {"conv", "depthwise", "pointwise", "linear"}
    selected_kinds = set(kinds)
    unknown_kinds = selected_kinds - valid_kinds
    if unknown_kinds:
        raise ValueError(f"unknown read compensation kinds: {sorted(unknown_kinds)}")

    counts = {kind: 0 for kind in sorted(valid_kinds)}
    histogram: dict[str, int] = {}
    total_repeats = 0
    for name, module in model.named_modules():
        if not _matches_name_prefixes(name, name_prefixes):
            continue
        kind = _noisy_layer_kind(module)
        if kind not in selected_kinds:
            continue
        scale = _activation_scale_as_float(module)
        requested_repeats = base_repeats / scale**exponent
        repeats = min(
            max_repeats,
            max(base_repeats, math.ceil(requested_repeats - 1e-5)),
        )
        module.read_repeats = repeats
        counts[kind] += 1
        histogram[str(repeats)] = histogram.get(str(repeats), 0) + 1
        total_repeats += repeats

    layer_count = sum(counts.values())
    return {
        **counts,
        "layers": layer_count,
        "read_histogram": histogram,
        "mean_read_repeats": total_repeats / layer_count if layer_count else 0.0,
        "total_read_repeats": total_repeats,
    }


def apply_layerwise_noise_scales(
    model: nn.Module,
    base_noise_config: NoiseConfig,
    *,
    depthwise_noise_scale: float | None = None,
    pointwise_noise_scale: float | None = None,
    linear_noise_scale: float | None = None,
) -> dict[str, int]:
    """Assign scaled noise configs to selected layer classes.

    This lets architecture-sensitive operators, such as EfficientNet pointwise
    convolutions, use a calibrated nonideality level while the rest of the model
    keeps the global noise configuration.
    """

    counts = {
        "depthwise": 0,
        "pointwise": 0,
        "linear": 0,
    }
    cached_configs: dict[tuple[str, float], NoiseConfig] = {}

    def scaled(kind: str, scale: float) -> NoiseConfig:
        key = (kind, scale)
        if key not in cached_configs:
            cached_configs[key] = scale_noise_config(base_noise_config, scale)
        return cached_configs[key]

    for module in model.modules():
        if isinstance(module, NoisyConv2d):
            if module.is_depthwise and depthwise_noise_scale is not None:
                module.noise_config = scaled("depthwise", depthwise_noise_scale)
                counts["depthwise"] += 1
            elif (
                module.groups == 1
                and _kernel_pair(module.kernel_size) == (1, 1)
                and pointwise_noise_scale is not None
            ):
                module.noise_config = scaled("pointwise", pointwise_noise_scale)
                counts["pointwise"] += 1
        elif isinstance(module, NoisyLinear) and linear_noise_scale is not None:
            module.noise_config = scaled("linear", linear_noise_scale)
            counts["linear"] += 1
    return counts


def apply_layerwise_mapping_gains(
    model: nn.Module,
    *,
    mapping_gain: float = 1.0,
    depthwise_mapping_gain: float | None = None,
    pointwise_mapping_gain: float | None = None,
    linear_mapping_gain: float | None = None,
) -> dict[str, int]:
    """Set hardware mapping gains for noisy layers.

    A gain maps weights to a larger conductance range and divides the layer
    output by the same factor, preserving the ideal operation while changing
    the signal-to-noise ratio under absolute programming/output noise.
    """

    for name, gain in {
        "mapping_gain": mapping_gain,
        "depthwise_mapping_gain": depthwise_mapping_gain,
        "pointwise_mapping_gain": pointwise_mapping_gain,
        "linear_mapping_gain": linear_mapping_gain,
    }.items():
        if gain is not None and gain <= 0:
            raise ValueError(f"{name} must be positive")

    counts = {
        "conv": 0,
        "depthwise": 0,
        "pointwise": 0,
        "linear": 0,
    }

    for module in model.modules():
        if isinstance(module, NoisyConv2d):
            gain = mapping_gain
            if module.is_depthwise and depthwise_mapping_gain is not None:
                gain = depthwise_mapping_gain
                counts["depthwise"] += 1
            elif (
                module.groups == 1
                and _kernel_pair(module.kernel_size) == (1, 1)
                and pointwise_mapping_gain is not None
            ):
                gain = pointwise_mapping_gain
                counts["pointwise"] += 1
            else:
                counts["conv"] += 1
            module.mapping_gain = gain
        elif isinstance(module, NoisyLinear):
            module.mapping_gain = (
                linear_mapping_gain
                if linear_mapping_gain is not None
                else mapping_gain
            )
            counts["linear"] += 1

    return counts


def apply_gradient_noise_statistics(
    model: nn.Module,
    statistics_path: str | Path,
    *,
    variance_strength: float = 1.0,
    bias_strength: float = 0.0,
    scale_floor: float = 0.25,
    kinds: tuple[str, ...] = ("depthwise", "pointwise"),
    depthwise_variance_strength: float | None = None,
    pointwise_variance_strength: float | None = None,
    linear_variance_strength: float | None = None,
    name_prefixes: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Attach measured per-layer noise statistics to STE layers.

    New diagnostics provide separate stochastic and systematic ratios. Older
    residual-only CSV files remain usable through a conservative fallback.
    """

    if variance_strength < 0 or bias_strength < 0:
        raise ValueError("gradient statistic strengths must be non-negative")
    if not 0 < scale_floor <= 1:
        raise ValueError("gradient statistic scale_floor must be in (0, 1]")
    valid_kinds = {"conv", "depthwise", "pointwise", "linear"}
    selected_kinds = set(kinds)
    unknown_kinds = selected_kinds - valid_kinds
    if unknown_kinds:
        raise ValueError(f"unknown gradient statistic kinds: {sorted(unknown_kinds)}")

    strength_by_kind = {
        "conv": variance_strength,
        "depthwise": depthwise_variance_strength
        if depthwise_variance_strength is not None
        else variance_strength,
        "pointwise": pointwise_variance_strength
        if pointwise_variance_strength is not None
        else variance_strength,
        "linear": linear_variance_strength
        if linear_variance_strength is not None
        else variance_strength,
    }
    for kind, strength in strength_by_kind.items():
        if strength < 0:
            raise ValueError(f"{kind} variance strength must be non-negative")

    with Path(statistics_path).open(newline="") as handle:
        rows = {row["name"]: row for row in csv.DictReader(handle)}

    counts = {
        "conv": 0,
        "depthwise": 0,
        "pointwise": 0,
        "linear": 0,
        "fallback_residual": 0,
        "missing": 0,
    }
    for name, module in model.named_modules():
        if not _matches_name_prefixes(name, name_prefixes):
            continue
        kind = None
        if isinstance(module, NoisyConv2d):
            if module.is_depthwise:
                kind = "depthwise"
            elif module.groups == 1 and _kernel_pair(module.kernel_size) == (1, 1):
                kind = "pointwise"
            else:
                kind = "conv"
        elif isinstance(module, NoisyLinear):
            kind = "linear"

        if kind is None or kind not in selected_kinds:
            continue
        row = rows.get(name)
        if row is None:
            counts["missing"] += 1
            continue

        stochastic_value = row.get("stochastic_to_signal", "")
        bias_value = row.get("bias_to_signal", "")
        if stochastic_value in (None, ""):
            stochastic_value = row.get("residual_to_signal", "0")
            bias_value = "0"
            counts["fallback_residual"] += 1
        stochastic_ratio = float(stochastic_value)
        bias_ratio = float(bias_value or 0.0)
        if not math.isfinite(stochastic_ratio) or stochastic_ratio < 0:
            raise ValueError(f"invalid stochastic ratio for layer {name}")
        if not math.isfinite(bias_ratio) or bias_ratio < 0:
            raise ValueError(f"invalid bias ratio for layer {name}")

        module.gradient_stochastic_to_signal = stochastic_ratio
        module.gradient_bias_to_signal = bias_ratio
        module.gradient_variance_strength = strength_by_kind[kind]
        module.gradient_bias_strength = bias_strength
        module.gradient_scale_floor = scale_floor
        counts[kind] += 1

    return counts


def apply_activation_range_scaling(
    model: nn.Module,
    statistics_path: str | Path,
    *,
    target_abs: float = 4.0,
    scale_floor: float = 0.1,
    kinds: tuple[str, ...] = ("depthwise", "pointwise"),
    statistic_field: str = "clean_abs_p99_mean",
    depthwise_target_abs: float | None = None,
    pointwise_target_abs: float | None = None,
    linear_target_abs: float | None = None,
    name_prefixes: tuple[str, ...] | None = None,
) -> dict[str, int]:
    """Precondition layer inputs to keep noisy MACs in their linear range.

    The noisy layer divides its output by the same scale, so the ideal linear
    operation is unchanged. Unlike weight down-scaling, this does not amplify
    additive programming noise after the inverse scale.
    """

    if target_abs <= 0:
        raise ValueError("activation target_abs must be positive")
    if not 0 < scale_floor <= 1:
        raise ValueError("activation scale_floor must be in (0, 1]")
    valid_kinds = {"conv", "depthwise", "pointwise", "linear"}
    selected_kinds = set(kinds)
    unknown_kinds = selected_kinds - valid_kinds
    if unknown_kinds:
        raise ValueError(f"unknown activation statistic kinds: {sorted(unknown_kinds)}")

    target_by_kind = {
        "conv": target_abs,
        "depthwise": depthwise_target_abs
        if depthwise_target_abs is not None
        else target_abs,
        "pointwise": pointwise_target_abs
        if pointwise_target_abs is not None
        else target_abs,
        "linear": linear_target_abs if linear_target_abs is not None else target_abs,
    }
    for kind, target in target_by_kind.items():
        if target <= 0:
            raise ValueError(f"{kind} activation target must be positive")

    with Path(statistics_path).open(newline="") as handle:
        rows = {row["name"]: row for row in csv.DictReader(handle)}

    counts = {
        "conv": 0,
        "depthwise": 0,
        "pointwise": 0,
        "linear": 0,
        "scaled": 0,
        "floor_limited": 0,
        "missing": 0,
    }
    for name, module in model.named_modules():
        if not _matches_name_prefixes(name, name_prefixes):
            continue
        kind = None
        if isinstance(module, NoisyConv2d):
            if module.is_depthwise:
                kind = "depthwise"
            elif module.groups == 1 and _kernel_pair(module.kernel_size) == (1, 1):
                kind = "pointwise"
            else:
                kind = "conv"
        elif isinstance(module, NoisyLinear):
            kind = "linear"

        if kind is None or kind not in selected_kinds:
            continue
        row = rows.get(name)
        if row is None or row.get(statistic_field, "") in (None, ""):
            counts["missing"] += 1
            continue
        observed_abs = float(row[statistic_field])
        if not math.isfinite(observed_abs) or observed_abs < 0:
            raise ValueError(f"invalid activation statistic for layer {name}")
        raw_scale = 1.0 if observed_abs == 0 else min(
            1.0, target_by_kind[kind] / observed_abs
        )
        module.activation_scale = max(scale_floor, raw_scale)
        counts[kind] += 1
        counts["scaled"] += int(module.activation_scale < 1.0)
        counts["floor_limited"] += int(raw_scale < scale_floor)

    return counts
