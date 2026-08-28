from dataclasses import asdict, dataclass, replace
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class NoiseConfig:
    """Parameters adapted from the organizer-provided sample_noise.py."""

    prog_noise_std: float = 0.01
    drift_factor: float = 0.005
    nonlinear_alpha: float = 0.1
    nonlinear_beta: float = 0.05
    output_noise_std: float = 0.01
    quantization_bits: Optional[int] = None
    crosstalk_factor: float = 0.002
    temperature_factor: float = 0.001
    retention_loss: float = 0.001
    supply_variation_std: float = 0.01
    gradient_floor: float = 0.1
    gradient_scale_eps: float = 1e-6
    gradient_scale_clip: Optional[float] = 4.0


@dataclass(frozen=True)
class MatmulNoiseState:
    """Random state shared by one physical array read.

    Weight-side random draws and the global supply fluctuation belong to the
    physical read, not to an implementation chunk. Input crosstalk and output
    noise remain local to each output tensor produced by ``noisy_matmul``.
    """

    programming_standard_normal: torch.Tensor
    drift_standard_normal: torch.Tensor
    retention_centered_uniform: torch.Tensor
    temperature_standard_normal: torch.Tensor
    supply_standard_normal: torch.Tensor


NOISE_SCALE_FIELDS = (
    "prog_noise_std",
    "drift_factor",
    "nonlinear_alpha",
    "nonlinear_beta",
    "output_noise_std",
    "crosstalk_factor",
    "temperature_factor",
    "retention_loss",
    "supply_variation_std",
)

READ_AVERAGED_NOISE_FIELDS = (
    "prog_noise_std",
    "drift_factor",
    "output_noise_std",
    "crosstalk_factor",
    "temperature_factor",
    "retention_loss",
    "supply_variation_std",
)


def scale_noise_config(config: NoiseConfig, scale: float) -> NoiseConfig:
    if scale < 0:
        raise ValueError("noise scale must be non-negative")
    values = asdict(config)
    for field in NOISE_SCALE_FIELDS:
        values[field] *= scale
    return NoiseConfig(**values)


def moment_matched_read_config(
    config: NoiseConfig, read_repeats: int
) -> NoiseConfig:
    """Approximate an average of independent reads with one noisy draw.

    Random zero-mean terms shrink with ``1 / sqrt(K)`` while the nonlinear
    transfer function and ADC quantizer stay unchanged. This is a training-time
    approximation; exact evaluation still executes all requested reads.
    """

    if read_repeats < 1:
        raise ValueError("read_repeats must be positive")
    if read_repeats == 1:
        return config
    factor = 1.0 / read_repeats**0.5
    updates = {
        field: getattr(config, field) * factor
        for field in READ_AVERAGED_NOISE_FIELDS
    }
    return replace(config, **updates)


def _randn_like(value: torch.Tensor) -> torch.Tensor:
    return torch.randn_like(value)


def _rand_like(value: torch.Tensor) -> torch.Tensor:
    return torch.rand_like(value)


def sample_matmul_noise_state(
    weight: torch.Tensor, *, sample_supply: bool = True
) -> MatmulNoiseState:
    """Sample the weight/supply state for one physical read of an array."""

    if weight.ndim not in (2, 3):
        raise ValueError("matmul noise state expects a 2D or grouped 3D weight")
    return MatmulNoiseState(
        programming_standard_normal=_randn_like(weight),
        drift_standard_normal=_randn_like(weight),
        retention_centered_uniform=_rand_like(weight) - 0.5,
        temperature_standard_normal=_randn_like(weight),
        supply_standard_normal=(
            torch.randn((), device=weight.device, dtype=weight.dtype)
            if sample_supply
            else torch.zeros((), device=weight.device, dtype=weight.dtype)
        ),
    )


def slice_matmul_noise_state(
    state: MatmulNoiseState, start: int, end: int
) -> MatmulNoiseState:
    """Slice a shared state along the input-feature dimension for MAC tiling."""

    weight_ndim = state.programming_standard_normal.ndim
    if weight_ndim == 2:
        index = (slice(start, end), slice(None))
    elif weight_ndim == 3:
        index = (slice(None), slice(start, end), slice(None))
    else:
        raise ValueError("matmul noise state expects a 2D or grouped 3D weight")
    return MatmulNoiseState(
        programming_standard_normal=state.programming_standard_normal[index],
        drift_standard_normal=state.drift_standard_normal[index],
        retention_centered_uniform=state.retention_centered_uniform[index],
        temperature_standard_normal=state.temperature_standard_normal[index],
        supply_standard_normal=state.supply_standard_normal,
    )


def _validate_matmul_noise_state(
    state: MatmulNoiseState, weight: torch.Tensor
) -> None:
    weight_tensors = (
        state.programming_standard_normal,
        state.drift_standard_normal,
        state.retention_centered_uniform,
        state.temperature_standard_normal,
    )
    for value in weight_tensors:
        if value.shape != weight.shape:
            raise ValueError(
                "matmul noise state shape does not match weight: "
                f"{tuple(value.shape)} vs {tuple(weight.shape)}"
            )
        if value.device != weight.device or value.dtype != weight.dtype:
            raise ValueError("matmul noise state must match weight device and dtype")
    if state.supply_standard_normal.ndim != 0:
        raise ValueError("supply noise state must be scalar")
    if (
        state.supply_standard_normal.device != weight.device
        or state.supply_standard_normal.dtype != weight.dtype
    ):
        raise ValueError("supply noise state must match weight device and dtype")


def _apply_weight_noise_state(
    weight: torch.Tensor,
    config: NoiseConfig,
    state: MatmulNoiseState,
) -> torch.Tensor:
    _validate_matmul_noise_state(state, weight)
    prog_noise = state.programming_standard_normal * config.prog_noise_std
    drift_noise = (
        state.drift_standard_normal * config.drift_factor * weight.abs()
    )
    retention_noise = (
        weight * config.retention_loss * state.retention_centered_uniform
    )
    temp_noise = (
        state.temperature_standard_normal
        * config.temperature_factor
        * torch.sqrt(weight.abs())
    )
    return weight + prog_noise + drift_noise + retention_noise + temp_noise


def _asymmetric_saturation(
    result: torch.Tensor, alpha: float, beta: float
) -> torch.Tensor:
    pos_scale = alpha
    neg_scale = alpha + beta

    positive = result if pos_scale == 0 else torch.tanh(pos_scale * result) / pos_scale
    negative = result if neg_scale == 0 else torch.tanh(neg_scale * result) / neg_scale
    return torch.where(result >= 0, positive, negative)


def _adc_quantize(result: torch.Tensor, bits: Optional[int]) -> torch.Tensor:
    if bits is None:
        return result
    if bits < 2:
        raise ValueError("quantization_bits must be at least 2")

    max_val = result.detach().abs().max().clamp_min(torch.finfo(result.dtype).eps)
    scale = (2 ** (bits - 1) - 1) / max_val
    quantized = torch.round(result * scale) / scale
    quant_noise = (_rand_like(result) - 0.5) / scale
    return quantized + quant_noise


def _spatially_correlated_noise(noise: torch.Tensor) -> torch.Tensor:
    if noise.ndim != 2:
        return noise

    kernel = torch.ones((1, 1, 3, 3), device=noise.device, dtype=noise.dtype) / 9
    return F.conv2d(noise[None, None], kernel, padding=1)[0, 0]


def noisy_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    """Run organizer-compatible noisy matrix multiplication.

    Inputs are flattened to two-dimensional matrices by the noisy layer wrappers,
    which also makes the organizer's spatial-correlation operation well-defined.
    """

    if input.ndim != 2 or weight.ndim != 2:
        raise ValueError("noisy_matmul expects two-dimensional matrices")
    if input.shape[1] != weight.shape[0]:
        raise ValueError(
            f"incompatible matmul shapes: {tuple(input.shape)} and {tuple(weight.shape)}"
        )

    state = noise_state or sample_matmul_noise_state(weight, sample_supply=False)
    noisy_weight = _apply_weight_noise_state(weight, config, state)

    if config.crosstalk_factor > 0:
        crosstalk = (
            _randn_like(input)
            * config.crosstalk_factor
            * torch.linalg.vector_norm(input, dim=-1, keepdim=True)
        )
        noisy_input = input + crosstalk
    else:
        noisy_input = input

    result = torch.matmul(noisy_input, noisy_weight)
    result = _asymmetric_saturation(
        result, config.nonlinear_alpha, config.nonlinear_beta
    )
    result = _adc_quantize(result, config.quantization_bits)

    output_noise = _randn_like(result) * config.output_noise_std
    correlation_seed = _randn_like(result) * config.output_noise_std * 0.3
    spatial_corr = _spatially_correlated_noise(correlation_seed)

    supply_standard_normal = (
        state.supply_standard_normal
        if noise_state is not None
        else torch.randn((), device=result.device, dtype=result.dtype)
    )
    supply_variation = 1 + supply_standard_normal * config.supply_variation_std
    return (result + output_noise + spatial_corr) * supply_variation


def noisy_grouped_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    """Noisy grouped matrix multiplication.

    input:  [rows, groups, in_features_per_group]
    weight: [groups, in_features_per_group, out_features_per_group]
    output: [rows, groups, out_features_per_group]
    """

    if input.ndim != 3 or weight.ndim != 3:
        raise ValueError("noisy_grouped_matmul expects three-dimensional tensors")
    if input.shape[1] != weight.shape[0] or input.shape[2] != weight.shape[1]:
        raise ValueError(
            "incompatible grouped matmul shapes: "
            f"{tuple(input.shape)} and {tuple(weight.shape)}"
        )

    state = noise_state or sample_matmul_noise_state(weight, sample_supply=False)
    noisy_weight = _apply_weight_noise_state(weight, config, state)

    if config.crosstalk_factor > 0:
        crosstalk = (
            _randn_like(input)
            * config.crosstalk_factor
            * torch.linalg.vector_norm(input, dim=-1, keepdim=True)
        )
        noisy_input = input + crosstalk
    else:
        noisy_input = input

    result = torch.einsum("rgk,gko->rgo", noisy_input, noisy_weight)
    result = _asymmetric_saturation(
        result, config.nonlinear_alpha, config.nonlinear_beta
    )
    result = _adc_quantize(result, config.quantization_bits)

    output_noise = _randn_like(result) * config.output_noise_std
    correlation_seed = _randn_like(result) * config.output_noise_std * 0.3
    spatial_corr = _spatially_correlated_noise(
        correlation_seed.reshape(correlation_seed.shape[0], -1)
    ).reshape_as(result)

    supply_standard_normal = (
        state.supply_standard_normal
        if noise_state is not None
        else torch.randn((), device=result.device, dtype=result.dtype)
    )
    supply_variation = 1 + supply_standard_normal * config.supply_variation_std
    return (result + output_noise + spatial_corr) * supply_variation
