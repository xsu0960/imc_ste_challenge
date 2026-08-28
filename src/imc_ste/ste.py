import math
from dataclasses import dataclass

import torch

from .noise import MatmulNoiseState, NoiseConfig, noisy_grouped_matmul, noisy_matmul

STE_STRATEGIES = (
    "identity",
    "saturation_aware",
    "adaptive_saturation_aware",
    "variance_aware",
    "variance_saturation_aware",
    "adaptive_variance_saturation_aware",
)


@dataclass(frozen=True)
class STEGradientProfile:
    """Layer-level statistics used by the noise-aware surrogate gradient.

    The stochastic term is reduced by repeated reads, while the systematic bias
    term is not. Both strengths default to zero so legacy STE behavior is
    preserved when no measured profile is attached to a layer.
    """

    stochastic_to_signal: float = 0.0
    bias_to_signal: float = 0.0
    variance_strength: float = 0.0
    bias_strength: float = 0.0
    scale_floor: float = 0.25
    read_repeats: int = 1
    channel_scale: torch.Tensor | None = None

    def __post_init__(self) -> None:
        for name, value in {
            "stochastic_to_signal": self.stochastic_to_signal,
            "bias_to_signal": self.bias_to_signal,
            "variance_strength": self.variance_strength,
            "bias_strength": self.bias_strength,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not math.isfinite(self.scale_floor) or not 0 < self.scale_floor <= 1:
            raise ValueError("scale_floor must be in (0, 1]")
        if self.read_repeats < 1:
            raise ValueError("read_repeats must be positive")
        if self.channel_scale is not None and self.channel_scale.ndim != 1:
            raise ValueError("channel_scale must be one-dimensional")


def _noise_confidence_scale(profile: STEGradientProfile) -> float:
    stochastic_error = (
        profile.variance_strength
        * profile.stochastic_to_signal**2
        / profile.read_repeats
    )
    systematic_error = profile.bias_strength * profile.bias_to_signal**2
    scale = 1.0 / math.sqrt(1.0 + stochastic_error + systematic_error)
    return max(profile.scale_floor, scale)


def _apply_noise_profile(
    surrogate_grad: torch.Tensor,
    profile: STEGradientProfile,
    *,
    grouped: bool,
) -> torch.Tensor:
    surrogate_grad = surrogate_grad * _noise_confidence_scale(profile)
    channel_scale = profile.channel_scale
    if channel_scale is None:
        return surrogate_grad

    if grouped:
        expected_channels = surrogate_grad.shape[1] * surrogate_grad.shape[2]
        if channel_scale.numel() != expected_channels:
            raise RuntimeError(
                "grouped STE channel scale does not match output channels: "
                f"{channel_scale.numel()} vs {expected_channels}"
            )
        channel_scale = channel_scale.reshape(
            1, surrogate_grad.shape[1], surrogate_grad.shape[2]
        )
    else:
        if channel_scale.numel() != surrogate_grad.shape[-1]:
            raise RuntimeError(
                "STE channel scale does not match output channels: "
                f"{channel_scale.numel()} vs {surrogate_grad.shape[-1]}"
            )
        channel_scale = channel_scale.reshape(1, -1)
    return surrogate_grad * channel_scale


def _saturation_scale(
    ideal_output: torch.Tensor,
    config: NoiseConfig,
    *,
    adaptive: bool,
) -> torch.Tensor:
    pos_scale = config.nonlinear_alpha
    neg_scale = config.nonlinear_alpha + config.nonlinear_beta
    scale = torch.where(
        ideal_output >= 0,
        1 - torch.tanh(pos_scale * ideal_output).square(),
        1 - torch.tanh(neg_scale * ideal_output).square(),
    )
    scale = config.gradient_floor + (1 - config.gradient_floor) * scale

    if adaptive:
        channel_mean = scale.detach().mean(dim=0, keepdim=True)
        scale = scale / channel_mean.clamp_min(config.gradient_scale_eps)
        if config.gradient_scale_clip is not None:
            scale = scale.clamp(max=config.gradient_scale_clip)
    return scale


class _NoisyMatmulSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        config: NoiseConfig,
        strategy: str,
        gradient_profile: STEGradientProfile,
        noise_state: MatmulNoiseState | None,
    ) -> torch.Tensor:
        ideal_output = torch.matmul(input, weight)
        ctx.save_for_backward(input, weight, ideal_output)
        ctx.config = config
        ctx.strategy = strategy
        ctx.gradient_profile = gradient_profile
        return noisy_matmul(input, weight, config, noise_state)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, ideal_output = ctx.saved_tensors

        if ctx.strategy in ("identity", "variance_aware"):
            surrogate_grad = grad_output
        elif ctx.strategy in (
            "saturation_aware",
            "adaptive_saturation_aware",
            "variance_saturation_aware",
            "adaptive_variance_saturation_aware",
        ):
            scale = _saturation_scale(
                ideal_output,
                ctx.config,
                adaptive=ctx.strategy
                in (
                    "adaptive_saturation_aware",
                    "adaptive_variance_saturation_aware",
                ),
            )
            surrogate_grad = grad_output * scale
        else:
            raise RuntimeError(f"unknown STE strategy: {ctx.strategy}")

        if ctx.strategy in (
            "variance_aware",
            "variance_saturation_aware",
            "adaptive_variance_saturation_aware",
        ):
            surrogate_grad = _apply_noise_profile(
                surrogate_grad,
                ctx.gradient_profile,
                grouped=False,
            )

        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.matmul(surrogate_grad, weight.transpose(0, 1))
        if ctx.needs_input_grad[1]:
            grad_weight = torch.matmul(input.transpose(0, 1), surrogate_grad)
        return grad_input, grad_weight, None, None, None, None


def ste_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    strategy: str = "identity",
    gradient_profile: STEGradientProfile | None = None,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    """Noisy forward pass with a controllable surrogate backward pass."""

    if strategy not in STE_STRATEGIES:
        raise ValueError(f"unknown STE strategy: {strategy}")
    profile = gradient_profile or STEGradientProfile()
    return _NoisyMatmulSTE.apply(
        input, weight, config, strategy, profile, noise_state
    )


class _NoisyGroupedMatmulSTE(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        input: torch.Tensor,
        weight: torch.Tensor,
        config: NoiseConfig,
        strategy: str,
        gradient_profile: STEGradientProfile,
        noise_state: MatmulNoiseState | None,
    ) -> torch.Tensor:
        ideal_output = torch.einsum("rgk,gko->rgo", input, weight)
        ctx.save_for_backward(input, weight, ideal_output)
        ctx.config = config
        ctx.strategy = strategy
        ctx.gradient_profile = gradient_profile
        return noisy_grouped_matmul(input, weight, config, noise_state)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        input, weight, ideal_output = ctx.saved_tensors

        if ctx.strategy in ("identity", "variance_aware"):
            surrogate_grad = grad_output
        elif ctx.strategy in (
            "saturation_aware",
            "adaptive_saturation_aware",
            "variance_saturation_aware",
            "adaptive_variance_saturation_aware",
        ):
            scale = _saturation_scale(
                ideal_output,
                ctx.config,
                adaptive=ctx.strategy
                in (
                    "adaptive_saturation_aware",
                    "adaptive_variance_saturation_aware",
                ),
            )
            surrogate_grad = grad_output * scale
        else:
            raise RuntimeError(f"unknown STE strategy: {ctx.strategy}")

        if ctx.strategy in (
            "variance_aware",
            "variance_saturation_aware",
            "adaptive_variance_saturation_aware",
        ):
            surrogate_grad = _apply_noise_profile(
                surrogate_grad,
                ctx.gradient_profile,
                grouped=True,
            )

        grad_input = None
        grad_weight = None
        if ctx.needs_input_grad[0]:
            grad_input = torch.einsum("rgo,gko->rgk", surrogate_grad, weight)
        if ctx.needs_input_grad[1]:
            grad_weight = torch.einsum("rgk,rgo->gko", input, surrogate_grad)
        return grad_input, grad_weight, None, None, None, None


def ste_grouped_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    strategy: str = "identity",
    gradient_profile: STEGradientProfile | None = None,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    """Noisy grouped forward pass with a controllable surrogate backward pass."""

    if strategy not in STE_STRATEGIES:
        raise ValueError(f"unknown STE strategy: {strategy}")
    profile = gradient_profile or STEGradientProfile()
    return _NoisyGroupedMatmulSTE.apply(
        input, weight, config, strategy, profile, noise_state
    )
