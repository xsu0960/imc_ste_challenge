from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from .layers import NoisyConv2d, NoisyLinear


@dataclass(frozen=True)
class ChannelNoiseEstimate:
    """Per-output-channel decomposition of a paired noisy residual."""

    stochastic_ratio_sq: torch.Tensor
    bias_ratio_sq: torch.Tensor
    valid: torch.Tensor
    samples_per_channel: int


def estimate_channel_noise(
    clean: torch.Tensor,
    noisy: torch.Tensor,
    *,
    channel_dim: int,
    eps: float = 1e-6,
) -> ChannelNoiseEstimate:
    """Project the noisy residual onto the clean signal, channel by channel.

    The aligned component captures systematic gain-like error. The orthogonal
    residual is treated as stochastic/unmodelled error. This decomposition only
    needs one paired clean/noisy read and therefore can run inside training.
    """

    if clean.shape != noisy.shape:
        raise ValueError("paired clean/noisy outputs must have identical shapes")
    if clean.ndim == 0:
        raise ValueError("paired outputs must have at least one dimension")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")

    channel_dim %= clean.ndim
    reduce_dims = tuple(dim for dim in range(clean.ndim) if dim != channel_dim)
    clean_float = clean.detach().float()
    residual = noisy.detach().float() - clean_float
    if reduce_dims:
        signal_m2 = clean_float.square().mean(dim=reduce_dims)
        residual_m2 = residual.square().mean(dim=reduce_dims)
        signal_residual = (clean_float * residual).mean(dim=reduce_dims)
    else:
        signal_m2 = clean_float.square()
        residual_m2 = residual.square()
        signal_residual = clean_float * residual

    valid = torch.isfinite(signal_m2) & torch.isfinite(residual_m2)
    valid &= torch.isfinite(signal_residual) & (signal_m2 > eps)
    safe_signal = signal_m2.clamp_min(eps)
    bias_m2 = signal_residual.square() / safe_signal
    stochastic_m2 = (residual_m2 - bias_m2).clamp_min(0.0)
    bias_ratio_sq = bias_m2 / safe_signal
    stochastic_ratio_sq = stochastic_m2 / safe_signal
    zeros = torch.zeros_like(signal_m2)
    return ChannelNoiseEstimate(
        stochastic_ratio_sq=torch.where(valid, stochastic_ratio_sq, zeros),
        bias_ratio_sq=torch.where(valid, bias_ratio_sq, zeros),
        valid=valid,
        samples_per_channel=clean.numel() // clean.shape[channel_dim],
    )


@dataclass
class _OnlineLayerState:
    stochastic_ratio_sq: torch.Tensor
    bias_ratio_sq: torch.Tensor
    updates: int = 0
    matched_calls: int = 0
    shape_mismatches: int = 0
    skipped_calls: int = 0


class OnlineGradientProfile:
    """Update bounded STE channel confidence from the current training stream."""

    def __init__(
        self,
        model: nn.Module,
        name_prefixes: tuple[str, ...],
        *,
        ema_decay: float = 0.95,
        variance_strength: float = 0.5,
        bias_strength: float = 0.25,
        scale_floor: float = 0.5,
        scale_ceiling: float = 1.0,
        warmup_updates: int = 16,
        eps: float = 1e-6,
    ) -> None:
        if not 0 <= ema_decay < 1:
            raise ValueError("online profile ema_decay must be in [0, 1)")
        if variance_strength < 0 or bias_strength < 0:
            raise ValueError("online profile strengths must be non-negative")
        if not 0 < scale_floor <= scale_ceiling <= 1:
            raise ValueError(
                "online profile scale bounds must satisfy 0 < floor <= ceiling <= 1"
            )
        if warmup_updates < 0:
            raise ValueError("online profile warmup_updates must be non-negative")
        if not math.isfinite(eps) or eps <= 0:
            raise ValueError("online profile eps must be finite and positive")

        self.ema_decay = ema_decay
        self.variance_strength = variance_strength
        self.bias_strength = bias_strength
        self.scale_floor = scale_floor
        self.scale_ceiling = scale_ceiling
        self.warmup_updates = warmup_updates
        self.eps = eps
        self.phase = "disabled"
        self.clean_outputs: dict[str, list[torch.Tensor]] = {}
        self.pending_estimates: dict[str, list[ChannelNoiseEstimate]] = {}
        self.call_indices: dict[str, int] = {}
        self.modules: dict[str, NoisyConv2d | NoisyLinear] = {}
        self.states: dict[str, _OnlineLayerState] = {}
        self.handles: list[torch.utils.hooks.RemovableHandle] = []
        self._epoch_matched_calls = 0
        self._epoch_shape_mismatches = 0
        self._epoch_skipped_calls = 0

        for name, module in model.named_modules():
            if not isinstance(module, (NoisyConv2d, NoisyLinear)):
                continue
            if not any(name.startswith(prefix) for prefix in name_prefixes):
                continue
            module.gradient_channel_scale.fill_(1.0)
            self.modules[name] = module
            self.handles.append(
                module.register_forward_hook(self._make_hook(name, module))
            )

        if not self.modules:
            raise ValueError("online profile did not match any noisy layers")

    @property
    def layer_names(self) -> list[str]:
        return list(self.modules)

    @staticmethod
    def _channel_dim(module: NoisyConv2d | NoisyLinear) -> int:
        return 1 if isinstance(module, NoisyConv2d) else -1

    def _make_hook(self, name: str, module: NoisyConv2d | NoisyLinear):
        def hook(_module, _inputs, output):
            if self.phase == "disabled" or not isinstance(output, torch.Tensor):
                return
            call_index = self.call_indices.get(name, 0)
            self.call_indices[name] = call_index + 1
            if self.phase == "clean":
                self.clean_outputs.setdefault(name, []).append(output.detach())
                return

            clean_values = self.clean_outputs.get(name, [])
            if call_index >= len(clean_values):
                self._record_skip(name, module)
                return
            clean = clean_values[call_index]
            if clean.shape != output.shape:
                self._record_mismatch(name, module)
                return
            estimate = estimate_channel_noise(
                clean, output, channel_dim=self._channel_dim(module), eps=self.eps
            )
            self.pending_estimates.setdefault(name, []).append(estimate)

        return hook

    def _state_for(
        self, name: str, module: NoisyConv2d | NoisyLinear
    ) -> _OnlineLayerState:
        state = self.states.get(name)
        if state is None:
            channels = module.gradient_channel_scale.numel()
            zeros = module.gradient_channel_scale.new_zeros(channels, dtype=torch.float32)
            state = _OnlineLayerState(zeros.clone(), zeros.clone())
            self.states[name] = state
        return state

    def _record_skip(self, name: str, module: NoisyConv2d | NoisyLinear) -> None:
        state = self._state_for(name, module)
        state.skipped_calls += 1
        self._epoch_skipped_calls += 1

    def _record_mismatch(
        self, name: str, module: NoisyConv2d | NoisyLinear
    ) -> None:
        state = self._state_for(name, module)
        state.shape_mismatches += 1
        self._epoch_shape_mismatches += 1

    @torch.no_grad()
    def _update(
        self,
        name: str,
        module: NoisyConv2d | NoisyLinear,
        estimate: ChannelNoiseEstimate,
        matched_calls: int,
    ) -> None:
        state = self._state_for(name, module)
        if estimate.valid.numel() != module.gradient_channel_scale.numel():
            self._record_mismatch(name, module)
            return

        valid = estimate.valid
        if state.updates == 0:
            state.stochastic_ratio_sq[valid] = estimate.stochastic_ratio_sq[valid]
            state.bias_ratio_sq[valid] = estimate.bias_ratio_sq[valid]
        else:
            keep = self.ema_decay
            update = 1.0 - keep
            state.stochastic_ratio_sq[valid] = (
                state.stochastic_ratio_sq[valid] * keep
                + estimate.stochastic_ratio_sq[valid] * update
            )
            state.bias_ratio_sq[valid] = (
                state.bias_ratio_sq[valid] * keep
                + estimate.bias_ratio_sq[valid] * update
            )
        state.updates += 1
        state.matched_calls += matched_calls
        self._epoch_matched_calls += matched_calls

        if state.updates < self.warmup_updates:
            module.gradient_channel_scale.fill_(1.0)
            return
        total_error = (
            self.variance_strength * state.stochastic_ratio_sq
            + self.bias_strength * state.bias_ratio_sq
        )
        scale = torch.rsqrt(1.0 + total_error).clamp(
            min=self.scale_floor, max=self.scale_ceiling
        )
        module.gradient_channel_scale.copy_(
            scale.to(
                device=module.gradient_channel_scale.device,
                dtype=module.gradient_channel_scale.dtype,
            )
        )

    @torch.no_grad()
    def finalize_batch(self) -> None:
        if self.phase != "noisy":
            raise RuntimeError("online profile batch finalization requires noisy phase")
        for name, estimates in self.pending_estimates.items():
            if not estimates:
                continue
            module = self.modules[name]
            channels = module.gradient_channel_scale.numel()
            weighted_stochastic = module.gradient_channel_scale.new_zeros(
                channels, dtype=torch.float32
            )
            weighted_bias = weighted_stochastic.clone()
            total_weight = weighted_stochastic.clone()
            for estimate in estimates:
                weight = float(estimate.samples_per_channel)
                valid_float = estimate.valid.to(dtype=torch.float32)
                weighted_stochastic.add_(
                    estimate.stochastic_ratio_sq * valid_float, alpha=weight
                )
                weighted_bias.add_(estimate.bias_ratio_sq * valid_float, alpha=weight)
                total_weight.add_(valid_float, alpha=weight)
            valid = total_weight > 0
            safe_weight = total_weight.clamp_min(1.0)
            combined = ChannelNoiseEstimate(
                stochastic_ratio_sq=weighted_stochastic / safe_weight,
                bias_ratio_sq=weighted_bias / safe_weight,
                valid=valid,
                samples_per_channel=0,
            )
            self._update(name, module, combined, len(estimates))
        self.pending_estimates.clear()

    def begin_clean(self) -> None:
        self.phase = "clean"
        self.clean_outputs.clear()
        self.pending_estimates.clear()
        self.call_indices.clear()

    def begin_noisy(self) -> None:
        if self.phase != "clean":
            raise RuntimeError("online profile noisy phase requires a clean phase")
        self.phase = "noisy"
        self.call_indices.clear()

    def disable(self) -> None:
        self.phase = "disabled"
        self.clean_outputs.clear()
        self.pending_estimates.clear()
        self.call_indices.clear()

    def summary(self, *, reset_epoch_counters: bool = False) -> dict[str, float | int]:
        initialized = [
            (name, state)
            for name, state in self.states.items()
            if state.updates > 0
        ]
        scales = [
            module.gradient_channel_scale.detach().float().cpu()
            for module in self.modules.values()
        ]
        concatenated = torch.cat(scales) if scales else torch.ones(1)
        stochastic = [
            state.stochastic_ratio_sq.sqrt().mean().item()
            for _, state in initialized
        ]
        bias = [state.bias_ratio_sq.sqrt().mean().item() for _, state in initialized]
        result: dict[str, float | int] = {
            "layers": len(self.modules),
            "initialized_layers": len(initialized),
            "matched_calls": self._epoch_matched_calls,
            "shape_mismatches": self._epoch_shape_mismatches,
            "skipped_calls": self._epoch_skipped_calls,
            "scale_mean": float(concatenated.mean().item()),
            "scale_min": float(concatenated.min().item()),
            "scale_max": float(concatenated.max().item()),
            "floor_fraction": float(
                (concatenated <= self.scale_floor + 1e-7).float().mean().item()
            ),
            "stochastic_ratio_mean": sum(stochastic) / len(stochastic)
            if stochastic
            else 0.0,
            "bias_ratio_mean": sum(bias) / len(bias) if bias else 0.0,
        }
        if reset_epoch_counters:
            self._epoch_matched_calls = 0
            self._epoch_shape_mismatches = 0
            self._epoch_skipped_calls = 0
        return result

    def rows(self) -> list[dict[str, float | int | str]]:
        rows: list[dict[str, float | int | str]] = []
        for name, module in self.modules.items():
            state = self.states.get(name)
            scale = module.gradient_channel_scale.detach().float().cpu()
            if state is None:
                stochastic_ratio = 0.0
                bias_ratio = 0.0
                updates = matched = mismatches = skipped = 0
            else:
                stochastic_ratio = float(
                    state.stochastic_ratio_sq.sqrt().mean().item()
                )
                bias_ratio = float(state.bias_ratio_sq.sqrt().mean().item())
                updates = state.updates
                matched = state.matched_calls
                mismatches = state.shape_mismatches
                skipped = state.skipped_calls
            if isinstance(module, NoisyLinear):
                kind = "linear"
            elif module.is_depthwise:
                kind = "depthwise"
            elif module.groups == 1 and module.kernel_size == (1, 1):
                kind = "pointwise"
            else:
                kind = "conv"
            rows.append(
                {
                    "name": name,
                    "kind": kind,
                    "channels": scale.numel(),
                    "updates": updates,
                    "matched_calls": matched,
                    "shape_mismatches": mismatches,
                    "skipped_calls": skipped,
                    "stochastic_to_signal": stochastic_ratio,
                    "bias_to_signal": bias_ratio,
                    "gradient_scale_mean": float(scale.mean().item()),
                    "gradient_scale_min": float(scale.min().item()),
                    "gradient_scale_max": float(scale.max().item()),
                    "gradient_scale_floor_fraction": float(
                        (scale <= self.scale_floor + 1e-7).float().mean().item()
                    ),
                }
            )
        return rows

    def close(self) -> None:
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
