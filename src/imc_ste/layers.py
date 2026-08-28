import os
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .noise import (
    MatmulNoiseState,
    NoiseConfig,
    moment_matched_read_config,
    noisy_grouped_matmul,
    noisy_matmul,
    sample_matmul_noise_state,
    slice_matmul_noise_state,
)
from .ste import STEGradientProfile, ste_grouped_matmul, ste_matmul


ComputeMode = str
READ_APPROXIMATION_MODES = ("exact", "moment_matched")
WEIGHT_NOISE_SCOPES = ("read", "chunk")
COMPUTE_MODES = (
    "clean",
    "noise",
    "ste",
    "sat_aware_ste",
    "adaptive_sat_aware_ste",
    "variance_aware_ste",
    "variance_sat_aware_ste",
    "adaptive_variance_sat_aware_ste",
    "dw_clean_noise",
    "dw_clean_ste",
    "dw_clean_sat_aware_ste",
    "dw_clean_adaptive_sat_aware_ste",
    "dw_clean_variance_aware_ste",
    "dw_clean_variance_sat_aware_ste",
    "dw_clean_adaptive_variance_sat_aware_ste",
)


def _pair(value: Union[int, Tuple[int, int]]) -> Tuple[int, int]:
    return value if isinstance(value, tuple) else (value, value)


def _env_positive_int(name: str) -> int | None:
    raw_value = os.environ.get(name)
    if raw_value is None or raw_value == "":
        return None
    value = int(raw_value)
    return value if value > 0 else None


def _base_compute_mode(mode: ComputeMode) -> ComputeMode:
    return mode.removeprefix("dw_clean_")


def _activation_scale_value(module: nn.Module) -> float | torch.Tensor:
    logit = module.activation_scale_logit
    if logit is None:
        return module._fixed_activation_scale
    scale_range = module.activation_scale_max - module.activation_scale_min
    return module.activation_scale_min + scale_range * torch.sigmoid(logit)


def _set_fixed_activation_scale(module: nn.Module, value: float) -> None:
    value = float(value)
    if not 0 < value <= 1:
        raise ValueError("activation_scale must be in (0, 1]")
    if module.activation_scale_logit is not None:
        raise RuntimeError("cannot replace an enabled learnable activation scale")
    module._fixed_activation_scale = value
    module.activation_scale_reference = value


def _enable_learnable_activation_scale(
    module: nn.Module,
    *,
    scale_min: float,
    scale_max: float,
) -> None:
    if not 0 < scale_min < scale_max <= 1:
        raise ValueError("learnable activation scale bounds must satisfy 0 < min < max <= 1")
    initial = _activation_scale_value(module)
    if isinstance(initial, torch.Tensor):
        initial = float(initial.detach().cpu())
    initial = min(scale_max, max(scale_min, float(initial)))
    normalized = (initial - scale_min) / (scale_max - scale_min)
    eps = 1e-6
    normalized = min(1 - eps, max(eps, normalized))
    logit = torch.logit(module.weight.new_tensor(normalized))
    module.activation_scale_min = scale_min
    module.activation_scale_max = scale_max
    module.activation_scale_reference = initial
    module.activation_scale_logit = nn.Parameter(logit)


def _read_plan(
    config: NoiseConfig,
    read_repeats: int,
    approximation: str,
) -> tuple[int, NoiseConfig]:
    if approximation not in READ_APPROXIMATION_MODES:
        raise ValueError(f"unknown read approximation: {approximation}")
    if approximation == "moment_matched" and read_repeats > 1:
        return 1, moment_matched_read_config(config, read_repeats)
    return read_repeats, config


def _dispatch_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    mode: ComputeMode,
    gradient_profile: STEGradientProfile,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    mode = _base_compute_mode(mode)
    if mode == "clean":
        return torch.matmul(input, weight)
    if mode == "noise":
        return noisy_matmul(input, weight, config, noise_state)
    if mode == "ste":
        return ste_matmul(
            input,
            weight,
            config,
            strategy="identity",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "sat_aware_ste":
        return ste_matmul(
            input,
            weight,
            config,
            strategy="saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "adaptive_sat_aware_ste":
        return ste_matmul(
            input,
            weight,
            config,
            strategy="adaptive_saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "variance_aware_ste":
        return ste_matmul(
            input,
            weight,
            config,
            strategy="variance_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "variance_sat_aware_ste":
        return ste_matmul(
            input,
            weight,
            config,
            strategy="variance_saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "adaptive_variance_sat_aware_ste":
        return ste_matmul(
            input,
            weight,
            config,
            strategy="adaptive_variance_saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    raise ValueError(f"unknown compute mode: {mode}")


def _dispatch_grouped_matmul(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    mode: ComputeMode,
    gradient_profile: STEGradientProfile,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    mode = _base_compute_mode(mode)
    if mode == "noise":
        return noisy_grouped_matmul(input, weight, config, noise_state)
    if mode == "ste":
        return ste_grouped_matmul(
            input,
            weight,
            config,
            strategy="identity",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "sat_aware_ste":
        return ste_grouped_matmul(
            input,
            weight,
            config,
            strategy="saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "adaptive_sat_aware_ste":
        return ste_grouped_matmul(
            input,
            weight,
            config,
            strategy="adaptive_saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "variance_aware_ste":
        return ste_grouped_matmul(
            input,
            weight,
            config,
            strategy="variance_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "variance_sat_aware_ste":
        return ste_grouped_matmul(
            input,
            weight,
            config,
            strategy="variance_saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    if mode == "adaptive_variance_sat_aware_ste":
        return ste_grouped_matmul(
            input,
            weight,
            config,
            strategy="adaptive_variance_saturation_aware",
            gradient_profile=gradient_profile,
            noise_state=noise_state,
        )
    raise ValueError(f"unknown grouped compute mode: {mode}")


def _dispatch_matmul_tiled(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    mode: ComputeMode,
    tile_size: int | None,
    gradient_profile: STEGradientProfile,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    if tile_size is None or tile_size <= 0 or tile_size >= input.shape[-1]:
        return _dispatch_matmul(
            input, weight, config, mode, gradient_profile, noise_state
        )

    output = None
    for start in range(0, input.shape[-1], tile_size):
        end = min(start + tile_size, input.shape[-1])
        current = _dispatch_matmul(
            input[:, start:end],
            weight[start:end, :],
            config,
            mode,
            gradient_profile,
            slice_matmul_noise_state(noise_state, start, end)
            if noise_state is not None
            else None,
        )
        output = current if output is None else output + current
    return output


def _dispatch_grouped_matmul_tiled(
    input: torch.Tensor,
    weight: torch.Tensor,
    config: NoiseConfig,
    mode: ComputeMode,
    tile_size: int | None,
    gradient_profile: STEGradientProfile,
    noise_state: MatmulNoiseState | None = None,
) -> torch.Tensor:
    if tile_size is None or tile_size <= 0 or tile_size >= input.shape[-1]:
        return _dispatch_grouped_matmul(
            input, weight, config, mode, gradient_profile, noise_state
        )

    output = None
    for start in range(0, input.shape[-1], tile_size):
        end = min(start + tile_size, input.shape[-1])
        current = _dispatch_grouped_matmul(
            input[:, :, start:end],
            weight[:, start:end, :],
            config,
            mode,
            gradient_profile,
            slice_matmul_noise_state(noise_state, start, end)
            if noise_state is not None
            else None,
        )
        output = current if output is None else output + current
    return output


class NoisyLinear(nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        *,
        noise_config: Optional[NoiseConfig] = None,
        compute_mode: ComputeMode = "ste",
        mapping_gain: float = 1.0,
    ):
        super().__init__(in_features, out_features, bias=bias)
        self.noise_config = noise_config or NoiseConfig()
        self.compute_mode = compute_mode
        self.mapping_gain = mapping_gain
        self._fixed_activation_scale = 1.0
        self.register_parameter("activation_scale_logit", None)
        self.activation_scale_min = 0.1
        self.activation_scale_max = 1.0
        self.activation_scale_reference = 1.0
        self.read_repeats = 1
        self.read_approximation = "exact"
        self.mac_tile_size: int | None = None
        self.gradient_stochastic_to_signal = 0.0
        self.gradient_bias_to_signal = 0.0
        self.gradient_variance_strength = 0.0
        self.gradient_bias_strength = 0.0
        self.gradient_scale_floor = 0.25
        self.register_buffer(
            "gradient_channel_scale", torch.ones(out_features), persistent=False
        )

    @property
    def activation_scale(self) -> float | torch.Tensor:
        return _activation_scale_value(self)

    @activation_scale.setter
    def activation_scale(self, value: float) -> None:
        _set_fixed_activation_scale(self, value)

    def enable_learnable_activation_scale(
        self, *, scale_min: float = 0.1, scale_max: float = 1.0
    ) -> None:
        _enable_learnable_activation_scale(
            self, scale_min=scale_min, scale_max=scale_max
        )

    def _gradient_profile(self) -> STEGradientProfile:
        return STEGradientProfile(
            stochastic_to_signal=self.gradient_stochastic_to_signal,
            bias_to_signal=self.gradient_bias_to_signal,
            variance_strength=self.gradient_variance_strength,
            bias_strength=self.gradient_bias_strength,
            scale_floor=self.gradient_scale_floor,
            read_repeats=max(1, self.read_repeats),
            channel_scale=self.gradient_channel_scale,
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.compute_mode == "clean":
            return F.linear(input, self.weight, self.bias)
        original_shape = input.shape[:-1]
        activation_scale = self.activation_scale
        matrix_input = input.reshape(-1, input.shape[-1]) * activation_scale
        matrix_weight = (self.weight * self.mapping_gain).transpose(0, 1)
        output = None
        read_repeats = max(1, self.read_repeats)
        physical_repeats, read_config = _read_plan(
            self.noise_config, read_repeats, self.read_approximation
        )
        for _ in range(physical_repeats):
            share_across_tiles = (
                self.mac_tile_size is not None
                and 0 < self.mac_tile_size < matrix_input.shape[-1]
            )
            noise_state = (
                sample_matmul_noise_state(matrix_weight)
                if share_across_tiles
                else None
            )
            current_output = _dispatch_matmul_tiled(
                matrix_input,
                matrix_weight,
                read_config,
                self.compute_mode,
                self.mac_tile_size,
                self._gradient_profile(),
                noise_state,
            )
            output = current_output if output is None else output + current_output
        output = output / physical_repeats
        output = output / (self.mapping_gain * activation_scale)
        if self.bias is not None:
            output = output + self.bias
        return output.reshape(*original_shape, self.out_features)


class NoisyConv2d(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        *,
        noise_config: Optional[NoiseConfig] = None,
        compute_mode: ComputeMode = "ste",
        mapping_gain: float = 1.0,
    ):
        if isinstance(padding, str):
            raise NotImplementedError("NoisyConv2d expects explicit numeric padding")
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
            padding_mode=padding_mode,
        )
        self.noise_config = noise_config or NoiseConfig()
        self.compute_mode = compute_mode
        self.mapping_gain = mapping_gain
        self._fixed_activation_scale = 1.0
        self.register_parameter("activation_scale_logit", None)
        self.activation_scale_min = 0.1
        self.activation_scale_max = 1.0
        self.activation_scale_reference = 1.0
        self.read_repeats = 1
        self.read_approximation = "exact"
        self.mac_tile_size: int | None = None
        self.conv_chunk_rows = _env_positive_int("IMC_STE_CONV_CHUNK_ROWS")
        self.weight_noise_scope = "read"
        self.gradient_stochastic_to_signal = 0.0
        self.gradient_bias_to_signal = 0.0
        self.gradient_variance_strength = 0.0
        self.gradient_bias_strength = 0.0
        self.gradient_scale_floor = 0.25
        self.register_buffer(
            "gradient_channel_scale", torch.ones(out_channels), persistent=False
        )

    @property
    def activation_scale(self) -> float | torch.Tensor:
        return _activation_scale_value(self)

    @activation_scale.setter
    def activation_scale(self, value: float) -> None:
        _set_fixed_activation_scale(self, value)

    def enable_learnable_activation_scale(
        self, *, scale_min: float = 0.1, scale_max: float = 1.0
    ) -> None:
        _enable_learnable_activation_scale(
            self, scale_min=scale_min, scale_max=scale_max
        )

    @property
    def is_depthwise(self) -> bool:
        return self.groups == self.in_channels and self.out_channels == self.in_channels

    def _gradient_profile(self) -> STEGradientProfile:
        return STEGradientProfile(
            stochastic_to_signal=self.gradient_stochastic_to_signal,
            bias_to_signal=self.gradient_bias_to_signal,
            variance_strength=self.gradient_variance_strength,
            bias_strength=self.gradient_bias_strength,
            scale_floor=self.gradient_scale_floor,
            read_repeats=max(1, self.read_repeats),
            channel_scale=self.gradient_channel_scale,
        )

    def _output_hw(self, input_h: int, input_w: int) -> tuple[int, int]:
        kernel_h, kernel_w = _pair(self.kernel_size)
        stride_h, stride_w = _pair(self.stride)
        pad_h, pad_w = _pair(self.padding)
        dilation_h, dilation_w = _pair(self.dilation)
        output_h = (
            input_h + 2 * pad_h - dilation_h * (kernel_h - 1) - 1
        ) // stride_h + 1
        output_w = (
            input_w + 2 * pad_w - dilation_w * (kernel_w - 1) - 1
        ) // stride_w + 1
        return output_h, output_w

    def _padded_input(self, input: torch.Tensor) -> torch.Tensor:
        pad_h, pad_w = _pair(self.padding)
        if pad_h == 0 and pad_w == 0:
            return input
        padding = (pad_w, pad_w, pad_h, pad_h)
        if self.padding_mode == "zeros":
            return F.pad(input, padding)
        return F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode)

    def _patch_output(
        self,
        patches: torch.Tensor,
        matrix_weight: torch.Tensor,
        physical_repeats: int,
        read_config: NoiseConfig,
        noise_states: list[MatmulNoiseState | None],
    ) -> torch.Tensor:
        batch_size, patch_size, locations = patches.shape
        patch_size_per_group = patch_size // self.groups
        matrix_input = patches.transpose(1, 2).reshape(
            batch_size * locations, self.groups, patch_size_per_group
        )
        activation_scale = self.activation_scale
        matrix_input = matrix_input * activation_scale

        output = None
        for noise_state in noise_states:
            if self.groups == 1:
                current_output = _dispatch_matmul_tiled(
                    matrix_input[:, 0, :],
                    matrix_weight[0],
                    read_config,
                    self.compute_mode,
                    self.mac_tile_size,
                    self._gradient_profile(),
                    noise_state,
                )
            else:
                current_output = _dispatch_grouped_matmul_tiled(
                    matrix_input,
                    matrix_weight,
                    read_config,
                    self.compute_mode,
                    self.mac_tile_size,
                    self._gradient_profile(),
                    noise_state,
                ).reshape(batch_size * locations, self.out_channels)
            output = current_output if output is None else output + current_output
        output = output / physical_repeats
        output = output / (self.mapping_gain * activation_scale)
        return output.reshape(batch_size, locations, self.out_channels).transpose(1, 2)

    def _forward_unfolded(
        self,
        input: torch.Tensor,
        output_h: int,
        output_w: int,
        matrix_weight: torch.Tensor,
        physical_repeats: int,
        read_config: NoiseConfig,
        noise_states: list[MatmulNoiseState | None],
    ) -> torch.Tensor:
        batch_size = input.shape[0]
        patches = F.unfold(
            input,
            kernel_size=self.kernel_size,
            dilation=self.dilation,
            padding=0,
            stride=self.stride,
        )
        output = self._patch_output(
            patches,
            matrix_weight,
            physical_repeats,
            read_config,
            noise_states,
        )
        return output.reshape(batch_size, self.out_channels, output_h, output_w)

    def _forward_unfolded_chunked(
        self,
        input: torch.Tensor,
        output_h: int,
        output_w: int,
        chunk_rows: int,
        matrix_weight: torch.Tensor,
        physical_repeats: int,
        read_config: NoiseConfig,
        noise_states: list[MatmulNoiseState | None],
    ) -> torch.Tensor:
        batch_size = input.shape[0]
        stride_h, _ = _pair(self.stride)
        kernel_h, _ = _pair(self.kernel_size)
        dilation_h, _ = _pair(self.dilation)
        chunks = []
        receptive_h = dilation_h * (kernel_h - 1) + 1
        for row_start in range(0, output_h, chunk_rows):
            row_end = min(row_start + chunk_rows, output_h)
            input_start = row_start * stride_h
            input_end = (row_end - 1) * stride_h + receptive_h
            input_chunk = input[:, :, input_start:input_end, :]
            patches = F.unfold(
                input_chunk,
                kernel_size=self.kernel_size,
                dilation=self.dilation,
                padding=0,
                stride=self.stride,
            )
            chunk_h = row_end - row_start
            output = self._patch_output(
                patches,
                matrix_weight,
                physical_repeats,
                read_config,
                noise_states,
            )
            chunks.append(
                output.reshape(batch_size, self.out_channels, chunk_h, output_w)
            )
        return torch.cat(chunks, dim=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self.compute_mode == "clean" or (
            self.compute_mode.startswith("dw_clean_") and self.is_depthwise
        ):
            return self._conv_forward(input, self.weight, self.bias)
        input_h, input_w = input.shape[-2:]
        output_h, output_w = self._output_hw(input_h, input_w)
        input = self._padded_input(input)

        if self.weight_noise_scope not in WEIGHT_NOISE_SCOPES:
            raise ValueError(
                f"unknown weight noise scope: {self.weight_noise_scope}"
            )
        kernel_h, kernel_w = _pair(self.kernel_size)
        patch_size = self.in_channels * kernel_h * kernel_w
        matrix_weight = (
            (self.weight * self.mapping_gain).reshape(
                self.groups,
                self.out_channels // self.groups,
                patch_size // self.groups,
            )
            .transpose(1, 2)
            .contiguous()
        )
        read_repeats = max(1, self.read_repeats)
        physical_repeats, read_config = _read_plan(
            self.noise_config, read_repeats, self.read_approximation
        )
        if self.weight_noise_scope == "read":
            state_weight = matrix_weight[0] if self.groups == 1 else matrix_weight
            noise_states: list[MatmulNoiseState | None] = [
                sample_matmul_noise_state(state_weight)
                for _ in range(physical_repeats)
            ]
        else:
            noise_states = [None] * physical_repeats

        if self.conv_chunk_rows is not None and self.conv_chunk_rows < output_h:
            output = self._forward_unfolded_chunked(
                input,
                output_h,
                output_w,
                self.conv_chunk_rows,
                matrix_weight,
                physical_repeats,
                read_config,
                noise_states,
            )
        else:
            output = self._forward_unfolded(
                input,
                output_h,
                output_w,
                matrix_weight,
                physical_repeats,
                read_config,
                noise_states,
            )

        if self.bias is not None:
            output = output + self.bias[None, :, None, None]
        return output
