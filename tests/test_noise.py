import torch

from imc_ste.noise import (
    NoiseConfig,
    noisy_grouped_matmul,
    noisy_matmul,
    sample_matmul_noise_state,
    slice_matmul_noise_state,
)
from imc_ste.ste import STEGradientProfile, ste_matmul


def quiet_config(**overrides):
    values = {
        "prog_noise_std": 0.0,
        "drift_factor": 0.0,
        "nonlinear_alpha": 0.0,
        "nonlinear_beta": 0.0,
        "output_noise_std": 0.0,
        "quantization_bits": None,
        "crosstalk_factor": 0.0,
        "temperature_factor": 0.0,
        "retention_loss": 0.0,
        "supply_variation_std": 0.0,
        "gradient_floor": 0.1,
    }
    values.update(overrides)
    return NoiseConfig(**values)


def test_quiet_noise_model_matches_matmul():
    input = torch.randn(5, 3)
    weight = torch.randn(3, 4)
    actual = noisy_matmul(input, weight, quiet_config())
    expected = torch.matmul(input, weight)
    torch.testing.assert_close(actual, expected)


def test_shared_read_state_matches_row_chunking_for_weight_noise():
    config = quiet_config(prog_noise_std=0.2, drift_factor=0.1)
    input = torch.randn(11, 5)
    weight = torch.randn(5, 4)
    state = sample_matmul_noise_state(weight)

    full = noisy_matmul(input, weight, config, state)
    chunked = torch.cat(
        [
            noisy_matmul(input[:4], weight, config, state),
            noisy_matmul(input[4:8], weight, config, state),
            noisy_matmul(input[8:], weight, config, state),
        ]
    )

    torch.testing.assert_close(chunked, full)


def test_shared_grouped_read_state_matches_row_chunking():
    config = quiet_config(prog_noise_std=0.2, retention_loss=0.1)
    input = torch.randn(9, 3, 4)
    weight = torch.randn(3, 4, 2)
    state = sample_matmul_noise_state(weight)

    full = noisy_grouped_matmul(input, weight, config, state)
    chunked = torch.cat(
        [
            noisy_grouped_matmul(input[:3], weight, config, state),
            noisy_grouped_matmul(input[3:], weight, config, state),
        ]
    )

    torch.testing.assert_close(chunked, full)


def test_shared_read_state_can_be_sliced_for_mac_tiles():
    config = quiet_config(prog_noise_std=0.2, temperature_factor=0.1)
    input = torch.randn(7, 6)
    weight = torch.randn(6, 3)
    state = sample_matmul_noise_state(weight)

    full = noisy_matmul(input, weight, config, state)
    tiled = sum(
        noisy_matmul(
            input[:, start:end],
            weight[start:end],
            config,
            slice_matmul_noise_state(state, start, end),
        )
        for start, end in ((0, 2), (2, 4), (4, 6))
    )

    torch.testing.assert_close(tiled, full, atol=1e-6, rtol=1e-5)


def test_identity_ste_uses_clean_matmul_gradient():
    input = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()

    ste_matmul(input, weight, NoiseConfig(), strategy="identity").sum().backward()
    torch.matmul(reference_input, reference_weight).sum().backward()

    torch.testing.assert_close(input.grad, reference_input.grad)
    torch.testing.assert_close(weight.grad, reference_weight.grad)


def test_quantized_noise_model_stays_finite_for_zero_input():
    output = noisy_matmul(
        torch.zeros(2, 3),
        torch.zeros(3, 4),
        quiet_config(quantization_bits=8),
    )
    assert torch.isfinite(output).all()


def test_saturation_aware_ste_has_finite_gradients():
    input = torch.full((4, 3), 10.0, requires_grad=True)
    weight = torch.ones(3, 2, requires_grad=True)
    output = ste_matmul(
        input, weight, quiet_config(nonlinear_alpha=0.2), strategy="saturation_aware"
    )
    output.sum().backward()
    assert torch.isfinite(input.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_adaptive_saturation_aware_ste_has_finite_gradients():
    input = torch.full((4, 3), 10.0, requires_grad=True)
    weight = torch.ones(3, 2, requires_grad=True)
    output = ste_matmul(
        input,
        weight,
        quiet_config(nonlinear_alpha=0.2),
        strategy="adaptive_saturation_aware",
    )
    output.sum().backward()
    assert torch.isfinite(input.grad).all()
    assert torch.isfinite(weight.grad).all()


def test_variance_aware_ste_scales_gradient_by_measured_confidence():
    input = torch.randn(5, 3, requires_grad=True)
    weight = torch.randn(3, 4, requires_grad=True)
    reference_input = input.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    profile = STEGradientProfile(
        stochastic_to_signal=1.0,
        variance_strength=3.0,
        scale_floor=0.1,
    )

    ste_matmul(
        input,
        weight,
        quiet_config(),
        strategy="variance_aware",
        gradient_profile=profile,
    ).sum().backward()
    torch.matmul(reference_input, reference_weight).sum().backward()

    torch.testing.assert_close(input.grad, reference_input.grad * 0.5)
    torch.testing.assert_close(weight.grad, reference_weight.grad * 0.5)


def test_variance_aware_ste_accounts_for_repeated_reads():
    single_read = STEGradientProfile(
        stochastic_to_signal=1.0,
        variance_strength=3.0,
        scale_floor=0.1,
        read_repeats=1,
    )
    four_reads = STEGradientProfile(
        stochastic_to_signal=1.0,
        variance_strength=3.0,
        scale_floor=0.1,
        read_repeats=4,
    )
    gradients = []
    for profile in (single_read, four_reads):
        input = torch.ones(2, 3, requires_grad=True)
        weight = torch.ones(3, 1, requires_grad=True)
        ste_matmul(
            input,
            weight,
            quiet_config(),
            strategy="variance_aware",
            gradient_profile=profile,
        ).sum().backward()
        gradients.append(float(weight.grad.norm()))

    assert gradients[1] > gradients[0]
