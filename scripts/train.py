#!/usr/bin/env python3
import argparse
import json
import math
import random
import sys
import time
import urllib.request
import zipfile
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
from torchvision.datasets.folder import default_loader
import torchvision.transforms as transforms
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
DATASETS = ("fake", "cifar10", "cifar100", "tinyimagenet")
TINYIMAGENET_URL = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import (
    COMPUTE_MODES,
    READ_APPROXIMATION_MODES,
    NoiseConfig,
    NoisyConv2d,
    NoisyLinear,
    activation_scale_regularization_loss,
    activation_scale_summary,
    apply_activation_range_scaling,
    apply_gradient_noise_statistics,
    apply_layerwise_mapping_gains,
    apply_layerwise_mac_tile_sizes,
    apply_layerwise_noise_scales,
    apply_layerwise_read_repeats,
    apply_output_noise_read_compensation,
    convert_model,
    enable_learnable_activation_scales,
    scale_noise_config,
    set_compute_mode,
    set_read_approximation,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--dataset", choices=DATASETS, default="fake")
    parser.add_argument("--mode", choices=COMPUTE_MODES, default="ste")
    parser.add_argument(
        "--eval-mode",
        choices=COMPUTE_MODES,
        help="Forward mode used during evaluation; defaults to --mode.",
    )
    parser.add_argument("--seed", type=int, help="Overrides the seed in the config.")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--image-size",
        type=int,
        help="Input image size. Defaults to training.image_size, or dataset native size.",
    )
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--momentum", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--label-smoothing", type=float)
    parser.add_argument(
        "--lr-scheduler",
        choices=["none", "cosine", "multistep"],
        help="Learning-rate schedule. Defaults to the config value or none.",
    )
    parser.add_argument("--lr-milestones", nargs="+", type=int)
    parser.add_argument("--lr-gamma", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--augmentation",
        choices=["standard", "autoaugment"],
        help="Training augmentation policy. Defaults to the config value or standard.",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="Evaluate every N epochs; always evaluates the first and final epoch.",
    )
    parser.add_argument("--grad-clip-norm", type=float)
    parser.add_argument("--max-grad-norm", type=float, dest="grad_clip_norm")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    parser.add_argument(
        "--train-mc-samples",
        type=int,
        default=1,
        help="Average K independent noisy forward passes during training loss computation.",
    )
    parser.add_argument(
        "--mc-consistency-weight",
        type=float,
        default=0.0,
        help="Penalize disagreement among independent noisy training forward passes.",
    )
    parser.add_argument(
        "--mc-consistency-temperature",
        type=float,
        default=2.0,
        help="Temperature used by the multi-read consistency loss.",
    )
    parser.add_argument(
        "--teacher-checkpoint",
        type=Path,
        help="Clean teacher checkpoint for logit distillation during noisy training.",
    )
    parser.add_argument("--distill-alpha", type=float, default=0.0)
    parser.add_argument("--distill-temperature", type=float, default=2.0)
    parser.add_argument(
        "--snr-rms-floor",
        type=float,
        default=0.0,
        help="Target per-output-channel weight RMS for SNR regularization.",
    )
    parser.add_argument("--snr-regularization", type=float, default=0.0)
    parser.add_argument(
        "--snr-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["pointwise", "linear"],
    )
    parser.add_argument(
        "--trainable-scope",
        choices=[
            "all",
            "classifier",
            "last_stage",
            "last_two_stages",
            "activation_scales",
        ],
        default="all",
    )
    parser.add_argument(
        "--freeze-bn",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep BatchNorm modules in eval mode during training.",
    )
    parser.add_argument(
        "--noise-scale",
        type=float,
        default=1.0,
        help="Global multiplier for analog noise and nonideality amplitudes.",
    )
    parser.add_argument(
        "--depthwise-noise-scale",
        type=float,
        help="Additional multiplier for depthwise convolution noise.",
    )
    parser.add_argument(
        "--pointwise-noise-scale",
        type=float,
        help="Additional multiplier for 1x1 pointwise convolution noise.",
    )
    parser.add_argument(
        "--linear-noise-scale",
        type=float,
        help="Additional multiplier for linear layer noise.",
    )
    parser.add_argument(
        "--mapping-gain",
        type=float,
        default=1.0,
        help="Global hardware mapping gain for noisy layers.",
    )
    parser.add_argument(
        "--depthwise-mapping-gain",
        type=float,
        help="Hardware mapping gain for depthwise convolutions.",
    )
    parser.add_argument(
        "--pointwise-mapping-gain",
        type=float,
        help="Hardware mapping gain for 1x1 pointwise convolutions.",
    )
    parser.add_argument(
        "--linear-mapping-gain",
        type=float,
        help="Hardware mapping gain for linear layers.",
    )
    parser.add_argument(
        "--layer-read-repeats",
        type=int,
        default=1,
        help="Average K independent noisy MAC reads inside every noisy layer.",
    )
    parser.add_argument("--depthwise-read-repeats", type=int)
    parser.add_argument("--pointwise-read-repeats", type=int)
    parser.add_argument("--linear-read-repeats", type=int)
    parser.add_argument(
        "--gradient-stat-csv",
        type=Path,
        help="Per-layer noise statistics used by variance-aware STE modes.",
    )
    parser.add_argument(
        "--gradient-stat-strength",
        type=float,
        default=1.0,
        help="Global strength of stochastic variance correction.",
    )
    parser.add_argument("--depthwise-gradient-stat-strength", type=float)
    parser.add_argument("--pointwise-gradient-stat-strength", type=float)
    parser.add_argument("--linear-gradient-stat-strength", type=float)
    parser.add_argument(
        "--gradient-bias-strength",
        type=float,
        default=0.0,
        help="Strength of systematic bias correction; zero leaves it disabled.",
    )
    parser.add_argument(
        "--gradient-stat-floor",
        type=float,
        default=0.25,
        help="Minimum layer confidence multiplier used by variance-aware STE.",
    )
    parser.add_argument(
        "--gradient-stat-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["depthwise", "pointwise"],
    )
    parser.add_argument(
        "--activation-stat-csv",
        type=Path,
        help="Per-layer clean activation statistics for saturation preconditioning.",
    )
    parser.add_argument("--activation-target", type=float, default=4.0)
    parser.add_argument("--depthwise-activation-target", type=float)
    parser.add_argument("--pointwise-activation-target", type=float)
    parser.add_argument("--linear-activation-target", type=float)
    parser.add_argument("--activation-scale-floor", type=float, default=0.1)
    parser.add_argument(
        "--activation-stat-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["depthwise", "pointwise"],
    )
    parser.add_argument(
        "--learnable-activation-scales",
        action="store_true",
        help="Optimize bounded per-layer activation preconditioners initialized from statistics.",
    )
    parser.add_argument(
        "--learnable-activation-scale-max", type=float, default=1.0
    )
    parser.add_argument(
        "--activation-scale-lr-multiplier", type=float, default=1.0
    )
    parser.add_argument(
        "--activation-scale-regularization",
        type=float,
        default=0.0,
        help="Log-scale penalty toward the statistics-derived initialization.",
    )
    parser.add_argument(
        "--output-noise-read-compensation",
        action="store_true",
        help="Allocate per-layer exact reads according to inverse activation scale.",
    )
    parser.add_argument("--output-noise-read-base", type=int, default=1)
    parser.add_argument("--output-noise-read-max", type=int, default=8)
    parser.add_argument("--output-noise-read-exponent", type=float, default=2.0)
    parser.add_argument(
        "--output-noise-read-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["depthwise", "pointwise"],
    )
    parser.add_argument(
        "--train-read-approximation",
        choices=READ_APPROXIMATION_MODES,
        default="exact",
        help="Training-only approximation for the configured logical read count.",
    )
    parser.add_argument(
        "--train-read-approximation-kinds",
        nargs="+",
        choices=["conv", "depthwise", "pointwise", "linear"],
        default=["depthwise", "pointwise"],
    )
    parser.add_argument("--mac-tile-size", type=int)
    parser.add_argument("--depthwise-mac-tile-size", type=int)
    parser.add_argument("--pointwise-mac-tile-size", type=int)
    parser.add_argument("--linear-mac-tile-size", type=int)
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def infer_num_classes(dataset_name: str, configured_num_classes: int) -> int:
    if dataset_name == "cifar10":
        return 10
    if dataset_name == "cifar100":
        return 100
    if dataset_name == "tinyimagenet":
        return 200
    return configured_num_classes


def resolve_weights(model_name: str, model_config: dict):
    weight_spec = model_config.get("weights") or model_config.get("pretrained")
    if weight_spec in (None, False):
        return None
    weight_spec = str(weight_spec).lower()
    if weight_spec in ("", "none", "false"):
        return None

    if model_name == "resnet18":
        enum = torchvision.models.ResNet18_Weights
    elif model_name == "efficientnet_b0":
        enum = torchvision.models.EfficientNet_B0_Weights
    else:
        raise ValueError(f"pretrained weights unsupported for model: {model_name}")

    if weight_spec in ("true", "default", "imagenet", "imagenet1k", "imagenet1k_v1"):
        return enum.DEFAULT
    return enum[weight_spec.upper()]


def build_model(model_config: dict, num_classes: int) -> nn.Module:
    model_name = model_config.get("name", "resnet18").lower()
    weights = resolve_weights(model_name, model_config)
    if model_name == "resnet18":
        model = torchvision.models.resnet18(weights=weights)
        if model_config.get("cifar_stem", True):
            model.conv1 = nn.Conv2d(
                3, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            model.maxpool = nn.Identity()
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model
    if model_name == "efficientnet_b0":
        model = torchvision.models.efficientnet_b0(weights=weights)
        first_conv = model.features[0][0]
        if model_config.get("cifar_stem", True):
            first_conv.stride = (1, 1)
        model.classifier[-1] = nn.Linear(
            model.classifier[-1].in_features, num_classes
        )
        return model
    raise ValueError(f"unsupported model: {model_name}")


class TinyImageNetValDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root: Path,
        class_to_idx: dict[str, int],
        transform=None,
        loader=default_loader,
    ):
        self.root = root
        self.transform = transform
        self.loader = loader
        annotations_path = root / "val" / "val_annotations.txt"
        images_dir = root / "val" / "images"
        if not annotations_path.exists():
            raise FileNotFoundError(f"missing TinyImageNet annotations: {annotations_path}")
        if not images_dir.exists():
            raise FileNotFoundError(f"missing TinyImageNet val images: {images_dir}")

        self.samples: list[tuple[Path, int]] = []
        for line in annotations_path.read_text().splitlines():
            if not line.strip():
                continue
            image_name, wnid, *_ = line.split("\t")
            if wnid not in class_to_idx:
                continue
            self.samples.append((images_dir / image_name, class_to_idx[wnid]))
        if not self.samples:
            raise RuntimeError(f"no TinyImageNet val samples found under {root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        image = self.loader(path)
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def ensure_tinyimagenet(data_root: Path) -> Path:
    dataset_root = data_root / "tiny-imagenet-200"
    if dataset_root.exists():
        return dataset_root

    zip_path = data_root / "tiny-imagenet-200.zip"
    data_root.mkdir(parents=True, exist_ok=True)
    if not zip_path.exists():
        print(f"downloading TinyImageNet from {TINYIMAGENET_URL}")
        urllib.request.urlretrieve(TINYIMAGENET_URL, zip_path)

    print(f"extracting TinyImageNet from {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(data_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"TinyImageNet extraction did not create {dataset_root}")
    return dataset_root


def build_loaders(
    dataset_name: str,
    batch_size: int,
    num_workers: int,
    seed: int,
    num_classes: int,
    augmentation: str,
    image_size: int | None = None,
):
    normalization = {
        "fake": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar10": ((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        "cifar100": ((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        "tinyimagenet": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    }
    mean, std = normalization[dataset_name]
    native_image_size = 64 if dataset_name == "tinyimagenet" else 32
    image_size = image_size or native_image_size
    if image_size > native_image_size:
        resize_size = round(image_size * 256 / 224)
        train_transforms = [
            transforms.Resize(resize_size),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(),
        ]
        eval_transforms = [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
        ]
    else:
        train_transforms = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
        ]
        eval_transforms = []
    if augmentation == "autoaugment":
        autoaugment_policy = (
            transforms.AutoAugmentPolicy.IMAGENET
            if dataset_name == "tinyimagenet"
            else transforms.AutoAugmentPolicy.CIFAR10
        )
        train_transforms.append(
            transforms.AutoAugment(autoaugment_policy)
        )
    train_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )
    train_transform = transforms.Compose(train_transforms)
    eval_transform = transforms.Compose(
        eval_transforms
        + [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    if dataset_name == "fake":
        train_data = torchvision.datasets.FakeData(
            size=512,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=train_transform,
        )
        eval_data = torchvision.datasets.FakeData(
            size=256,
            image_size=(3, 32, 32),
            num_classes=num_classes,
            transform=eval_transform,
        )
    elif dataset_name == "cifar10":
        data_root = PROJECT_ROOT / "data"
        train_data = torchvision.datasets.CIFAR10(
            data_root, train=True, download=True, transform=train_transform
        )
        eval_data = torchvision.datasets.CIFAR10(
            data_root, train=False, download=True, transform=eval_transform
        )
    elif dataset_name == "cifar100":
        data_root = PROJECT_ROOT / "data"
        train_data = torchvision.datasets.CIFAR100(
            data_root, train=True, download=True, transform=train_transform
        )
        eval_data = torchvision.datasets.CIFAR100(
            data_root, train=False, download=True, transform=eval_transform
        )
    else:
        data_root = PROJECT_ROOT / "data"
        tiny_root = ensure_tinyimagenet(data_root)
        train_data = torchvision.datasets.ImageFolder(
            tiny_root / "train", transform=train_transform
        )
        eval_data = TinyImageNetValDataset(
            tiny_root, train_data.class_to_idx, transform=eval_transform
        )

    generator = torch.Generator()
    generator.manual_seed(seed)

    def seed_worker(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, eval_loader


def limit_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def build_scheduler(optimizer, scheduler_name: str, epochs: int, config: dict):
    if scheduler_name == "none":
        return None
    if scheduler_name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    if scheduler_name == "multistep":
        milestones = config.get("lr_milestones", [60, 120, 160])
        gamma = config.get("lr_gamma", 0.2)
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=gamma
        )
    raise ValueError(f"unsupported lr scheduler: {scheduler_name}")


def set_trainable_scope(model: nn.Module, scope: str) -> dict[str, int]:
    for parameter in model.parameters():
        parameter.requires_grad = scope == "all"

    if scope == "activation_scales":
        for name, parameter in model.named_parameters():
            if name.endswith("activation_scale_logit"):
                parameter.requires_grad = True
    elif scope != "all":
        trainable_prefixes = {
            "classifier": ("classifier",),
            "last_stage": ("features.7", "features.8", "classifier"),
            "last_two_stages": ("features.6", "features.7", "features.8", "classifier"),
        }[scope]
        for name, parameter in model.named_parameters():
            if name.startswith(trainable_prefixes):
                parameter.requires_grad = True

    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    total = sum(parameter.numel() for parameter in model.parameters())
    return {"trainable": trainable, "total": total}


def freeze_batchnorm_stats(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def layer_kind(module: nn.Module) -> str | None:
    if isinstance(module, NoisyConv2d):
        kernel = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        if module.is_depthwise:
            return "depthwise"
        if module.groups == 1 and kernel == (1, 1):
            return "pointwise"
        return "conv"
    if isinstance(module, NoisyLinear):
        return "linear"
    return None


def snr_regularization_loss(
    model: nn.Module,
    *,
    rms_floor: float,
    kinds: set[str],
) -> torch.Tensor | None:
    if rms_floor <= 0:
        return None
    penalties = []
    for module in model.modules():
        kind = layer_kind(module)
        if kind not in kinds:
            continue
        weight = module.weight
        per_output = weight.reshape(weight.shape[0], -1)
        rms = per_output.pow(2).mean(dim=1).sqrt()
        penalties.append(torch.relu(rms_floor - rms).square().mean())
    if not penalties:
        return None
    return torch.stack(penalties).mean()


def load_training_checkpoint(model: nn.Module, checkpoint: Path, device) -> None:
    """Load legacy or learnable-scale checkpoints without hiding real mismatches."""

    state = torch.load(checkpoint, map_location=device)
    incompatible = model.load_state_dict(state, strict=False)
    optional_suffix = "activation_scale_logit"
    missing = [
        key for key in incompatible.missing_keys if not key.endswith(optional_suffix)
    ]
    unexpected = [
        key for key in incompatible.unexpected_keys if not key.endswith(optional_suffix)
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"checkpoint mismatch for {checkpoint}: missing={missing}, unexpected={unexpected}"
        )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            "checkpoint contains learned activation scales; rerun with "
            "--learnable-activation-scales and the original activation statistics"
        )


def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    max_batches=None,
    grad_clip_norm=None,
    stop_on_nonfinite=False,
    train_mc_samples=1,
    mc_consistency_weight=0.0,
    mc_consistency_temperature=2.0,
    teacher_model=None,
    distill_alpha=0.0,
    distill_temperature=2.0,
    snr_rms_floor=0.0,
    snr_regularization=0.0,
    snr_kinds=None,
    activation_scale_regularization=0.0,
    freeze_bn=False,
):
    training = optimizer is not None
    model.train(training)
    frozen_bn_count = freeze_batchnorm_stats(model) if training and freeze_bn else 0
    total_loss = 0.0
    total_correct = 0
    total_examples = 0
    nonfinite_batches = 0
    last_grad_norm = None
    total_activation_scale_penalty = 0.0
    snr_kinds = set(snr_kinds or [])
    train_mc_samples = max(1, train_mc_samples if training else 1)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch_index, (images, labels) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images, labels = images.to(device), labels.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
            logits_samples = []
            logits = None
            for _ in range(train_mc_samples):
                current_logits = model(images)
                if training and mc_consistency_weight > 0 and train_mc_samples > 1:
                    logits_samples.append(current_logits)
                logits = current_logits if logits is None else logits + current_logits
            logits = logits / train_mc_samples
            ce_loss = criterion(logits, labels)
            loss = ce_loss
            if (
                training
                and mc_consistency_weight > 0
                and len(logits_samples) > 1
            ):
                temperature = mc_consistency_temperature
                with torch.no_grad():
                    target_probs = torch.nn.functional.softmax(
                        logits.detach() / temperature, dim=1
                    )
                consistency_terms = [
                    torch.nn.functional.kl_div(
                        torch.nn.functional.log_softmax(sample / temperature, dim=1),
                        target_probs,
                        reduction="batchmean",
                    )
                    for sample in logits_samples
                ]
                consistency_loss = (
                    torch.stack(consistency_terms).mean() * (temperature * temperature)
                )
                loss = loss + mc_consistency_weight * consistency_loss
            if training and teacher_model is not None and distill_alpha > 0:
                with torch.no_grad():
                    teacher_logits = teacher_model(images)
                temperature = distill_temperature
                distill_loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits / temperature, dim=1),
                    torch.nn.functional.softmax(teacher_logits / temperature, dim=1),
                    reduction="batchmean",
                ) * (temperature * temperature)
                loss = (1 - distill_alpha) * ce_loss + distill_alpha * distill_loss
            snr_loss = None
            if training and snr_regularization > 0 and snr_rms_floor > 0:
                snr_loss = snr_regularization_loss(
                    model,
                    rms_floor=snr_rms_floor,
                    kinds=snr_kinds,
                )
                if snr_loss is not None:
                    loss = loss + snr_regularization * snr_loss
            activation_scale_penalty = None
            if training and activation_scale_regularization > 0:
                activation_scale_penalty = activation_scale_regularization_loss(model)
                if activation_scale_penalty is not None:
                    loss = (
                        loss
                        + activation_scale_regularization
                        * activation_scale_penalty
                    )
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                if stop_on_nonfinite:
                    return {
                        "loss": total_loss / total_examples
                        if total_examples
                        else None,
                        "accuracy": total_correct / total_examples
                        if total_examples
                        else None,
                        "examples": total_examples,
                        "nonfinite": True,
                        "nonfinite_batches": nonfinite_batches,
                        "stop_reason": "nonfinite_loss",
                        "stop_batch": batch_index,
                    }
                continue

            if training:
                loss.backward()
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    grad_norm_is_finite = bool(torch.isfinite(grad_norm).item())
                    last_grad_norm = (
                        float(grad_norm.detach().cpu())
                        if grad_norm_is_finite
                        else None
                    )
                    if not grad_norm_is_finite:
                        nonfinite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        if stop_on_nonfinite:
                            return {
                                "loss": total_loss / total_examples
                                if total_examples
                                else None,
                                "accuracy": total_correct / total_examples
                                if total_examples
                                else None,
                                "examples": total_examples,
                                "nonfinite": True,
                                "nonfinite_batches": nonfinite_batches,
                                "stop_reason": "nonfinite_gradient",
                                "stop_batch": batch_index,
                                "grad_norm": last_grad_norm,
                            }
                        continue
                optimizer.step()

            total_loss += loss.item() * labels.numel()
            if activation_scale_penalty is not None:
                total_activation_scale_penalty += (
                    float(activation_scale_penalty.detach().cpu()) * labels.numel()
                )
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_examples += labels.numel()

    metrics = {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
        "examples": total_examples,
        "nonfinite": nonfinite_batches > 0,
        "nonfinite_batches": nonfinite_batches,
    }
    if last_grad_norm is not None:
        metrics["grad_norm"] = last_grad_norm
    if total_activation_scale_penalty:
        metrics["activation_scale_penalty"] = (
            total_activation_scale_penalty / max(total_examples, 1)
        )
    if frozen_bn_count:
        metrics["frozen_bn_count"] = frozen_bn_count
    return metrics


def main():
    args = parse_args()
    if args.learnable_activation_scales and args.activation_stat_csv is None:
        raise ValueError("--learnable-activation-scales requires --activation-stat-csv")
    if args.activation_scale_lr_multiplier <= 0:
        raise ValueError("--activation-scale-lr-multiplier must be positive")
    if args.activation_scale_regularization < 0:
        raise ValueError("--activation-scale-regularization must be non-negative")
    config = yaml.safe_load(args.config.read_text())
    seed = args.seed if args.seed is not None else config["seed"]
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = select_device(config["device"])
    training_config = config["training"]
    noise_config = scale_noise_config(NoiseConfig(**config["noise"]), args.noise_scale)

    num_classes = infer_num_classes(args.dataset, config["model"]["num_classes"])
    model = build_model(config["model"], num_classes)
    model = convert_model(model, noise_config, args.mode).to(device)
    layerwise_noise_counts = apply_layerwise_noise_scales(
        model,
        noise_config,
        depthwise_noise_scale=args.depthwise_noise_scale,
        pointwise_noise_scale=args.pointwise_noise_scale,
        linear_noise_scale=args.linear_noise_scale,
    )
    layerwise_mapping_gain_counts = apply_layerwise_mapping_gains(
        model,
        mapping_gain=args.mapping_gain,
        depthwise_mapping_gain=args.depthwise_mapping_gain,
        pointwise_mapping_gain=args.pointwise_mapping_gain,
        linear_mapping_gain=args.linear_mapping_gain,
    )
    layerwise_read_repeat_counts = apply_layerwise_read_repeats(
        model,
        read_repeats=args.layer_read_repeats,
        depthwise_read_repeats=args.depthwise_read_repeats,
        pointwise_read_repeats=args.pointwise_read_repeats,
        linear_read_repeats=args.linear_read_repeats,
    )
    if args.gradient_stat_csv is not None:
        gradient_stat_counts = apply_gradient_noise_statistics(
            model,
            args.gradient_stat_csv,
            variance_strength=args.gradient_stat_strength,
            bias_strength=args.gradient_bias_strength,
            scale_floor=args.gradient_stat_floor,
            kinds=tuple(args.gradient_stat_kinds),
            depthwise_variance_strength=args.depthwise_gradient_stat_strength,
            pointwise_variance_strength=args.pointwise_gradient_stat_strength,
            linear_variance_strength=args.linear_gradient_stat_strength,
        )
    else:
        gradient_stat_counts = {
            "conv": 0,
            "depthwise": 0,
            "pointwise": 0,
            "linear": 0,
            "fallback_residual": 0,
            "missing": 0,
        }
    if args.activation_stat_csv is not None:
        activation_stat_counts = apply_activation_range_scaling(
            model,
            args.activation_stat_csv,
            target_abs=args.activation_target,
            scale_floor=args.activation_scale_floor,
            kinds=tuple(args.activation_stat_kinds),
            depthwise_target_abs=args.depthwise_activation_target,
            pointwise_target_abs=args.pointwise_activation_target,
            linear_target_abs=args.linear_activation_target,
        )
    else:
        activation_stat_counts = {
            "conv": 0,
            "depthwise": 0,
            "pointwise": 0,
            "linear": 0,
            "scaled": 0,
            "floor_limited": 0,
            "missing": 0,
        }
    if args.learnable_activation_scales:
        learnable_activation_counts = enable_learnable_activation_scales(
            model,
            scale_min=args.activation_scale_floor,
            scale_max=args.learnable_activation_scale_max,
            kinds=tuple(args.activation_stat_kinds),
        )
    else:
        learnable_activation_counts = {
            "conv": 0,
            "depthwise": 0,
            "pointwise": 0,
            "linear": 0,
            "enabled": 0,
        }
    layerwise_mac_tile_counts = apply_layerwise_mac_tile_sizes(
        model,
        mac_tile_size=args.mac_tile_size,
        depthwise_mac_tile_size=args.depthwise_mac_tile_size,
        pointwise_mac_tile_size=args.pointwise_mac_tile_size,
        linear_mac_tile_size=args.linear_mac_tile_size,
    )
    if args.checkpoint:
        load_training_checkpoint(model, args.checkpoint, device)
    if args.output_noise_read_compensation:
        output_noise_read_counts = apply_output_noise_read_compensation(
            model,
            base_repeats=args.output_noise_read_base,
            max_repeats=args.output_noise_read_max,
            exponent=args.output_noise_read_exponent,
            kinds=tuple(args.output_noise_read_kinds),
        )
    else:
        output_noise_read_counts = {
            "conv": 0,
            "depthwise": 0,
            "pointwise": 0,
            "linear": 0,
            "layers": 0,
            "read_histogram": {},
            "mean_read_repeats": 0.0,
            "total_read_repeats": 0,
        }
    read_approximation_counts = set_read_approximation(
        model,
        "exact",
        kinds=tuple(args.train_read_approximation_kinds),
    )
    trainable_counts = set_trainable_scope(model, args.trainable_scope)
    teacher_model = None
    if args.teacher_checkpoint is not None:
        teacher_model = build_model(config["model"], num_classes).to(device)
        teacher_model.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
        teacher_model.eval()
        for parameter in teacher_model.parameters():
            parameter.requires_grad = False
    batch_size = args.batch_size or training_config["batch_size"]
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else training_config.get("num_workers", 0)
    )
    augmentation = (
        args.augmentation
        if args.augmentation is not None
        else training_config.get("augmentation", "standard")
    )
    max_train_batches = limit_or_none(args.max_train_batches)
    max_eval_batches = limit_or_none(args.max_eval_batches)
    train_loader, eval_loader = build_loaders(
        args.dataset,
        batch_size,
        num_workers,
        seed,
        num_classes,
        augmentation,
        args.image_size or training_config.get("image_size"),
    )

    learning_rate = (
        args.learning_rate
        if args.learning_rate is not None
        else training_config["learning_rate"]
    )
    momentum = (
        args.momentum if args.momentum is not None else training_config["momentum"]
    )
    weight_decay = (
        args.weight_decay
        if args.weight_decay is not None
        else training_config["weight_decay"]
    )
    base_parameters = []
    activation_scale_parameters = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if name.endswith("activation_scale_logit"):
            activation_scale_parameters.append(parameter)
        else:
            base_parameters.append(parameter)
    if not base_parameters and not activation_scale_parameters:
        raise ValueError(f"no trainable parameters for scope: {args.trainable_scope}")
    optimizer_groups = []
    if base_parameters:
        optimizer_groups.append({"params": base_parameters})
    if activation_scale_parameters:
        optimizer_groups.append(
            {
                "params": activation_scale_parameters,
                "lr": learning_rate * args.activation_scale_lr_multiplier,
                "weight_decay": 0.0,
            }
        )
    optimizer = torch.optim.SGD(
        optimizer_groups,
        lr=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    label_smoothing = (
        args.label_smoothing
        if args.label_smoothing is not None
        else training_config.get("label_smoothing", 0.0)
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    epochs = args.epochs or training_config["epochs"]
    eval_mode = args.eval_mode or args.mode
    scheduler_name = (
        args.lr_scheduler
        if args.lr_scheduler is not None
        else training_config.get("lr_scheduler", "none")
    )
    scheduler_config = dict(training_config)
    if args.lr_milestones is not None:
        scheduler_config["lr_milestones"] = args.lr_milestones
    if args.lr_gamma is not None:
        scheduler_config["lr_gamma"] = args.lr_gamma
    scheduler = build_scheduler(optimizer, scheduler_name, epochs, scheduler_config)
    if args.eval_every < 1:
        raise ValueError("--eval-every must be positive")

    run_dir = PROJECT_ROOT / "runs"
    run_dir.mkdir(exist_ok=True)
    run_path = (
        run_dir / f"{int(time.time())}_{args.dataset}_{args.mode}_seed{seed}.jsonl"
    )
    checkpoint_path = run_path.with_suffix(".pt")
    best_checkpoint_path = run_path.with_name(f"{run_path.stem}_best.pt")
    best_eval_accuracy = -math.inf
    best_epoch = None
    metadata = {
        "config": str(args.config),
        "dataset": args.dataset,
        "seed": seed,
        "model": {**config["model"], "num_classes": num_classes},
        "train_mode": args.mode,
        "eval_mode": eval_mode,
        "device": str(device),
        "batch_size": batch_size,
        "image_size": args.image_size or training_config.get("image_size"),
        "epochs": epochs,
        "learning_rate": learning_rate,
        "momentum": momentum,
        "weight_decay": weight_decay,
        "label_smoothing": label_smoothing,
        "lr_scheduler": scheduler_name,
        "lr_milestones": scheduler_config.get("lr_milestones", ""),
        "lr_gamma": scheduler_config.get("lr_gamma", ""),
        "num_workers": num_workers,
        "augmentation": augmentation,
        "eval_every": args.eval_every,
        "grad_clip_norm": args.grad_clip_norm,
        "stop_on_nonfinite": args.stop_on_nonfinite,
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "noise_scale": args.noise_scale,
        "depthwise_noise_scale": args.depthwise_noise_scale,
        "pointwise_noise_scale": args.pointwise_noise_scale,
        "linear_noise_scale": args.linear_noise_scale,
        "layerwise_noise_counts": layerwise_noise_counts,
        "mapping_gain": args.mapping_gain,
        "depthwise_mapping_gain": args.depthwise_mapping_gain,
        "pointwise_mapping_gain": args.pointwise_mapping_gain,
        "linear_mapping_gain": args.linear_mapping_gain,
        "layerwise_mapping_gain_counts": layerwise_mapping_gain_counts,
        "layer_read_repeats": args.layer_read_repeats,
        "depthwise_read_repeats": args.depthwise_read_repeats,
        "pointwise_read_repeats": args.pointwise_read_repeats,
        "linear_read_repeats": args.linear_read_repeats,
        "layerwise_read_repeat_counts": layerwise_read_repeat_counts,
        "gradient_stat_csv": str(args.gradient_stat_csv)
        if args.gradient_stat_csv
        else "",
        "gradient_stat_strength": args.gradient_stat_strength,
        "depthwise_gradient_stat_strength": args.depthwise_gradient_stat_strength,
        "pointwise_gradient_stat_strength": args.pointwise_gradient_stat_strength,
        "linear_gradient_stat_strength": args.linear_gradient_stat_strength,
        "gradient_bias_strength": args.gradient_bias_strength,
        "gradient_stat_floor": args.gradient_stat_floor,
        "gradient_stat_kinds": args.gradient_stat_kinds,
        "gradient_stat_counts": gradient_stat_counts,
        "activation_stat_csv": str(args.activation_stat_csv)
        if args.activation_stat_csv
        else "",
        "activation_target": args.activation_target,
        "depthwise_activation_target": args.depthwise_activation_target,
        "pointwise_activation_target": args.pointwise_activation_target,
        "linear_activation_target": args.linear_activation_target,
        "activation_scale_floor": args.activation_scale_floor,
        "activation_stat_kinds": args.activation_stat_kinds,
        "activation_stat_counts": activation_stat_counts,
        "learnable_activation_scales": args.learnable_activation_scales,
        "learnable_activation_scale_max": args.learnable_activation_scale_max,
        "learnable_activation_counts": learnable_activation_counts,
        "activation_scale_lr_multiplier": args.activation_scale_lr_multiplier,
        "activation_scale_regularization": args.activation_scale_regularization,
        "activation_scale_summary": activation_scale_summary(model),
        "output_noise_read_compensation": args.output_noise_read_compensation,
        "output_noise_read_base": args.output_noise_read_base,
        "output_noise_read_max": args.output_noise_read_max,
        "output_noise_read_exponent": args.output_noise_read_exponent,
        "output_noise_read_kinds": args.output_noise_read_kinds,
        "output_noise_read_counts": output_noise_read_counts,
        "train_read_approximation": args.train_read_approximation,
        "train_read_approximation_kinds": args.train_read_approximation_kinds,
        "read_approximation_counts": read_approximation_counts,
        "mac_tile_size": args.mac_tile_size,
        "depthwise_mac_tile_size": args.depthwise_mac_tile_size,
        "pointwise_mac_tile_size": args.pointwise_mac_tile_size,
        "linear_mac_tile_size": args.linear_mac_tile_size,
        "layerwise_mac_tile_counts": layerwise_mac_tile_counts,
        "trainable_scope": args.trainable_scope,
        "trainable_parameters": trainable_counts["trainable"],
        "total_parameters": trainable_counts["total"],
        "freeze_bn": args.freeze_bn,
        "train_mc_samples": args.train_mc_samples,
        "mc_consistency_weight": args.mc_consistency_weight,
        "mc_consistency_temperature": args.mc_consistency_temperature,
        "teacher_checkpoint": str(args.teacher_checkpoint) if args.teacher_checkpoint else "",
        "distill_alpha": args.distill_alpha,
        "distill_temperature": args.distill_temperature,
        "snr_rms_floor": args.snr_rms_floor,
        "snr_regularization": args.snr_regularization,
        "snr_kinds": args.snr_kinds,
        "noise": asdict(noise_config),
    }
    print(json.dumps(metadata, ensure_ascii=False))

    with run_path.open("w") as output:
        output.write(json.dumps({"metadata": metadata}) + "\n")
        output.flush()
        for epoch in range(1, epochs + 1):
            epoch_started_at = time.perf_counter()
            if args.output_noise_read_compensation:
                output_noise_read_counts = apply_output_noise_read_compensation(
                    model,
                    base_repeats=args.output_noise_read_base,
                    max_repeats=args.output_noise_read_max,
                    exponent=args.output_noise_read_exponent,
                    kinds=tuple(args.output_noise_read_kinds),
                )
            set_read_approximation(
                model,
                args.train_read_approximation,
                kinds=tuple(args.train_read_approximation_kinds),
            )
            set_compute_mode(model, args.mode)
            train_started_at = time.perf_counter()
            train_metrics = run_epoch(
                model,
                train_loader,
                criterion,
                device,
                optimizer,
                max_train_batches,
                args.grad_clip_norm,
                args.stop_on_nonfinite,
                train_mc_samples=args.train_mc_samples,
                mc_consistency_weight=args.mc_consistency_weight,
                mc_consistency_temperature=args.mc_consistency_temperature,
                teacher_model=teacher_model,
                distill_alpha=args.distill_alpha,
                distill_temperature=args.distill_temperature,
                snr_rms_floor=args.snr_rms_floor,
                snr_regularization=args.snr_regularization,
                snr_kinds=args.snr_kinds,
                activation_scale_regularization=args.activation_scale_regularization,
                freeze_bn=args.freeze_bn,
            )
            train_seconds = time.perf_counter() - train_started_at
            if args.output_noise_read_compensation:
                output_noise_read_counts = apply_output_noise_read_compensation(
                    model,
                    base_repeats=args.output_noise_read_base,
                    max_repeats=args.output_noise_read_max,
                    exponent=args.output_noise_read_exponent,
                    kinds=tuple(args.output_noise_read_kinds),
                )
            set_read_approximation(
                model,
                "exact",
                kinds=tuple(args.train_read_approximation_kinds),
            )
            stopped_early = bool(
                args.stop_on_nonfinite and train_metrics.get("nonfinite")
            )
            eval_seconds = 0.0
            if stopped_early:
                eval_metrics = {
                    "loss": None,
                    "accuracy": None,
                    "examples": 0,
                    "skipped": True,
                    "skip_reason": train_metrics.get("stop_reason"),
                }
            elif epoch == 1 or epoch == epochs or epoch % args.eval_every == 0:
                set_compute_mode(model, eval_mode)
                eval_started_at = time.perf_counter()
                eval_metrics = run_epoch(
                    model,
                    eval_loader,
                    criterion,
                    device,
                    max_batches=max_eval_batches,
                    stop_on_nonfinite=args.stop_on_nonfinite,
                )
                eval_seconds = time.perf_counter() - eval_started_at
                stopped_early = bool(
                    args.stop_on_nonfinite and eval_metrics.get("nonfinite")
                )
            else:
                eval_metrics = {
                    "loss": None,
                    "accuracy": None,
                    "examples": 0,
                    "skipped": True,
                    "skip_reason": "eval_interval",
                }
            metrics = {
                "epoch": epoch,
                "train": train_metrics,
                "eval": eval_metrics,
                "stopped_early": stopped_early,
                "learning_rate": optimizer.param_groups[0]["lr"],
                "train_seconds": train_seconds,
                "eval_seconds": eval_seconds,
                "epoch_seconds": time.perf_counter() - epoch_started_at,
                "activation_scale_summary": activation_scale_summary(model),
                "output_noise_read_counts": output_noise_read_counts,
            }
            if activation_scale_parameters:
                metrics["activation_scale_learning_rate"] = optimizer.param_groups[-1][
                    "lr"
                ]
            eval_accuracy = eval_metrics.get("accuracy")
            if (
                isinstance(eval_accuracy, (float, int))
                and math.isfinite(eval_accuracy)
                and eval_accuracy > best_eval_accuracy
            ):
                best_eval_accuracy = eval_accuracy
                best_epoch = epoch
                metrics["best_so_far"] = True
                torch.save(model.state_dict(), best_checkpoint_path)
            if stopped_early:
                metrics["stop_reason"] = train_metrics.get(
                    "stop_reason", eval_metrics.get("stop_reason")
                )
            print(json.dumps(metrics))
            output.write(json.dumps(metrics) + "\n")
            output.flush()
            if stopped_early:
                break
            if scheduler is not None:
                scheduler.step()

    torch.save(model.state_dict(), checkpoint_path)
    print(f"metrics: {run_path}")
    print(f"checkpoint: {checkpoint_path}")
    if best_epoch is not None:
        print(f"best_checkpoint: {best_checkpoint_path} (epoch {best_epoch})")


if __name__ == "__main__":
    main()
