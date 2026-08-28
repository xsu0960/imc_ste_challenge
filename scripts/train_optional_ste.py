#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_ROOT = PROJECT_ROOT / "data"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from imc_ste import (  # noqa: E402
    COMPUTE_MODES,
    READ_APPROXIMATION_MODES,
    WEIGHT_NOISE_SCOPES,
    NoiseConfig,
    NoisyConv2d,
    NoisyLinear,
    OnlineGradientProfile,
    ProposalAlignedROIConsistency,
    TaskOutputConsistency,
    apply_activation_range_scaling,
    apply_gradient_noise_statistics,
    apply_named_layer_read_repeats,
    apply_output_noise_read_compensation,
    convert_model,
    scale_noise_config,
    set_compute_mode,
    set_conv_chunk_rows,
    set_conv_weight_noise_scope,
    set_read_approximation,
)


VOC_TO_COCO_LABEL = {
    "aeroplane": 5,
    "bicycle": 2,
    "bird": 16,
    "boat": 9,
    "bottle": 44,
    "bus": 6,
    "car": 3,
    "cat": 17,
    "chair": 62,
    "cow": 21,
    "diningtable": 67,
    "dog": 18,
    "horse": 19,
    "motorbike": 4,
    "person": 1,
    "pottedplant": 64,
    "sheep": 20,
    "sofa": 63,
    "train": 7,
    "tvmonitor": 72,
}

ROLE_PREFIXES = {
    "detection": {
        "backbone": ("backbone.body.",),
        "fpn": ("backbone.fpn.",),
        "rpn": ("rpn.",),
        "roi": ("roi_heads.",),
    },
    "segmentation": {
        "backbone": ("backbone.",),
        "classifier": ("classifier.",),
        "aux_classifier": ("aux_classifier.",),
    },
}
ALL_LAYER_KINDS = ("conv", "depthwise", "pointwise", "linear")
ALL_OPTIONAL_ROLES = tuple(
    sorted({role for task_roles in ROLE_PREFIXES.values() for role in task_roles})
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VOC detection/segmentation training with internal noisy STE layers."
    )
    parser.add_argument("--task", choices=["detection", "segmentation"], required=True)
    parser.add_argument("--mode", choices=COMPUTE_MODES, default="sat_aware_ste")
    parser.add_argument(
        "--eval-mode",
        choices=COMPUTE_MODES,
        help="Forward mode during validation. Defaults to noise for non-clean training.",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument(
        "--eval-only",
        action="store_true",
        help="Skip training and run a single validation pass.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--stop-on-nonfinite", action="store_true")
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--noise-scale", type=float, default=1.0)
    parser.add_argument(
        "--conv-chunk-rows",
        type=int,
        default=8,
        help="Rows per noisy convolution chunk; use 0 to disable chunking.",
    )
    parser.add_argument(
        "--conv-weight-noise-scope",
        choices=WEIGHT_NOISE_SCOPES,
        default="read",
        help="Share weight/supply noise per physical read; chunk reproduces the legacy behavior.",
    )
    parser.add_argument(
        "--activation-stat-csv",
        type=Path,
        help="Role-aware layer statistics used for ideal-operation-preserving activation preconditioning.",
    )
    parser.add_argument("--activation-target", type=float, default=4.0)
    parser.add_argument("--activation-scale-floor", type=float, default=0.1)
    parser.add_argument(
        "--activation-stat-kinds",
        nargs="+",
        choices=ALL_LAYER_KINDS,
        default=list(ALL_LAYER_KINDS),
    )
    parser.add_argument(
        "--activation-stat-roles",
        nargs="+",
        choices=ALL_OPTIONAL_ROLES,
        help="Subsystems to precondition. Defaults to every subsystem in the selected task.",
    )
    parser.add_argument(
        "--gradient-stat-csv",
        type=Path,
        help="Measured stochastic/systematic layer statistics for variance-aware STE.",
    )
    parser.add_argument("--gradient-stat-strength", type=float, default=1.0)
    parser.add_argument("--gradient-bias-strength", type=float, default=0.0)
    parser.add_argument("--gradient-stat-floor", type=float, default=0.25)
    parser.add_argument(
        "--gradient-stat-kinds",
        nargs="+",
        choices=ALL_LAYER_KINDS,
        default=list(ALL_LAYER_KINDS),
    )
    parser.add_argument(
        "--gradient-stat-roles",
        nargs="+",
        choices=ALL_OPTIONAL_ROLES,
        help="Subsystems receiving measured variance-aware gradient profiles.",
    )
    parser.add_argument(
        "--online-gradient-profile",
        action="store_true",
        help="Estimate per-channel bias/variance online from paired clean/noisy passes.",
    )
    parser.add_argument(
        "--online-gradient-roles",
        nargs="+",
        choices=ALL_OPTIONAL_ROLES,
        help="Subsystems receiving online gradient profiles; defaults to RPN or segmentation heads.",
    )
    parser.add_argument(
        "--online-gradient-name-prefixes",
        nargs="+",
        help="Explicit noisy-layer prefixes; overrides --online-gradient-roles.",
    )
    parser.add_argument("--online-gradient-ema-decay", type=float, default=0.95)
    parser.add_argument("--online-gradient-variance-strength", type=float, default=0.5)
    parser.add_argument("--online-gradient-bias-strength", type=float, default=0.25)
    parser.add_argument("--online-gradient-scale-floor", type=float, default=0.5)
    parser.add_argument("--online-gradient-scale-ceiling", type=float, default=1.0)
    parser.add_argument("--online-gradient-warmup-updates", type=int, default=16)
    parser.add_argument("--online-gradient-eps", type=float, default=1e-6)
    parser.add_argument(
        "--online-gradient-stats-output",
        type=Path,
        help="Final per-layer online profile CSV; defaults beside the metrics JSONL.",
    )
    parser.add_argument("--read-repeats", type=int, default=1)
    parser.add_argument("--backbone-read-repeats", type=int)
    parser.add_argument("--fpn-read-repeats", type=int)
    parser.add_argument("--rpn-read-repeats", type=int)
    parser.add_argument("--roi-read-repeats", type=int)
    parser.add_argument("--classifier-read-repeats", type=int)
    parser.add_argument("--aux-classifier-read-repeats", type=int)
    parser.add_argument(
        "--output-noise-read-compensation",
        action="store_true",
        help="Allocate reads as a bounded inverse power of each activation scale.",
    )
    parser.add_argument("--output-noise-read-base", type=int, default=1)
    parser.add_argument("--output-noise-read-max", type=int, default=8)
    parser.add_argument("--output-noise-read-exponent", type=float, default=2.0)
    parser.add_argument(
        "--output-noise-read-roles",
        nargs="+",
        choices=ALL_OPTIONAL_ROLES,
        help="Subsystems eligible for activation-scale-aware read allocation.",
    )
    parser.add_argument(
        "--train-read-approximation",
        choices=READ_APPROXIMATION_MODES,
        default="exact",
        help="Training-only approximation; validation always uses exact physical reads.",
    )
    parser.add_argument(
        "--clean-feature-consistency-weight",
        type=float,
        default=0.0,
        help="Relative-MSE weight between clean-teacher and noisy internal features.",
    )
    parser.add_argument(
        "--clean-feature-consistency-roles",
        nargs="+",
        choices=ALL_OPTIONAL_ROLES,
        help="Subsystems used for dual-pass feature consistency; defaults to FPN or segmentation classifier.",
    )
    parser.add_argument("--clean-feature-consistency-eps", type=float, default=1e-6)
    parser.add_argument(
        "--task-output-consistency-weight",
        type=float,
        default=0.0,
        help="Clean-teacher consistency on aligned RPN outputs or segmentation logits.",
    )
    parser.add_argument("--task-output-consistency-temperature", type=float, default=2.0)
    parser.add_argument("--task-output-consistency-box-weight", type=float, default=0.25)
    parser.add_argument("--task-output-consistency-aux-weight", type=float, default=0.4)
    parser.add_argument("--task-output-consistency-eps", type=float, default=1e-6)
    parser.add_argument(
        "--proposal-roi-consistency",
        action="store_true",
        help="Execute the proposal-aligned ROI auxiliary path even when its weight is zero.",
    )
    parser.add_argument(
        "--proposal-roi-consistency-weight",
        type=float,
        default=0.0,
        help="Detection-only auxiliary loss on exactly aligned ROI proposals.",
    )
    parser.add_argument(
        "--proposal-roi-objective",
        choices=("teacher", "target", "foreground_target"),
        default="teacher",
        help=(
            "Use clean predictions, all ground-truth targets, or foreground-only "
            "targets on clean sampled proposals."
        ),
    )
    parser.add_argument("--proposal-roi-consistency-temperature", type=float, default=2.0)
    parser.add_argument("--proposal-roi-consistency-box-weight", type=float, default=0.25)
    parser.add_argument("--proposal-roi-consistency-eps", type=float, default=1e-6)
    parser.add_argument("--detection-min-size", type=int, default=320)
    parser.add_argument("--detection-max-size", type=int, default=512)
    parser.add_argument("--detection-score-threshold", type=float, default=0.05)
    parser.add_argument("--trainable-backbone-layers", type=int, default=3)
    parser.add_argument("--segmentation-image-size", type=int, default=256)
    parser.add_argument("--aux-loss-weight", type=float, default=0.4)
    parser.add_argument(
        "--freeze-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep BatchNorm layers in eval mode during training; useful for small-batch VOC fine-tuning.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSONL metrics path. Defaults to runs/<timestamp>_<task>_<mode>_seed<seed>.jsonl.",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        help="CSV summary path. Defaults to metrics path with _summary.csv suffix.",
    )
    parser.add_argument(
        "--save-checkpoint",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save final/best model states; disable for evaluation-only sweeps.",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def limit_or_none(value: int | None) -> int | None:
    if value is None or value <= 0:
        return None
    return value


def prefixes_for_roles(
    task: str, selected_roles: list[str] | None
) -> tuple[str, ...]:
    role_map = ROLE_PREFIXES[task]
    roles = list(role_map) if selected_roles is None else selected_roles
    invalid_roles = set(roles) - set(role_map)
    if invalid_roles:
        raise ValueError(
            f"roles {sorted(invalid_roles)} do not belong to {task}; "
            f"choose from {sorted(role_map)}"
        )
    return tuple(prefix for role in roles for prefix in role_map[role])


def configured_role_read_repeats(args: argparse.Namespace) -> dict[str, int]:
    values = {
        "backbone": args.backbone_read_repeats,
        "fpn": args.fpn_read_repeats,
        "rpn": args.rpn_read_repeats,
        "roi": args.roi_read_repeats,
        "classifier": args.classifier_read_repeats,
        "aux_classifier": args.aux_classifier_read_repeats,
    }
    role_map = ROLE_PREFIXES[args.task]
    return {
        prefix: int(values[role])
        for role, prefixes in role_map.items()
        if values[role] is not None
        for prefix in prefixes
    }


def read_approximation_prefixes(
    args: argparse.Namespace, prefix_read_repeats: dict[str, int]
) -> tuple[str, ...] | None:
    if args.read_repeats > 1:
        return None
    return tuple(
        prefix for prefix, repeats in prefix_read_repeats.items() if repeats > 1
    )


class CleanFeatureConsistency:
    """Capture matched clean/noisy layer outputs without changing noisy forward."""

    def __init__(self, model: nn.Module, name_prefixes: tuple[str, ...], eps: float):
        if eps <= 0:
            raise ValueError("clean feature consistency eps must be positive")
        self.eps = eps
        self.phase = "disabled"
        self.call_indices: dict[str, int] = {}
        self.clean_outputs: dict[str, list[torch.Tensor]] = {}
        self.noisy_outputs: dict[str, list[torch.Tensor]] = {}
        self.handles = []
        self.layer_names = []
        for name, module in model.named_modules():
            if not isinstance(module, (NoisyConv2d, NoisyLinear)):
                continue
            if not any(name.startswith(prefix) for prefix in name_prefixes):
                continue
            self.layer_names.append(name)
            self.handles.append(module.register_forward_hook(self._make_hook(name)))

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if self.phase == "disabled" or not isinstance(output, torch.Tensor):
                return
            call_index = self.call_indices.get(name, 0)
            self.call_indices[name] = call_index + 1
            if self.phase == "clean":
                self.clean_outputs.setdefault(name, []).append(output.detach().clone())
            else:
                self.noisy_outputs.setdefault(name, []).append(output.clone())

        return hook

    def begin_clean(self) -> None:
        self.phase = "clean"
        self.call_indices.clear()
        self.clean_outputs.clear()
        self.noisy_outputs.clear()

    def begin_noisy(self) -> None:
        self.phase = "noisy"
        self.call_indices.clear()

    def loss(self) -> tuple[torch.Tensor | None, int]:
        terms = []
        for name, noisy_values in self.noisy_outputs.items():
            clean_values = self.clean_outputs.get(name, [])
            for clean_value, noisy_value in zip(clean_values, noisy_values):
                if clean_value.shape != noisy_value.shape:
                    continue
                signal_mse = clean_value.square().mean()
                residual_mse = (noisy_value - clean_value).square().mean()
                terms.append(residual_mse / (signal_mse + self.eps))
        if not terms:
            return None, 0
        return torch.stack(terms).mean(), len(terms)

    def disable(self) -> None:
        self.phase = "disabled"
        self.call_indices.clear()
        self.clean_outputs.clear()
        self.noisy_outputs.clear()

    def close(self) -> None:
        self.disable()
        for handle in self.handles:
            handle.remove()


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    torch.manual_seed(worker_seed + worker_id)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary: {path}")


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def detection_collate(batch):
    return tuple(zip(*batch))


def parse_voc_detection_target(
    target: dict[str, Any], image_id: int
) -> dict[str, torch.Tensor]:
    annotation = target["annotation"]
    objects = annotation.get("object", [])
    if isinstance(objects, dict):
        objects = [objects]
    boxes = []
    labels = []
    areas = []
    iscrowd = []
    for obj in objects:
        label = VOC_TO_COCO_LABEL.get(obj.get("name"))
        if label is None:
            continue
        bbox = obj["bndbox"]
        xmin = float(bbox["xmin"]) - 1.0
        ymin = float(bbox["ymin"]) - 1.0
        xmax = float(bbox["xmax"]) - 1.0
        ymax = float(bbox["ymax"]) - 1.0
        if xmax <= xmin or ymax <= ymin:
            continue
        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(label)
        areas.append((xmax - xmin) * (ymax - ymin))
        iscrowd.append(1 if int(obj.get("difficult", 0)) else 0)

    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
        "labels": torch.tensor(labels, dtype=torch.int64),
        "image_id": torch.tensor([image_id], dtype=torch.int64),
        "area": torch.tensor(areas, dtype=torch.float32),
        "iscrowd": torch.tensor(iscrowd, dtype=torch.int64),
    }


class VOCDetectionTensor(torch.utils.data.Dataset):
    def __init__(self, root: Path, image_set: str, download: bool):
        self.dataset = torchvision.datasets.VOCDetection(
            root=root,
            year="2007",
            image_set=image_set,
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return F.to_tensor(image), parse_voc_detection_target(target, index)


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    if boxes1.numel() == 0 or boxes2.numel() == 0:
        return torch.zeros((boxes1.shape[0], boxes2.shape[0]), dtype=torch.float32)
    area1 = (boxes1[:, 2] - boxes1[:, 0]).clamp(min=0) * (
        boxes1[:, 3] - boxes1[:, 1]
    ).clamp(min=0)
    area2 = (boxes2[:, 2] - boxes2[:, 0]).clamp(min=0) * (
        boxes2[:, 3] - boxes2[:, 1]
    ).clamp(min=0)
    lt = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]
    union = area1[:, None] + area2 - inter
    return inter / union.clamp(min=1e-8)


def voc_map50(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    score_threshold: float,
) -> float:
    gt_by_class: dict[int, dict[int, torch.Tensor]] = {}
    total_gt: dict[int, int] = {}
    detections: dict[int, list[tuple[int, float, torch.Tensor]]] = {}

    for image_id, target in enumerate(targets):
        for label in target["labels"].unique().tolist():
            mask = target["labels"] == label
            gt_by_class.setdefault(int(label), {})[image_id] = target["boxes"][mask]
            total_gt[int(label)] = total_gt.get(int(label), 0) + int(mask.sum())

    for image_id, prediction in enumerate(predictions):
        keep = prediction["scores"] >= score_threshold
        boxes = prediction["boxes"][keep].detach().cpu()
        labels = prediction["labels"][keep].detach().cpu()
        scores = prediction["scores"][keep].detach().cpu()
        for box, label, score in zip(boxes, labels, scores):
            detections.setdefault(int(label), []).append((image_id, float(score), box))

    aps = []
    for label, num_gt in total_gt.items():
        matched: dict[int, set[int]] = {}
        class_detections = sorted(
            detections.get(label, []), key=lambda item: item[1], reverse=True
        )
        tp = []
        fp = []
        for image_id, _, box in class_detections:
            gt_boxes = gt_by_class[label].get(image_id, torch.empty((0, 4)))
            if gt_boxes.numel() == 0:
                tp.append(0.0)
                fp.append(1.0)
                continue
            ious = box_iou(box.view(1, 4), gt_boxes).view(-1)
            best_iou, best_index = torch.max(ious, dim=0)
            best_index_int = int(best_index.item())
            image_matches = matched.setdefault(image_id, set())
            if best_iou.item() >= 0.5 and best_index_int not in image_matches:
                tp.append(1.0)
                fp.append(0.0)
                image_matches.add(best_index_int)
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not tp:
            aps.append(0.0)
            continue
        tp_cum = torch.tensor(tp).cumsum(0)
        fp_cum = torch.tensor(fp).cumsum(0)
        recall = tp_cum / max(num_gt, 1)
        precision = tp_cum / (tp_cum + fp_cum).clamp(min=1e-8)
        recall = torch.cat([torch.tensor([0.0]), recall, torch.tensor([1.0])])
        precision = torch.cat([torch.tensor([0.0]), precision, torch.tensor([0.0])])
        for index in range(precision.numel() - 1, 0, -1):
            precision[index - 1] = torch.maximum(precision[index - 1], precision[index])
        changes = torch.where(recall[1:] != recall[:-1])[0]
        aps.append(
            float(
                torch.sum(
                    (recall[changes + 1] - recall[changes])
                    * precision[changes + 1]
                ).item()
            )
        )
    return mean(aps) if aps else 0.0


class VOCSegmentationTensor(torch.utils.data.Dataset):
    def __init__(self, root: Path, image_set: str, image_size: int, download: bool):
        self.dataset = torchvision.datasets.VOCSegmentation(
            root=root,
            year="2012",
            image_set=image_set,
            download=download,
        )
        self.image_set = image_set
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]
        if self.image_set == "train" and random.random() < 0.5:
            image = F.hflip(image)
            mask = F.hflip(mask)
        image = F.resize(image, (self.image_size, self.image_size), interpolation=Image.BILINEAR)
        mask = F.resize(mask, (self.image_size, self.image_size), interpolation=Image.NEAREST)
        image_tensor = F.to_tensor(image)
        image_tensor = F.normalize(
            image_tensor,
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
        mask_tensor = F.pil_to_tensor(mask).squeeze(0).to(torch.int64)
        return image_tensor, mask_tensor


def segmentation_miou(predictions: list[torch.Tensor], targets: list[torch.Tensor]) -> float:
    intersections = torch.zeros(21, dtype=torch.float64)
    unions = torch.zeros(21, dtype=torch.float64)
    for pred, target in zip(predictions, targets):
        valid = target != 255
        pred = pred[valid]
        target = target[valid]
        for label in range(21):
            pred_mask = pred == label
            target_mask = target == label
            intersections[label] += torch.logical_and(pred_mask, target_mask).sum()
            unions[label] += torch.logical_or(pred_mask, target_mask).sum()
    valid_classes = unions > 0
    if not bool(valid_classes.any()):
        return 0.0
    return float((intersections[valid_classes] / unions[valid_classes]).mean().item())


def build_detection_model(args: argparse.Namespace, noise_config: NoiseConfig) -> nn.Module:
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights=weights,
        trainable_backbone_layers=args.trainable_backbone_layers,
        min_size=args.detection_min_size,
        max_size=args.detection_max_size,
    )
    return convert_model(model, noise_config, args.mode, inplace=True)


def build_segmentation_model(args: argparse.Namespace, noise_config: NoiseConfig) -> nn.Module:
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    return convert_model(model, noise_config, args.mode, inplace=True)


def move_detection_targets(
    targets: list[dict[str, torch.Tensor]], device: torch.device
) -> list[dict[str, torch.Tensor]]:
    return [{key: value.to(device) for key, value in target.items()} for target in targets]


def run_detection_epoch(
    model: nn.Module,
    loader,
    device: torch.device,
    *,
    optimizer=None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
    score_threshold: float = 0.05,
    stop_on_nonfinite: bool = False,
    train_compute_mode: str | None = None,
    feature_consistency: CleanFeatureConsistency | None = None,
    feature_consistency_weight: float = 0.0,
    task_consistency: TaskOutputConsistency | None = None,
    task_consistency_weight: float = 0.0,
    online_gradient_profile: OnlineGradientProfile | None = None,
    proposal_roi_consistency: ProposalAlignedROIConsistency | None = None,
    proposal_roi_consistency_weight: float = 0.0,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    examples = 0
    nonfinite_batches = 0
    total_feature_consistency = 0.0
    feature_consistency_terms = 0
    total_task_consistency = 0.0
    total_task_classification = 0.0
    total_task_regression = 0.0
    task_consistency_terms = 0
    total_proposal_consistency = 0.0
    total_proposal_classification = 0.0
    total_proposal_regression = 0.0
    proposal_consistency_terms = 0
    proposal_foreground_terms = 0
    context = torch.enable_grad() if training else torch.no_grad()
    predictions: list[dict[str, torch.Tensor]] = []
    targets_cpu: list[dict[str, torch.Tensor]] = []

    with context:
        for batch_index, (images, targets) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = [image.to(device) for image in images]
            targets_device = move_detection_targets(list(targets), device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                use_feature_consistency = (
                    feature_consistency is not None and feature_consistency_weight > 0
                )
                use_task_consistency = (
                    task_consistency is not None and task_consistency_weight > 0
                )
                use_online_profile = online_gradient_profile is not None
                use_proposal_consistency = proposal_roi_consistency is not None
                if (
                    use_feature_consistency
                    or use_task_consistency
                    or use_online_profile
                    or use_proposal_consistency
                ):
                    if train_compute_mode is None:
                        raise ValueError("train_compute_mode is required for consistency")
                    if use_feature_consistency:
                        feature_consistency.begin_clean()
                    if use_task_consistency:
                        task_consistency.begin_clean()
                    if use_online_profile:
                        online_gradient_profile.begin_clean()
                    if use_proposal_consistency:
                        proposal_roi_consistency.begin_clean()
                    set_compute_mode(model, "clean")
                    with torch.no_grad():
                        model(images, targets_device)
                    if use_feature_consistency:
                        feature_consistency.begin_noisy()
                    if use_task_consistency:
                        task_consistency.begin_noisy()
                    if use_online_profile:
                        online_gradient_profile.begin_noisy()
                    if use_proposal_consistency:
                        proposal_roi_consistency.begin_noisy()
                    set_compute_mode(model, train_compute_mode)
                loss_dict = model(images, targets_device)
                if online_gradient_profile is not None:
                    online_gradient_profile.finalize_batch()
                loss = sum(loss for loss in loss_dict.values())
                proposal_consistency_loss = None
                proposal_consistency_details: dict[str, float | int] = {
                    "classification": 0.0,
                    "regression": 0.0,
                    "proposals": 0,
                    "foreground": 0,
                }
                if use_proposal_consistency:
                    proposal_consistency_loss, proposal_consistency_details = (
                        proposal_roi_consistency.loss()
                    )
                    loss = (
                        loss
                        + proposal_roi_consistency_weight
                        * proposal_consistency_loss
                    )
                consistency_loss = None
                matched_consistency_terms = 0
                if feature_consistency is not None and feature_consistency_weight > 0:
                    consistency_loss, matched_consistency_terms = feature_consistency.loss()
                    if consistency_loss is not None:
                        loss = loss + feature_consistency_weight * consistency_loss
                task_consistency_loss = None
                task_consistency_details: dict[str, float | int] = {
                    "terms": 0,
                    "classification": 0.0,
                    "regression": 0.0,
                }
                if task_consistency is not None and task_consistency_weight > 0:
                    task_consistency_loss, task_consistency_details = (
                        task_consistency.loss()
                    )
                    if task_consistency_loss is not None:
                        loss = loss + task_consistency_weight * task_consistency_loss
                if not torch.isfinite(loss):
                    nonfinite_batches += 1
                    if stop_on_nonfinite:
                        break
                    continue
                loss.backward()
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    if not bool(torch.isfinite(grad_norm).item()):
                        nonfinite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        if stop_on_nonfinite:
                            break
                        continue
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * len(images)
                if consistency_loss is not None:
                    total_feature_consistency += (
                        float(consistency_loss.detach().cpu()) * len(images)
                    )
                    feature_consistency_terms += matched_consistency_terms
                if task_consistency_loss is not None:
                    total_task_consistency += (
                        float(task_consistency_loss.detach().cpu()) * len(images)
                    )
                    total_task_classification += (
                        float(task_consistency_details["classification"]) * len(images)
                    )
                    total_task_regression += (
                        float(task_consistency_details["regression"]) * len(images)
                    )
                    task_consistency_terms += int(task_consistency_details["terms"])
                if proposal_consistency_loss is not None:
                    total_proposal_consistency += (
                        float(proposal_consistency_loss.detach().cpu()) * len(images)
                    )
                    total_proposal_classification += (
                        float(proposal_consistency_details["classification"])
                        * len(images)
                    )
                    total_proposal_regression += (
                        float(proposal_consistency_details["regression"]) * len(images)
                    )
                    proposal_consistency_terms += int(
                        proposal_consistency_details["proposals"]
                    )
                    proposal_foreground_terms += int(
                        proposal_consistency_details["foreground"]
                    )
            else:
                outputs = model(images)
                predictions.extend(
                    [
                        {key: value.detach().cpu() for key, value in output.items()}
                        for output in outputs
                    ]
                )
                targets_cpu.extend(
                    [
                        {
                            "boxes": target["boxes"].detach().cpu(),
                            "labels": target["labels"].detach().cpu(),
                        }
                        for target in targets
                    ]
                )
            examples += len(images)

    metrics = {
        "loss": total_loss / max(examples, 1) if training else None,
        "examples": examples,
        "nonfinite": nonfinite_batches > 0,
        "nonfinite_batches": nonfinite_batches,
    }
    if training and feature_consistency_weight > 0:
        metrics["feature_consistency"] = total_feature_consistency / max(examples, 1)
        metrics["feature_consistency_terms"] = feature_consistency_terms
    if training and task_consistency_weight > 0:
        metrics["task_output_consistency"] = total_task_consistency / max(examples, 1)
        metrics["task_output_classification"] = total_task_classification / max(
            examples, 1
        )
        metrics["task_output_regression"] = total_task_regression / max(examples, 1)
        metrics["task_output_consistency_terms"] = task_consistency_terms
    if training and online_gradient_profile is not None:
        metrics["online_gradient_profile"] = online_gradient_profile.summary(
            reset_epoch_counters=True
        )
        online_gradient_profile.disable()
    if training and proposal_roi_consistency is not None:
        metrics["proposal_roi_consistency"] = total_proposal_consistency / max(
            examples, 1
        )
        metrics["proposal_roi_classification"] = (
            total_proposal_classification / max(examples, 1)
        )
        metrics["proposal_roi_regression"] = total_proposal_regression / max(
            examples, 1
        )
        metrics["proposal_roi_proposals"] = proposal_consistency_terms
        metrics["proposal_roi_foreground"] = proposal_foreground_terms
    if proposal_roi_consistency is not None:
        proposal_roi_consistency.disable()
    if feature_consistency is not None:
        feature_consistency.disable()
    if task_consistency is not None:
        task_consistency.disable()
    if not training:
        metrics["mAP50"] = voc_map50(predictions, targets_cpu, score_threshold)
    return metrics


def segmentation_collate(batch):
    images, masks = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(masks))


def freeze_batchnorm_stats(model: nn.Module) -> int:
    count = 0
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            count += 1
    return count


def run_segmentation_epoch(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    *,
    optimizer=None,
    max_batches: int | None = None,
    grad_clip_norm: float | None = None,
    aux_loss_weight: float = 0.4,
    freeze_bn: bool = True,
    stop_on_nonfinite: bool = False,
    train_compute_mode: str | None = None,
    feature_consistency: CleanFeatureConsistency | None = None,
    feature_consistency_weight: float = 0.0,
    task_consistency: TaskOutputConsistency | None = None,
    task_consistency_weight: float = 0.0,
    online_gradient_profile: OnlineGradientProfile | None = None,
) -> dict[str, Any]:
    training = optimizer is not None
    model.train(training)
    frozen_bn_count = freeze_batchnorm_stats(model) if training and freeze_bn else 0
    total_loss = 0.0
    examples = 0
    nonfinite_batches = 0
    total_feature_consistency = 0.0
    feature_consistency_terms = 0
    total_task_consistency = 0.0
    total_task_classification = 0.0
    total_task_regression = 0.0
    task_consistency_terms = 0
    predictions: list[torch.Tensor] = []
    targets_cpu: list[torch.Tensor] = []
    context = torch.enable_grad() if training else torch.no_grad()

    with context:
        for batch_index, (images, masks) in enumerate(loader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = images.to(device)
            masks = masks.to(device)
            if training:
                optimizer.zero_grad(set_to_none=True)
                use_feature_consistency = (
                    feature_consistency is not None and feature_consistency_weight > 0
                )
                use_task_consistency = (
                    task_consistency is not None and task_consistency_weight > 0
                )
                use_online_profile = online_gradient_profile is not None
                if use_feature_consistency or use_task_consistency or use_online_profile:
                    if train_compute_mode is None:
                        raise ValueError("train_compute_mode is required for consistency")
                    if use_feature_consistency:
                        feature_consistency.begin_clean()
                    if use_task_consistency:
                        task_consistency.begin_clean()
                    if use_online_profile:
                        online_gradient_profile.begin_clean()
                    set_compute_mode(model, "clean")
                    with torch.no_grad():
                        clean_outputs = model(images)
                    if use_task_consistency:
                        task_consistency.capture_segmentation_teacher(clean_outputs)
                    if use_feature_consistency:
                        feature_consistency.begin_noisy()
                    if use_task_consistency:
                        task_consistency.begin_noisy()
                    if use_online_profile:
                        online_gradient_profile.begin_noisy()
                    set_compute_mode(model, train_compute_mode)
            outputs = model(images)
            if training and online_gradient_profile is not None:
                online_gradient_profile.finalize_batch()
            loss = criterion(outputs["out"], masks)
            if "aux" in outputs:
                loss = loss + aux_loss_weight * criterion(outputs["aux"], masks)
            consistency_loss = None
            matched_consistency_terms = 0
            if training and feature_consistency is not None and feature_consistency_weight > 0:
                consistency_loss, matched_consistency_terms = feature_consistency.loss()
                if consistency_loss is not None:
                    loss = loss + feature_consistency_weight * consistency_loss
            task_consistency_loss = None
            task_consistency_details: dict[str, float | int] = {
                "terms": 0,
                "classification": 0.0,
                "regression": 0.0,
            }
            if training and task_consistency is not None and task_consistency_weight > 0:
                task_consistency_loss, task_consistency_details = task_consistency.loss(
                    outputs, masks
                )
                if task_consistency_loss is not None:
                    loss = loss + task_consistency_weight * task_consistency_loss
            if not torch.isfinite(loss):
                nonfinite_batches += 1
                if stop_on_nonfinite:
                    break
                continue
            if training:
                loss.backward()
                if grad_clip_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        grad_clip_norm,
                        error_if_nonfinite=False,
                    )
                    if not bool(torch.isfinite(grad_norm).item()):
                        nonfinite_batches += 1
                        optimizer.zero_grad(set_to_none=True)
                        if stop_on_nonfinite:
                            break
                        continue
                optimizer.step()
                if consistency_loss is not None:
                    total_feature_consistency += (
                        float(consistency_loss.detach().cpu()) * images.shape[0]
                    )
                    feature_consistency_terms += matched_consistency_terms
                if task_consistency_loss is not None:
                    total_task_consistency += (
                        float(task_consistency_loss.detach().cpu()) * images.shape[0]
                    )
                    total_task_classification += (
                        float(task_consistency_details["classification"])
                        * images.shape[0]
                    )
                    total_task_regression += (
                        float(task_consistency_details["regression"])
                        * images.shape[0]
                    )
                    task_consistency_terms += int(task_consistency_details["terms"])
            else:
                predictions.extend(outputs["out"].argmax(1).detach().cpu())
                targets_cpu.extend(masks.detach().cpu())
            total_loss += float(loss.detach().cpu()) * images.shape[0]
            examples += images.shape[0]

    metrics = {
        "loss": total_loss / max(examples, 1),
        "examples": examples,
        "nonfinite": nonfinite_batches > 0,
        "nonfinite_batches": nonfinite_batches,
        "frozen_bn_count": frozen_bn_count,
    }
    if training and feature_consistency_weight > 0:
        metrics["feature_consistency"] = total_feature_consistency / max(examples, 1)
        metrics["feature_consistency_terms"] = feature_consistency_terms
    if training and task_consistency_weight > 0:
        metrics["task_output_consistency"] = total_task_consistency / max(examples, 1)
        metrics["task_output_classification"] = total_task_classification / max(
            examples, 1
        )
        metrics["task_output_regression"] = total_task_regression / max(examples, 1)
        metrics["task_output_consistency_terms"] = task_consistency_terms
    if training and online_gradient_profile is not None:
        metrics["online_gradient_profile"] = online_gradient_profile.summary(
            reset_epoch_counters=True
        )
        online_gradient_profile.disable()
    if feature_consistency is not None:
        feature_consistency.disable()
    if task_consistency is not None:
        task_consistency.disable()
    if not training:
        metrics["mIoU"] = segmentation_miou(predictions, targets_cpu)
    return metrics


def build_loaders(args: argparse.Namespace):
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    if args.task == "detection":
        train_data = VOCDetectionTensor(DATA_ROOT, "train", args.download)
        eval_data = VOCDetectionTensor(DATA_ROOT, "val", args.download)
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            collate_fn=detection_collate,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        eval_loader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=detection_collate,
            worker_init_fn=seed_worker,
            generator=generator,
        )
        return train_loader, eval_loader

    train_data = VOCSegmentationTensor(
        DATA_ROOT, "train", args.segmentation_image_size, args.download
    )
    eval_data = VOCSegmentationTensor(
        DATA_ROOT, "val", args.segmentation_image_size, args.download
    )
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=segmentation_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    eval_loader = torch.utils.data.DataLoader(
        eval_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=segmentation_collate,
        worker_init_fn=seed_worker,
        generator=generator,
    )
    return train_loader, eval_loader


def summarize_rows(metrics_rows: list[dict[str, Any]], metadata: dict[str, Any]) -> list[dict[str, Any]]:
    eval_key = "mAP50" if metadata["task"] == "detection" else "mIoU"
    eval_values = [
        row["eval"][eval_key]
        for row in metrics_rows
        if isinstance(row.get("eval", {}).get(eval_key), (float, int))
    ]
    best_value = max(eval_values) if eval_values else None
    final_value = eval_values[-1] if eval_values else None
    return [
        {
            "task": metadata["task"],
            "dataset": metadata["dataset"],
            "model": metadata["model"],
            "train_mode": metadata["train_mode"],
            "eval_mode": metadata["eval_mode"],
            "metric": eval_key,
            "epochs": metadata["epochs"],
            "runs": 1,
            "final": final_value,
            "best": best_value,
            "mean": mean(eval_values) if eval_values else None,
            "std": stdev(eval_values) if len(eval_values) > 1 else 0.0,
            "ci95": ci95(eval_values),
            "notes": metadata["notes"],
        }
    ]


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = select_device(args.device)
    noise_config = scale_noise_config(NoiseConfig(), args.noise_scale)
    eval_mode = args.eval_mode or ("clean" if args.mode == "clean" else "noise")
    max_train_batches = limit_or_none(args.max_train_batches)
    max_eval_batches = limit_or_none(args.max_eval_batches)

    if args.task == "detection":
        model = build_detection_model(args, noise_config)
        model_name = "fasterrcnn_resnet50_fpn"
    else:
        model = build_segmentation_model(args, noise_config)
        model_name = "deeplabv3_resnet50"
    chunked_conv_count = set_conv_chunk_rows(model, args.conv_chunk_rows)
    weight_noise_scope_count = set_conv_weight_noise_scope(
        model, args.conv_weight_noise_scope
    )
    activation_name_prefixes = prefixes_for_roles(
        args.task, args.activation_stat_roles
    )
    gradient_name_prefixes = prefixes_for_roles(args.task, args.gradient_stat_roles)
    if args.activation_stat_csv is not None:
        activation_stat_counts = apply_activation_range_scaling(
            model,
            args.activation_stat_csv,
            target_abs=args.activation_target,
            scale_floor=args.activation_scale_floor,
            kinds=tuple(args.activation_stat_kinds),
            name_prefixes=activation_name_prefixes,
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
    if args.gradient_stat_csv is not None:
        gradient_stat_counts = apply_gradient_noise_statistics(
            model,
            args.gradient_stat_csv,
            variance_strength=args.gradient_stat_strength,
            bias_strength=args.gradient_bias_strength,
            scale_floor=args.gradient_stat_floor,
            kinds=tuple(args.gradient_stat_kinds),
            name_prefixes=gradient_name_prefixes,
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
    prefix_read_repeats = configured_role_read_repeats(args)
    read_repeat_summary = apply_named_layer_read_repeats(
        model,
        default_read_repeats=args.read_repeats,
        prefix_read_repeats=prefix_read_repeats,
    )
    output_noise_read_prefixes = prefixes_for_roles(
        args.task, args.output_noise_read_roles
    )
    if args.output_noise_read_compensation:
        output_noise_read_summary = apply_output_noise_read_compensation(
            model,
            base_repeats=args.output_noise_read_base,
            max_repeats=args.output_noise_read_max,
            exponent=args.output_noise_read_exponent,
            kinds=tuple(args.activation_stat_kinds),
            name_prefixes=output_noise_read_prefixes,
        )
    else:
        output_noise_read_summary = {
            "conv": 0,
            "depthwise": 0,
            "pointwise": 0,
            "linear": 0,
            "layers": 0,
            "read_histogram": {},
            "mean_read_repeats": 0.0,
            "total_read_repeats": 0,
        }
    approximation_name_prefixes = (
        output_noise_read_prefixes
        if args.output_noise_read_compensation
        else read_approximation_prefixes(args, prefix_read_repeats)
    )
    set_read_approximation(model, "exact", kinds=ALL_LAYER_KINDS)
    model.to(device)
    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    if args.online_gradient_profile:
        if "variance" not in args.mode:
            raise ValueError(
                "--online-gradient-profile requires a variance-aware STE mode"
            )
        online_gradient_roles = args.online_gradient_roles
        if args.online_gradient_name_prefixes:
            online_gradient_prefixes = tuple(args.online_gradient_name_prefixes)
        else:
            if online_gradient_roles is None:
                online_gradient_roles = (
                    ["rpn"]
                    if args.task == "detection"
                    else ["classifier", "aux_classifier"]
                )
            online_gradient_prefixes = prefixes_for_roles(
                args.task, online_gradient_roles
            )
        online_gradient_profile = OnlineGradientProfile(
            model,
            online_gradient_prefixes,
            ema_decay=args.online_gradient_ema_decay,
            variance_strength=args.online_gradient_variance_strength,
            bias_strength=args.online_gradient_bias_strength,
            scale_floor=args.online_gradient_scale_floor,
            scale_ceiling=args.online_gradient_scale_ceiling,
            warmup_updates=args.online_gradient_warmup_updates,
            eps=args.online_gradient_eps,
        )
    else:
        online_gradient_roles = args.online_gradient_roles
        online_gradient_prefixes = ()
        online_gradient_profile = None

    train_loader, eval_loader = build_loaders(args)
    if args.clean_feature_consistency_weight < 0:
        raise ValueError("--clean-feature-consistency-weight must be non-negative")
    if args.clean_feature_consistency_weight > 0:
        consistency_roles = args.clean_feature_consistency_roles
        if consistency_roles is None:
            consistency_roles = ["fpn"] if args.task == "detection" else ["classifier"]
        consistency_prefixes = prefixes_for_roles(args.task, consistency_roles)
        feature_consistency = CleanFeatureConsistency(
            model, consistency_prefixes, args.clean_feature_consistency_eps
        )
    else:
        consistency_roles = args.clean_feature_consistency_roles
        consistency_prefixes = ()
        feature_consistency = None
    if args.task_output_consistency_weight < 0:
        raise ValueError("--task-output-consistency-weight must be non-negative")
    if args.task_output_consistency_weight > 0:
        task_consistency = TaskOutputConsistency(
            model,
            args.task,
            temperature=args.task_output_consistency_temperature,
            box_weight=args.task_output_consistency_box_weight,
            aux_weight=args.task_output_consistency_aux_weight,
            eps=args.task_output_consistency_eps,
        )
    else:
        task_consistency = None
    if args.proposal_roi_consistency_weight < 0:
        raise ValueError("--proposal-roi-consistency-weight must be non-negative")
    if args.proposal_roi_consistency or args.proposal_roi_consistency_weight > 0:
        if args.task != "detection":
            raise ValueError("proposal ROI consistency is detection-only")
        proposal_roi_consistency = ProposalAlignedROIConsistency(
            model,
            objective=args.proposal_roi_objective,
            temperature=args.proposal_roi_consistency_temperature,
            box_weight=args.proposal_roi_consistency_box_weight,
            eps=args.proposal_roi_consistency_eps,
        )
    else:
        proposal_roi_consistency = None
    optimizer = torch.optim.SGD(
        [param for param in model.parameters() if param.requires_grad],
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    RUNS_DIR.mkdir(exist_ok=True)
    output_path = args.output or (
        RUNS_DIR / f"{int(time.time())}_{args.task}_{args.mode}_seed{args.seed}.jsonl"
    )
    summary_path = args.summary_output or output_path.with_name(
        f"{output_path.stem}_summary.csv"
    )
    checkpoint_path = output_path.with_suffix(".pt")
    best_checkpoint_path = output_path.with_name(f"{output_path.stem}_best.pt")
    online_gradient_stats_path = args.online_gradient_stats_output or output_path.with_name(
        f"{output_path.stem}_online_profile.csv"
    )

    metadata = {
        "task": args.task,
        "dataset": "VOC2007" if args.task == "detection" else "VOC2012",
        "model": model_name,
        "seed": args.seed,
        "train_mode": args.mode,
        "eval_mode": eval_mode,
        "device": str(device),
        "epochs": args.epochs,
        "eval_only": args.eval_only,
        "save_checkpoint": args.save_checkpoint,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "grad_clip_norm": args.grad_clip_norm,
        "max_train_batches": max_train_batches,
        "max_eval_batches": max_eval_batches,
        "noise_scale": args.noise_scale,
        "conv_chunk_rows": args.conv_chunk_rows,
        "chunked_conv_count": chunked_conv_count,
        "conv_weight_noise_scope": args.conv_weight_noise_scope,
        "weight_noise_scope_count": weight_noise_scope_count,
        "activation_stat_csv": str(args.activation_stat_csv)
        if args.activation_stat_csv
        else "",
        "activation_target": args.activation_target,
        "activation_scale_floor": args.activation_scale_floor,
        "activation_stat_kinds": args.activation_stat_kinds,
        "activation_stat_roles": args.activation_stat_roles,
        "activation_stat_counts": activation_stat_counts,
        "gradient_stat_csv": str(args.gradient_stat_csv)
        if args.gradient_stat_csv
        else "",
        "gradient_stat_strength": args.gradient_stat_strength,
        "gradient_bias_strength": args.gradient_bias_strength,
        "gradient_stat_floor": args.gradient_stat_floor,
        "gradient_stat_kinds": args.gradient_stat_kinds,
        "gradient_stat_roles": args.gradient_stat_roles,
        "gradient_stat_counts": gradient_stat_counts,
        "online_gradient_profile": args.online_gradient_profile,
        "online_gradient_roles": online_gradient_roles,
        "online_gradient_name_prefixes": args.online_gradient_name_prefixes,
        "online_gradient_layers": online_gradient_profile.layer_names
        if online_gradient_profile
        else [],
        "online_gradient_ema_decay": args.online_gradient_ema_decay,
        "online_gradient_variance_strength": args.online_gradient_variance_strength,
        "online_gradient_bias_strength": args.online_gradient_bias_strength,
        "online_gradient_scale_floor": args.online_gradient_scale_floor,
        "online_gradient_scale_ceiling": args.online_gradient_scale_ceiling,
        "online_gradient_warmup_updates": args.online_gradient_warmup_updates,
        "online_gradient_eps": args.online_gradient_eps,
        "online_gradient_stats_output": str(online_gradient_stats_path)
        if args.online_gradient_profile
        else "",
        "read_repeats": args.read_repeats,
        "prefix_read_repeats": prefix_read_repeats,
        "read_repeat_summary": read_repeat_summary,
        "output_noise_read_compensation": args.output_noise_read_compensation,
        "output_noise_read_base": args.output_noise_read_base,
        "output_noise_read_max": args.output_noise_read_max,
        "output_noise_read_exponent": args.output_noise_read_exponent,
        "output_noise_read_roles": args.output_noise_read_roles,
        "output_noise_read_summary": output_noise_read_summary,
        "train_read_approximation": args.train_read_approximation,
        "clean_feature_consistency_weight": args.clean_feature_consistency_weight,
        "clean_feature_consistency_roles": consistency_roles,
        "clean_feature_consistency_layers": len(feature_consistency.layer_names)
        if feature_consistency
        else 0,
        "clean_feature_consistency_eps": args.clean_feature_consistency_eps,
        "task_output_consistency_weight": args.task_output_consistency_weight,
        "task_output_consistency_temperature": args.task_output_consistency_temperature,
        "task_output_consistency_box_weight": args.task_output_consistency_box_weight,
        "task_output_consistency_aux_weight": args.task_output_consistency_aux_weight,
        "task_output_consistency_eps": args.task_output_consistency_eps,
        "task_output_consistency_layers": task_consistency.layer_names
        if task_consistency
        else [],
        "proposal_roi_consistency_weight": args.proposal_roi_consistency_weight,
        "proposal_roi_consistency": args.proposal_roi_consistency,
        "proposal_roi_objective": args.proposal_roi_objective,
        "proposal_roi_consistency_temperature": args.proposal_roi_consistency_temperature,
        "proposal_roi_consistency_box_weight": args.proposal_roi_consistency_box_weight,
        "proposal_roi_consistency_eps": args.proposal_roi_consistency_eps,
        "freeze_bn": args.freeze_bn,
        "noise": asdict(noise_config),
        "notes": "Internal Conv/Linear replacement with noisy forward and STE backward.",
    }
    print(json.dumps({"metadata": metadata}, ensure_ascii=False))

    metrics_rows: list[dict[str, Any]] = []
    best_eval = -math.inf
    with output_path.open("w") as handle:
        handle.write(json.dumps({"metadata": metadata}) + "\n")
        handle.flush()
        epoch_range = [0] if args.eval_only else range(1, args.epochs + 1)
        for epoch in epoch_range:
            if not args.eval_only:
                set_read_approximation(
                    model,
                    args.train_read_approximation,
                    kinds=ALL_LAYER_KINDS,
                    name_prefixes=approximation_name_prefixes,
                )
            set_compute_mode(model, args.mode)
            if args.task == "detection" and not args.eval_only:
                train_metrics = run_detection_epoch(
                    model,
                    train_loader,
                    device,
                    optimizer=optimizer,
                    max_batches=max_train_batches,
                    grad_clip_norm=args.grad_clip_norm,
                    score_threshold=args.detection_score_threshold,
                    stop_on_nonfinite=args.stop_on_nonfinite,
                    train_compute_mode=args.mode,
                    feature_consistency=feature_consistency,
                    feature_consistency_weight=args.clean_feature_consistency_weight,
                    task_consistency=task_consistency,
                    task_consistency_weight=args.task_output_consistency_weight,
                    online_gradient_profile=online_gradient_profile,
                    proposal_roi_consistency=proposal_roi_consistency,
                    proposal_roi_consistency_weight=args.proposal_roi_consistency_weight,
                )
            elif args.task == "detection":
                train_metrics = {
                    "loss": None,
                    "examples": 0,
                    "skipped": True,
                    "skip_reason": "eval_only",
                }

            if args.task == "detection":
                set_read_approximation(model, "exact", kinds=ALL_LAYER_KINDS)
                set_compute_mode(model, eval_mode)
                eval_metrics = run_detection_epoch(
                    model,
                    eval_loader,
                    device,
                    max_batches=max_eval_batches,
                    score_threshold=args.detection_score_threshold,
                    stop_on_nonfinite=args.stop_on_nonfinite,
                )
                eval_value = eval_metrics.get("mAP50")

            elif not args.eval_only:
                train_metrics = run_segmentation_epoch(
                    model,
                    train_loader,
                    criterion,
                    device,
                    optimizer=optimizer,
                    max_batches=max_train_batches,
                    grad_clip_norm=args.grad_clip_norm,
                    aux_loss_weight=args.aux_loss_weight,
                    freeze_bn=args.freeze_bn,
                    stop_on_nonfinite=args.stop_on_nonfinite,
                    train_compute_mode=args.mode,
                    feature_consistency=feature_consistency,
                    feature_consistency_weight=args.clean_feature_consistency_weight,
                    task_consistency=task_consistency,
                    task_consistency_weight=args.task_output_consistency_weight,
                    online_gradient_profile=online_gradient_profile,
                )
                set_read_approximation(model, "exact", kinds=ALL_LAYER_KINDS)
                set_compute_mode(model, eval_mode)
                eval_metrics = run_segmentation_epoch(
                    model,
                    eval_loader,
                    criterion,
                    device,
                    max_batches=max_eval_batches,
                    aux_loss_weight=args.aux_loss_weight,
                    stop_on_nonfinite=args.stop_on_nonfinite,
                )
                eval_value = eval_metrics.get("mIoU")
            else:
                train_metrics = {
                    "loss": None,
                    "examples": 0,
                    "skipped": True,
                    "skip_reason": "eval_only",
                }
                set_read_approximation(model, "exact", kinds=ALL_LAYER_KINDS)
                set_compute_mode(model, eval_mode)
                eval_metrics = run_segmentation_epoch(
                    model,
                    eval_loader,
                    criterion,
                    device,
                    max_batches=max_eval_batches,
                    aux_loss_weight=args.aux_loss_weight,
                    stop_on_nonfinite=args.stop_on_nonfinite,
                )
                eval_value = eval_metrics.get("mIoU")

            row = {
                "epoch": epoch,
                "train": train_metrics,
                "eval": eval_metrics,
                "learning_rate": optimizer.param_groups[0]["lr"],
            }
            if isinstance(eval_value, (float, int)) and math.isfinite(eval_value):
                if eval_value > best_eval:
                    best_eval = float(eval_value)
                    row["best_so_far"] = True
                    if args.save_checkpoint:
                        torch.save(model.state_dict(), best_checkpoint_path)
            print(json.dumps(row))
            handle.write(json.dumps(row) + "\n")
            handle.flush()
            metrics_rows.append(row)
            if args.stop_on_nonfinite and train_metrics.get("nonfinite"):
                break

    if feature_consistency is not None:
        feature_consistency.close()
    if task_consistency is not None:
        task_consistency.close()
    if online_gradient_profile is not None:
        write_csv(online_gradient_stats_path, online_gradient_profile.rows())
        online_gradient_profile.close()
    if proposal_roi_consistency is not None:
        proposal_roi_consistency.close()
    if args.save_checkpoint:
        torch.save(model.state_dict(), checkpoint_path)
    write_csv(summary_path, summarize_rows(metrics_rows, metadata))
    print(f"metrics: {output_path}")
    if args.save_checkpoint:
        print(f"checkpoint: {checkpoint_path}")
        print(f"best_checkpoint: {best_checkpoint_path}")


if __name__ == "__main__":
    main()
