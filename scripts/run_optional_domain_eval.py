#!/usr/bin/env python3
import argparse
import csv
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import torch
import torchvision
import torchvision.transforms.functional as F
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = PROJECT_ROOT / "runs"
DATA_ROOT = PROJECT_ROOT / "data"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optional detection/segmentation pilot evaluation."
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["detection", "segmentation"],
        default=["detection", "segmentation"],
    )
    parser.add_argument("--max-samples", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--box-noise-std", type=float, default=2.0)
    parser.add_argument("--score-noise-std", type=float, default=0.02)
    parser.add_argument("--logit-noise-std", type=float, default=0.25)
    parser.add_argument("--quantization-bits", type=int, default=8)
    parser.add_argument(
        "--output",
        type=Path,
        default=RUNS_DIR / "optional_domain_eval.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=RUNS_DIR / "optional_domain_eval_summary.csv",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {path}")


def ci95(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return 1.96 * stdev(values) / math.sqrt(len(values))


def limit_dataset(dataset, max_samples: int):
    if max_samples <= 0 or max_samples >= len(dataset):
        return dataset
    return torch.utils.data.Subset(dataset, list(range(max_samples)))


def detection_collate(batch):
    return tuple(zip(*batch))


def parse_voc_detection_target(target: dict[str, Any]) -> dict[str, torch.Tensor]:
    annotation = target["annotation"]
    objects = annotation.get("object", [])
    if isinstance(objects, dict):
        objects = [objects]
    boxes = []
    labels = []
    for obj in objects:
        name = obj.get("name")
        label = VOC_TO_COCO_LABEL.get(name)
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
    return {
        "boxes": torch.tensor(boxes, dtype=torch.float32),
        "labels": torch.tensor(labels, dtype=torch.int64),
    }


class VOCDetectionTensor(torch.utils.data.Dataset):
    def __init__(self, root: Path, download: bool):
        self.dataset = torchvision.datasets.VOCDetection(
            root=root,
            year="2007",
            image_set="val",
            download=download,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, target = self.dataset[index]
        return F.to_tensor(image), parse_voc_detection_target(target)


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


def apply_detection_noise(
    prediction: dict[str, torch.Tensor],
    *,
    box_noise_std: float,
    score_noise_std: float,
    quantization_bits: int,
    seed: int,
) -> dict[str, torch.Tensor]:
    generator = torch.Generator(device=prediction["boxes"].device)
    generator.manual_seed(seed)
    boxes = prediction["boxes"].clone()
    scores = prediction["scores"].clone()
    if box_noise_std > 0:
        boxes = boxes + torch.randn(
            boxes.shape, generator=generator, device=boxes.device
        ) * box_noise_std
    if score_noise_std > 0:
        scores = scores + torch.randn(
            scores.shape, generator=generator, device=scores.device
        ) * score_noise_std
    scores = scores.clamp(0.0, 1.0)
    if quantization_bits > 0:
        levels = 2**quantization_bits - 1
        scores = torch.round(scores * levels) / levels
    return {
        "boxes": boxes,
        "labels": prediction["labels"].clone(),
        "scores": scores,
    }


def voc_map50(
    predictions: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    score_threshold: float,
) -> tuple[float, dict[int, float]]:
    gt_by_class: dict[int, dict[int, torch.Tensor]] = defaultdict(dict)
    total_gt: dict[int, int] = defaultdict(int)
    detections: dict[int, list[tuple[int, float, torch.Tensor]]] = defaultdict(list)

    for image_id, target in enumerate(targets):
        for label in target["labels"].unique().tolist():
            mask = target["labels"] == label
            boxes = target["boxes"][mask]
            gt_by_class[int(label)][image_id] = boxes
            total_gt[int(label)] += int(boxes.shape[0])

    for image_id, prediction in enumerate(predictions):
        keep = prediction["scores"] >= score_threshold
        boxes = prediction["boxes"][keep].detach().cpu()
        labels = prediction["labels"][keep].detach().cpu()
        scores = prediction["scores"][keep].detach().cpu()
        for box, label, score in zip(boxes, labels, scores):
            detections[int(label)].append((image_id, float(score), box))

    ap_by_class: dict[int, float] = {}
    for label, num_gt in total_gt.items():
        if num_gt == 0:
            continue
        matched: dict[int, set[int]] = defaultdict(set)
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
            if best_iou.item() >= 0.5 and best_index_int not in matched[image_id]:
                tp.append(1.0)
                fp.append(0.0)
                matched[image_id].add(best_index_int)
            else:
                tp.append(0.0)
                fp.append(1.0)
        if not tp:
            ap_by_class[label] = 0.0
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
        ap = torch.sum((recall[changes + 1] - recall[changes]) * precision[changes + 1])
        ap_by_class[label] = float(ap.item())

    valid_aps = list(ap_by_class.values())
    return (mean(valid_aps) if valid_aps else 0.0), ap_by_class


def run_detection(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device).eval()
    dataset = limit_dataset(VOCDetectionTensor(DATA_ROOT, args.download), args.max_samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=detection_collate,
    )

    predictions_clean: list[dict[str, torch.Tensor]] = []
    predictions_noisy: list[dict[str, torch.Tensor]] = []
    targets: list[dict[str, torch.Tensor]] = []

    sample_index = 0
    with torch.no_grad():
        for images, batch_targets in loader:
            images = [image.to(device) for image in images]
            outputs = model(images)
            for output, target in zip(outputs, batch_targets):
                clean = {key: value.detach().cpu() for key, value in output.items()}
                noisy = apply_detection_noise(
                    {key: value.detach() for key, value in output.items()},
                    box_noise_std=args.box_noise_std,
                    score_noise_std=args.score_noise_std,
                    quantization_bits=args.quantization_bits,
                    seed=args.seed + sample_index,
                )
                predictions_clean.append(clean)
                predictions_noisy.append(
                    {key: value.detach().cpu() for key, value in noisy.items()}
                )
                targets.append(target)
                sample_index += 1

    clean_map, _ = voc_map50(predictions_clean, targets, args.score_threshold)
    noisy_map, _ = voc_map50(predictions_noisy, targets, args.score_threshold)
    return [
        {
            "task": "detection",
            "dataset": "VOC2007-val",
            "model": "fasterrcnn_resnet50_fpn_coco",
            "protocol": "clean",
            "metric": "mAP50",
            "value": clean_map,
            "samples": len(targets),
            "notes": "COCO-pretrained Faster R-CNN evaluated on VOC labels mapped to COCO ids.",
        },
        {
            "task": "detection",
            "dataset": "VOC2007-val",
            "model": "fasterrcnn_resnet50_fpn_coco",
            "protocol": "output_noise",
            "metric": "mAP50",
            "value": noisy_map,
            "samples": len(targets),
            "notes": json.dumps(
                {
                    "box_noise_std": args.box_noise_std,
                    "score_noise_std": args.score_noise_std,
                    "quantization_bits": args.quantization_bits,
                }
            ),
        },
    ]


class VOCSegmentationTensor(torch.utils.data.Dataset):
    def __init__(self, root: Path, download: bool):
        self.dataset = torchvision.datasets.VOCSegmentation(
            root=root,
            year="2012",
            image_set="val",
            download=download,
        )
        self.transform = torchvision.transforms.Compose(
            [
                torchvision.transforms.Resize((256, 256), interpolation=Image.BILINEAR),
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
                ),
            ]
        )
        self.target_transform = torchvision.transforms.Resize(
            (256, 256), interpolation=Image.NEAREST
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, mask = self.dataset[index]
        image_tensor = self.transform(image)
        mask_tensor = (
            F.pil_to_tensor(self.target_transform(mask)).squeeze(0).to(torch.int64)
        )
        return image_tensor, mask_tensor


def segmentation_collate(batch):
    images, masks = zip(*batch)
    return torch.stack(list(images)), torch.stack(list(masks))


def quantize_logits(logits: torch.Tensor, bits: int) -> torch.Tensor:
    if bits <= 0:
        return logits
    max_abs = logits.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    normalized = (logits / max_abs).clamp(-1.0, 1.0)
    levels = 2**bits - 1
    quantized = torch.round((normalized + 1.0) * 0.5 * levels) / levels
    return (quantized * 2.0 - 1.0) * max_abs


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


def run_segmentation(args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    weights = torchvision.models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
    model = torchvision.models.segmentation.deeplabv3_resnet50(weights=weights)
    model.to(device).eval()
    dataset = limit_dataset(VOCSegmentationTensor(DATA_ROOT, args.download), args.max_samples)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=segmentation_collate,
    )

    clean_predictions: list[torch.Tensor] = []
    noisy_predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device)
            logits = model(images)["out"]
            noisy_logits = logits
            if args.logit_noise_std > 0:
                noisy_logits = noisy_logits + torch.randn(
                    noisy_logits.shape, generator=generator, device=device
                ) * args.logit_noise_std
            noisy_logits = quantize_logits(noisy_logits, args.quantization_bits)
            clean_predictions.extend(logits.argmax(1).detach().cpu())
            noisy_predictions.extend(noisy_logits.argmax(1).detach().cpu())
            targets.extend(masks.detach().cpu())

    clean_miou = segmentation_miou(clean_predictions, targets)
    noisy_miou = segmentation_miou(noisy_predictions, targets)
    return [
        {
            "task": "segmentation",
            "dataset": "VOC2012-val",
            "model": "deeplabv3_resnet50_coco_voc",
            "protocol": "clean",
            "metric": "mIoU",
            "value": clean_miou,
            "samples": len(targets),
            "notes": "COCO/VOC-pretrained DeepLabV3 evaluated on VOC val.",
        },
        {
            "task": "segmentation",
            "dataset": "VOC2012-val",
            "model": "deeplabv3_resnet50_coco_voc",
            "protocol": "output_noise",
            "metric": "mIoU",
            "value": noisy_miou,
            "samples": len(targets),
            "notes": json.dumps(
                {
                    "logit_noise_std": args.logit_noise_std,
                    "quantization_bits": args.quantization_bits,
                }
            ),
        },
    ]


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    meta: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    for row in rows:
        key = (row["task"], row["dataset"], row["model"], row["protocol"])
        grouped[key].append(float(row["value"]))
        meta[key] = row
    output = []
    for key, values in sorted(grouped.items()):
        row = meta[key]
        output.append(
            {
                "task": row["task"],
                "dataset": row["dataset"],
                "model": row["model"],
                "protocol": row["protocol"],
                "metric": row["metric"],
                "runs": len(values),
                "mean": mean(values),
                "std": stdev(values) if len(values) > 1 else 0.0,
                "ci95": ci95(values),
                "samples": row["samples"],
                "notes": row["notes"],
            }
        )
    return output


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = select_device(args.device)
    rows: list[dict[str, Any]] = []
    if "detection" in args.tasks:
        rows.extend(run_detection(args, device))
    if "segmentation" in args.tasks:
        rows.extend(run_segmentation(args, device))
    write_csv(args.output, rows)
    write_csv(args.summary_output, summarize(rows))


if __name__ == "__main__":
    main()
