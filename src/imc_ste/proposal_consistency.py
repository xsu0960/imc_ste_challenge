from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from .task_consistency import normalized_smooth_l1, soft_target_kl


class ProposalAlignedROIConsistency:
    """Distill a noisy ROI head on the clean teacher's sampled proposals.

    The primary detection path remains unchanged and uses noisy RPN proposals.
    This auxiliary path reuses noisy backbone features but evaluates them at the
    clean training proposals, making teacher/student ROI rows exactly aligned.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        objective: str = "teacher",
        temperature: float = 2.0,
        box_weight: float = 0.25,
        eps: float = 1e-6,
    ) -> None:
        if objective not in {"teacher", "target", "foreground_target"}:
            raise ValueError(
                "proposal consistency objective must be teacher, target, "
                "or foreground_target"
            )
        if temperature <= 0:
            raise ValueError("proposal consistency temperature must be positive")
        if box_weight < 0:
            raise ValueError("proposal consistency box_weight must be non-negative")
        if eps <= 0:
            raise ValueError("proposal consistency eps must be positive")
        if not hasattr(model, "backbone") or not hasattr(model, "roi_heads"):
            raise ValueError("proposal consistency requires a Faster R-CNN style model")
        roi_heads = model.roi_heads
        for name in ("box_roi_pool", "box_head", "box_predictor"):
            if not hasattr(roi_heads, name):
                raise ValueError(f"proposal consistency requires roi_heads.{name}")
        if objective in {"target", "foreground_target"}:
            for name in ("assign_targets_to_proposals", "box_coder"):
                if not hasattr(roi_heads, name):
                    raise ValueError(
                        f"target-aligned ROI supervision requires roi_heads.{name}"
                    )

        self.model = model
        self.objective = objective
        self.temperature = temperature
        self.box_weight = box_weight
        self.eps = eps
        self.phase = "disabled"
        self.clean_proposals: list[torch.Tensor] | None = None
        self.clean_image_shapes: list[tuple[int, int]] | None = None
        self.clean_logits: torch.Tensor | None = None
        self.clean_box_regression: torch.Tensor | None = None
        self.clean_targets: list[dict[str, torch.Tensor]] | None = None
        self.noisy_features: Mapping[str, torch.Tensor] | torch.Tensor | None = None
        self.handles = [
            model.backbone.register_forward_hook(self._backbone_hook),
            roi_heads.register_forward_pre_hook(self._roi_heads_pre_hook),
            roi_heads.box_roi_pool.register_forward_pre_hook(self._roi_pool_pre_hook),
            roi_heads.box_predictor.register_forward_hook(self._box_predictor_hook),
        ]

    def _backbone_hook(self, _module, _inputs, output) -> None:
        if self.phase == "noisy":
            self.noisy_features = output

    def _roi_heads_pre_hook(self, _module, inputs) -> None:
        if self.phase != "clean" or len(inputs) < 4 or inputs[3] is None:
            return
        self.clean_targets = [
            {
                key: value.detach().clone()
                for key, value in target.items()
                if key in {"boxes", "labels"}
            }
            for target in inputs[3]
        ]

    def _roi_pool_pre_hook(self, _module, inputs) -> None:
        if self.phase != "clean" or len(inputs) < 3:
            return
        proposals = inputs[1]
        image_shapes = inputs[2]
        self.clean_proposals = [proposal.detach().clone() for proposal in proposals]
        self.clean_image_shapes = [tuple(shape) for shape in image_shapes]

    def _box_predictor_hook(self, _module, _inputs, output) -> None:
        if self.phase != "clean":
            return
        if not isinstance(output, tuple) or len(output) != 2:
            raise RuntimeError("ROI box predictor must return logits and box regression")
        self.clean_logits = output[0].detach().clone()
        self.clean_box_regression = output[1].detach().clone()

    def begin_clean(self) -> None:
        self.phase = "clean"
        self.clean_proposals = None
        self.clean_image_shapes = None
        self.clean_logits = None
        self.clean_box_regression = None
        self.clean_targets = None
        self.noisy_features = None

    def begin_noisy(self) -> None:
        if self.phase != "clean":
            raise RuntimeError("proposal consistency noisy phase requires a clean phase")
        self.phase = "noisy"

    def loss(self) -> tuple[torch.Tensor, dict[str, float | int]]:
        if self.phase != "noisy":
            raise RuntimeError("proposal consistency loss requires a noisy phase")
        if self.clean_proposals is None or self.clean_image_shapes is None:
            raise RuntimeError("clean ROI proposals were not captured")
        if self.objective == "teacher" and (
            self.clean_logits is None or self.clean_box_regression is None
        ):
            raise RuntimeError("clean ROI predictions were not captured")
        if self.objective in {"target", "foreground_target"} and (
            self.clean_targets is None
        ):
            raise RuntimeError("clean ROI targets were not captured")
        if self.noisy_features is None:
            raise RuntimeError("noisy backbone features were not captured")

        roi_heads = self.model.roi_heads
        self.phase = "auxiliary"
        try:
            box_features = roi_heads.box_roi_pool(
                self.noisy_features,
                self.clean_proposals,
                self.clean_image_shapes,
            )
            box_features = roi_heads.box_head(box_features)
            noisy_logits, noisy_box_regression = roi_heads.box_predictor(box_features)
        finally:
            self.phase = "noisy"

        if self.objective == "teacher":
            if noisy_logits.shape != self.clean_logits.shape:
                raise RuntimeError(
                    "proposal-aligned ROI logits changed shape: "
                    f"{tuple(noisy_logits.shape)} vs {tuple(self.clean_logits.shape)}"
                )
            if noisy_box_regression.shape != self.clean_box_regression.shape:
                raise RuntimeError(
                    "proposal-aligned ROI regression changed shape: "
                    f"{tuple(noisy_box_regression.shape)} vs "
                    f"{tuple(self.clean_box_regression.shape)}"
                )
            classification = soft_target_kl(
                noisy_logits,
                self.clean_logits,
                temperature=self.temperature,
            )
            regression = normalized_smooth_l1(
                noisy_box_regression,
                self.clean_box_regression,
                eps=self.eps,
            )
            foreground = 0
        else:
            labels, regression_targets = self._target_assignments()
            from torchvision.models.detection.roi_heads import fastrcnn_loss

            foreground = sum(int((label > 0).sum().item()) for label in labels)
            if self.objective == "foreground_target":
                flat_labels = torch.cat(labels, dim=0)
                flat_regression_targets = torch.cat(regression_targets, dim=0)
                foreground_mask = flat_labels > 0
                if bool(foreground_mask.any().item()):
                    classification, regression = fastrcnn_loss(
                        noisy_logits[foreground_mask],
                        noisy_box_regression[foreground_mask],
                        [flat_labels[foreground_mask]],
                        [flat_regression_targets[foreground_mask]],
                    )
                else:
                    classification = noisy_logits.sum() * 0.0
                    regression = noisy_box_regression.sum() * 0.0
            else:
                classification, regression = fastrcnn_loss(
                    noisy_logits,
                    noisy_box_regression,
                    labels,
                    regression_targets,
                )
        total = classification + self.box_weight * regression
        return total, {
            "classification": float(classification.detach().cpu()),
            "regression": float(regression.detach().cpu()),
            "proposals": sum(proposal.shape[0] for proposal in self.clean_proposals),
            "foreground": foreground,
        }

    def _target_assignments(self) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        if self.clean_proposals is None or self.clean_targets is None:
            raise RuntimeError("target-aligned ROI supervision has no clean targets")
        roi_heads = self.model.roi_heads
        gt_boxes = [target["boxes"] for target in self.clean_targets]
        gt_labels = [target["labels"] for target in self.clean_targets]
        matched_indices, labels = roi_heads.assign_targets_to_proposals(
            self.clean_proposals,
            gt_boxes,
            gt_labels,
        )
        matched_gt_boxes = []
        for proposals, boxes, matched in zip(
            self.clean_proposals, gt_boxes, matched_indices
        ):
            if boxes.numel() == 0:
                matched_gt_boxes.append(
                    torch.zeros_like(proposals, memory_format=torch.contiguous_format)
                )
            else:
                matched_gt_boxes.append(boxes[matched.clamp(min=0)])
        regression_targets = roi_heads.box_coder.encode(
            matched_gt_boxes,
            self.clean_proposals,
        )
        return labels, regression_targets

    def disable(self) -> None:
        self.phase = "disabled"
        self.clean_proposals = None
        self.clean_image_shapes = None
        self.clean_logits = None
        self.clean_box_regression = None
        self.clean_targets = None
        self.noisy_features = None

    def close(self) -> None:
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
