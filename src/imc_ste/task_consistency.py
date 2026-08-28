from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


DETECTION_CLASSIFICATION_LAYERS = ("rpn.head.cls_logits",)
DETECTION_REGRESSION_LAYERS = ("rpn.head.bbox_pred",)


def soft_target_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 2.0,
    class_dim: int = 1,
    valid_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """KL(teacher || student) with detached soft targets."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher logits must have matching shapes")
    teacher_scaled = teacher.detach() / temperature
    student_scaled = student / temperature
    teacher_probability = torch.softmax(teacher_scaled, dim=class_dim)
    teacher_log_probability = torch.log_softmax(teacher_scaled, dim=class_dim)
    student_log_probability = torch.log_softmax(student_scaled, dim=class_dim)
    loss = (
        teacher_probability * (teacher_log_probability - student_log_probability)
    ).sum(dim=class_dim)
    if valid_mask is not None:
        if valid_mask.shape != loss.shape:
            raise ValueError("valid mask must match logits after removing class dimension")
        loss = loss[valid_mask]
    if loss.numel() == 0:
        return student.sum() * 0.0
    return loss.mean() * temperature**2


def bernoulli_soft_target_kl(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    temperature: float = 2.0,
) -> torch.Tensor:
    """Bernoulli KL for aligned objectness logits."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher logits must have matching shapes")
    teacher_scaled = teacher.detach() / temperature
    student_scaled = student / temperature
    probability = torch.sigmoid(teacher_scaled)
    positive = probability * (
        F.logsigmoid(teacher_scaled) - F.logsigmoid(student_scaled)
    )
    negative = (1.0 - probability) * (
        F.logsigmoid(-teacher_scaled) - F.logsigmoid(-student_scaled)
    )
    return (positive + negative).mean() * temperature**2


def normalized_smooth_l1(
    student: torch.Tensor,
    teacher: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Scale-normalized regression consistency for aligned box deltas."""

    if eps <= 0:
        raise ValueError("eps must be positive")
    if student.shape != teacher.shape:
        raise ValueError("student and teacher tensors must have matching shapes")
    scale = teacher.detach().square().mean().sqrt().clamp_min(eps)
    return F.smooth_l1_loss(student / scale, teacher.detach() / scale)


class TaskOutputConsistency:
    """Task-level clean-teacher consistency for optional domain models.

    Detection uses RPN outputs because they are spatially aligned across clean
    and noisy passes. ROI rows depend on stochastic proposal sampling and are
    intentionally excluded from direct elementwise distillation.
    """

    def __init__(
        self,
        model: nn.Module,
        task: str,
        *,
        temperature: float = 2.0,
        box_weight: float = 0.25,
        aux_weight: float = 0.4,
        eps: float = 1e-6,
    ):
        if task not in ("detection", "segmentation"):
            raise ValueError(f"unsupported task consistency target: {task}")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if box_weight < 0 or aux_weight < 0:
            raise ValueError("task consistency weights must be non-negative")
        if eps <= 0:
            raise ValueError("task consistency eps must be positive")
        self.task = task
        self.temperature = temperature
        self.box_weight = box_weight
        self.aux_weight = aux_weight
        self.eps = eps
        self.phase = "disabled"
        self.clean_outputs: dict[str, list[torch.Tensor]] = {}
        self.noisy_outputs: dict[str, list[torch.Tensor]] = {}
        self.segmentation_teacher: dict[str, torch.Tensor] = {}
        self.handles: list[Any] = []
        self.layer_names: list[str] = []

        if task == "detection":
            selected = set(
                DETECTION_CLASSIFICATION_LAYERS + DETECTION_REGRESSION_LAYERS
            )
            for name, module in model.named_modules():
                if name not in selected:
                    continue
                self.layer_names.append(name)
                self.handles.append(module.register_forward_hook(self._make_hook(name)))
            missing = selected - set(self.layer_names)
            if missing:
                self.close()
                raise ValueError(
                    f"missing detection task-consistency layers: {sorted(missing)}"
                )
        else:
            self.layer_names = ["out", "aux"]

    def _make_hook(self, name: str):
        def hook(_module, _inputs, output):
            if self.phase == "disabled" or not isinstance(output, torch.Tensor):
                return
            if self.phase == "clean":
                self.clean_outputs.setdefault(name, []).append(
                    output.detach().clone()
                )
            else:
                self.noisy_outputs.setdefault(name, []).append(output)

        return hook

    def begin_clean(self) -> None:
        self.phase = "clean"
        self.clean_outputs.clear()
        self.noisy_outputs.clear()
        self.segmentation_teacher.clear()

    def capture_segmentation_teacher(self, outputs: dict[str, torch.Tensor]) -> None:
        if self.task != "segmentation" or self.phase != "clean":
            raise RuntimeError("segmentation teacher capture requires a clean phase")
        self.segmentation_teacher = {
            key: value.detach().clone()
            for key, value in outputs.items()
            if key in ("out", "aux") and isinstance(value, torch.Tensor)
        }
        if "out" not in self.segmentation_teacher:
            raise ValueError("segmentation model did not return out logits")

    def begin_noisy(self) -> None:
        self.phase = "noisy"
        self.noisy_outputs.clear()

    def _detection_loss(self) -> tuple[torch.Tensor | None, dict[str, float | int]]:
        classification_terms = []
        regression_terms = []
        matched_calls = 0
        for name in DETECTION_CLASSIFICATION_LAYERS:
            clean_values = self.clean_outputs.get(name, [])
            noisy_values = self.noisy_outputs.get(name, [])
            for clean_value, noisy_value in zip(clean_values, noisy_values):
                if clean_value.shape != noisy_value.shape:
                    continue
                classification_terms.append(
                    bernoulli_soft_target_kl(
                        noisy_value,
                        clean_value,
                        temperature=self.temperature,
                    )
                )
                matched_calls += 1
        for name in DETECTION_REGRESSION_LAYERS:
            clean_values = self.clean_outputs.get(name, [])
            noisy_values = self.noisy_outputs.get(name, [])
            for clean_value, noisy_value in zip(clean_values, noisy_values):
                if clean_value.shape != noisy_value.shape:
                    continue
                regression_terms.append(
                    normalized_smooth_l1(noisy_value, clean_value, eps=self.eps)
                )
                matched_calls += 1
        if not classification_terms and not regression_terms:
            return None, {"terms": 0, "classification": 0.0, "regression": 0.0}
        reference = (
            classification_terms[0]
            if classification_terms
            else regression_terms[0]
        )
        classification = (
            torch.stack(classification_terms).mean()
            if classification_terms
            else reference.new_zeros(())
        )
        regression = (
            torch.stack(regression_terms).mean()
            if regression_terms
            else reference.new_zeros(())
        )
        total = classification + self.box_weight * regression
        return total, {
            "terms": matched_calls,
            "classification": float(classification.detach().cpu()),
            "regression": float(regression.detach().cpu()),
        }

    def _segmentation_loss(
        self,
        outputs: dict[str, torch.Tensor],
        masks: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, dict[str, float | int]]:
        terms = []
        main_value = 0.0
        aux_value = 0.0
        for key, weight in (("out", 1.0), ("aux", self.aux_weight)):
            if weight == 0 or key not in outputs or key not in self.segmentation_teacher:
                continue
            student = outputs[key]
            teacher = self.segmentation_teacher[key]
            if student.shape != teacher.shape:
                continue
            valid_mask = None
            if masks is not None:
                valid_mask = masks != 255
                if valid_mask.shape != student.shape[:1] + student.shape[2:]:
                    valid_mask = F.interpolate(
                        valid_mask[:, None].to(student.dtype),
                        size=student.shape[-2:],
                        mode="nearest",
                    )[:, 0].to(torch.bool)
            value = soft_target_kl(
                student,
                teacher,
                temperature=self.temperature,
                class_dim=1,
                valid_mask=valid_mask,
            )
            terms.append(weight * value)
            if key == "out":
                main_value = float(value.detach().cpu())
            else:
                aux_value = float(value.detach().cpu())
        if not terms:
            return None, {"terms": 0, "classification": 0.0, "regression": 0.0}
        return torch.stack(terms).sum(), {
            "terms": len(terms),
            "classification": main_value,
            "regression": aux_value,
        }

    def loss(
        self,
        outputs: dict[str, torch.Tensor] | None = None,
        masks: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, float | int]]:
        if self.phase != "noisy":
            raise RuntimeError("task consistency loss requires a noisy phase")
        if self.task == "detection":
            return self._detection_loss()
        if outputs is None:
            raise ValueError("segmentation consistency requires model outputs")
        return self._segmentation_loss(outputs, masks)

    def disable(self) -> None:
        self.phase = "disabled"
        self.clean_outputs.clear()
        self.noisy_outputs.clear()
        self.segmentation_teacher.clear()

    def close(self) -> None:
        self.disable()
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
