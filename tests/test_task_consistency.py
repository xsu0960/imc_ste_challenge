import pytest
import torch
import torch.nn as nn

from imc_ste import (
    TaskOutputConsistency,
    bernoulli_soft_target_kl,
    normalized_smooth_l1,
    soft_target_kl,
)


class FakeRPNHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.cls_logits = nn.Identity()
        self.bbox_pred = nn.Identity()

    def forward(self, classification, regression):
        return self.cls_logits(classification), self.bbox_pred(regression)


class FakeRPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = FakeRPNHead()


class FakeDetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.rpn = FakeRPN()


def test_soft_target_losses_are_zero_for_identical_values():
    logits = torch.randn(2, 4, 3, 3)
    boxes = torch.randn(2, 8, 3, 3)

    assert float(soft_target_kl(logits, logits).abs()) < 1e-6
    assert float(bernoulli_soft_target_kl(logits, logits).abs()) < 1e-6
    assert float(normalized_smooth_l1(boxes, boxes)) == 0.0


def test_detection_task_consistency_tracks_aligned_rpn_outputs():
    model = FakeDetectionModel()
    consistency = TaskOutputConsistency(model, "detection", temperature=1.5)
    clean_classification = torch.randn(1, 3, 4, 4)
    clean_regression = torch.randn(1, 12, 4, 4)
    noisy_classification = (clean_classification + 0.3).requires_grad_()
    noisy_regression = (clean_regression - 0.2).requires_grad_()

    consistency.begin_clean()
    model.rpn.head(clean_classification, clean_regression)
    consistency.begin_noisy()
    model.rpn.head(noisy_classification, noisy_regression)
    loss, details = consistency.loss()

    assert loss is not None and float(loss) > 0
    assert details["terms"] == 2
    loss.backward()
    assert noisy_classification.grad is not None
    assert noisy_regression.grad is not None
    consistency.close()


def test_segmentation_task_consistency_masks_ignore_pixels():
    model = nn.Identity()
    consistency = TaskOutputConsistency(model, "segmentation", temperature=2.0)
    clean = {
        "out": torch.randn(1, 3, 4, 4),
        "aux": torch.randn(1, 3, 4, 4),
    }
    noisy = {
        key: (value + 0.2 * torch.randn_like(value)).requires_grad_()
        for key, value in clean.items()
    }
    masks = torch.zeros(1, 4, 4, dtype=torch.long)
    masks[:, 0, 0] = 255

    consistency.begin_clean()
    consistency.capture_segmentation_teacher(clean)
    consistency.begin_noisy()
    loss, details = consistency.loss(noisy, masks)

    assert loss is not None and float(loss) > 0
    assert details["terms"] == 2
    loss.backward()
    assert noisy["out"].grad is not None
    assert noisy["aux"].grad is not None


def test_task_consistency_validates_configuration_and_detection_layers():
    with pytest.raises(ValueError, match="unsupported task"):
        TaskOutputConsistency(nn.Identity(), "unknown")
    with pytest.raises(ValueError, match="missing detection"):
        TaskOutputConsistency(nn.Identity(), "detection")
    with pytest.raises(ValueError, match="temperature"):
        soft_target_kl(torch.zeros(1, 2), torch.zeros(1, 2), temperature=0)
