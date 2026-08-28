import torch
import torch.nn as nn

from imc_ste import ProposalAlignedROIConsistency


class _FakeROIPool(nn.Module):
    def forward(self, features, proposals, image_shapes):
        del image_shapes
        if isinstance(features, dict):
            features = features["0"]
        pooled = features.mean(dim=(-2, -1))
        rows = []
        for image_index, image_proposals in enumerate(proposals):
            rows.append(pooled[image_index : image_index + 1].expand(len(image_proposals), -1))
        return torch.cat(rows, dim=0)


class _FakePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Linear(4, 3)
        self.regressor = nn.Linear(4, 12)

    def forward(self, features):
        return self.classifier(features), self.regressor(features)


class _FakeROIHeads(nn.Module):
    def __init__(self):
        super().__init__()
        self.box_roi_pool = _FakeROIPool()
        self.box_head = nn.Sequential(nn.Linear(2, 4), nn.ReLU())
        self.box_predictor = _FakePredictor()
        self.box_coder = _FakeBoxCoder()

    def assign_targets_to_proposals(self, proposals, gt_boxes, gt_labels):
        matched = []
        labels = []
        for image_proposals, image_boxes, image_labels in zip(
            proposals, gt_boxes, gt_labels
        ):
            matched.append(
                torch.zeros(
                    len(image_proposals), dtype=torch.int64, device=image_proposals.device
                )
            )
            if image_boxes.numel() == 0:
                labels.append(
                    torch.zeros(
                        len(image_proposals),
                        dtype=torch.int64,
                        device=image_proposals.device,
                    )
                )
            else:
                labels.append(image_labels[:1].expand(len(image_proposals)))
        return matched, labels

    def forward(self, features, proposals, image_shapes, targets=None):
        del targets
        pooled = self.box_roi_pool(features, proposals, image_shapes)
        return self.box_predictor(self.box_head(pooled))


class _FakeBoxCoder:
    def encode(self, matched_gt_boxes, proposals):
        return [
            boxes - image_proposals
            for boxes, image_proposals in zip(matched_gt_boxes, proposals)
        ]


class _FakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Conv2d(2, 2, kernel_size=1)
        self.roi_heads = _FakeROIHeads()
        self.proposal_count = 2

    def forward(self, images, targets=None):
        features = {"0": self.backbone(images)}
        proposals = [
            images.new_zeros((self.proposal_count, 4))
            for _ in range(images.shape[0])
        ]
        image_shapes = [tuple(images.shape[-2:]) for _ in range(images.shape[0])]
        return self.roi_heads(features, proposals, image_shapes, targets)


def test_proposal_aligned_roi_consistency_reuses_clean_proposal_rows():
    model = _FakeDetector()
    consistency = ProposalAlignedROIConsistency(model, box_weight=0.25)
    inputs = torch.randn(1, 2, 3, 3, requires_grad=True)

    consistency.begin_clean()
    with torch.no_grad():
        model(inputs)
    consistency.begin_noisy()
    model.proposal_count = 3
    model(1.2 * inputs)
    loss, details = consistency.loss()

    assert torch.isfinite(loss)
    assert details["proposals"] == 2
    loss.backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())
    consistency.close()


def test_proposal_consistency_rejects_non_detector_model():
    try:
        ProposalAlignedROIConsistency(nn.Linear(2, 2))
    except ValueError as error:
        assert "Faster R-CNN" in str(error)
    else:
        raise AssertionError("expected a detector validation error")


def test_proposal_aligned_target_supervision_uses_clean_sampled_rows():
    model = _FakeDetector()
    consistency = ProposalAlignedROIConsistency(
        model,
        objective="target",
        box_weight=1.0,
    )
    inputs = torch.randn(1, 2, 3, 3, requires_grad=True)
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([1], dtype=torch.int64),
        }
    ]

    consistency.begin_clean()
    with torch.no_grad():
        model(inputs, targets)
    consistency.begin_noisy()
    model.proposal_count = 4
    model(1.1 * inputs, targets)
    loss, details = consistency.loss()

    assert torch.isfinite(loss)
    assert details["proposals"] == 2
    assert details["foreground"] == 2
    loss.backward()
    assert inputs.grad is not None
    assert bool(torch.isfinite(inputs.grad).all())
    consistency.close()


def test_proposal_aligned_foreground_supervision_is_finite():
    model = _FakeDetector()
    consistency = ProposalAlignedROIConsistency(
        model,
        objective="foreground_target",
        box_weight=1.0,
    )
    inputs = torch.randn(1, 2, 3, 3, requires_grad=True)
    targets = [
        {
            "boxes": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
            "labels": torch.tensor([2], dtype=torch.int64),
        }
    ]

    consistency.begin_clean()
    with torch.no_grad():
        model(inputs, targets)
    consistency.begin_noisy()
    model(0.9 * inputs, targets)
    loss, details = consistency.loss()

    assert torch.isfinite(loss)
    assert details["foreground"] == details["proposals"] == 2
    loss.backward()
    assert inputs.grad is not None
    consistency.close()
