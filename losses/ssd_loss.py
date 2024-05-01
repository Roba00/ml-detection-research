import warnings
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torchvision.models.detection.ssd import SSD
from torchvision.ops import boxes as box_ops


# Basically a copy of from torchvision.models.detection.ssd.SSD.compute_loss()
def compute_loss(
    model,
    targets: List[Dict[str, Tensor]],
    head_outputs: Dict[str, Tensor],
    anchors: List[Tensor],
    matched_idxs: List[Tensor],
) -> Dict[str, Tensor]:
    bbox_regression = head_outputs["bbox_regression"]
    cls_logits = head_outputs["cls_logits"]

    # Match original targets with default boxes
    num_foreground = 0
    bbox_loss = []
    cls_targets = []
    for (
        targets_per_image,
        bbox_regression_per_image,
        cls_logits_per_image,
        anchors_per_image,
        matched_idxs_per_image,
    ) in zip(targets, bbox_regression, cls_logits, anchors, matched_idxs):
        # produce the matching between boxes and targets
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        foreground_matched_idxs_per_image = matched_idxs_per_image[
            foreground_idxs_per_image
        ]
        num_foreground += foreground_matched_idxs_per_image.numel()

        # Calculate regression loss
        matched_gt_boxes_per_image = targets_per_image["boxes"][
            foreground_matched_idxs_per_image
        ]
        bbox_regression_per_image = bbox_regression_per_image[
            foreground_idxs_per_image, :
        ]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
        target_regression = model.box_coder.encode_single(
            matched_gt_boxes_per_image, anchors_per_image
        )
        bbox_loss.append(
            torch.nn.functional.smooth_l1_loss(
                bbox_regression_per_image, target_regression, reduction="sum"
            )
        )

        # Estimate ground truth for class targets
        gt_classes_target = torch.zeros(
            (cls_logits_per_image.size(0),),
            dtype=targets_per_image["labels"].dtype,
            device=targets_per_image["labels"].device,
        )
        gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
            foreground_matched_idxs_per_image
        ]
        cls_targets.append(gt_classes_target)

    bbox_loss = torch.stack(bbox_loss)
    cls_targets = torch.stack(cls_targets)

    # Calculate classification loss
    num_classes = cls_logits.size(-1)
    cls_loss = F.cross_entropy(
        cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none"
    ).view(cls_targets.size())

    # Hard Negative Sampling
    foreground_idxs = cls_targets > 0
    num_negative = model.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
    # num_negative[num_negative < model.neg_to_pos_ratio] = model.neg_to_pos_ratio
    negative_loss = cls_loss.clone()
    negative_loss[foreground_idxs] = -float(
        "inf"
    )  # use -inf to detect positive values that creeped in the sample
    values, idx = negative_loss.sort(1, descending=True)
    # background_idxs = torch.logical_and(idx.sort(1)[1] < num_negative, torch.isfinite(values))
    background_idxs = idx.sort(1)[1] < num_negative

    N = max(1, num_foreground)
    return {
        "bbox_regression": bbox_loss.sum() / N,
        "classification": (
            cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()
        )
        / N,
    }


# Basically a copy of from torchvision.models.detection.ssd.SSD.forward()
def eval_forward(
    model: SSD, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    model.eval()

    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                torch._assert(
                    len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                    f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                )
            else:
                torch._assert(
                    False,
                    f"Expected target boxes to be of type Tensor, got {type(boxes)}.",
                )

    # get the original image sizes
    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        torch._assert(
            len(val) == 2,
            f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
        )
        original_image_sizes.append((val[0], val[1]))

    # transform the input
    images, targets = model.transform(images, targets)

    # Check for degenerate boxes
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                torch._assert(
                    False,
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}.",
                )

    # get the features from the backbone
    features = model.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    # compute the ssd heads outputs using the features
    head_outputs = model.head(features)

    # create the set of anchors
    anchors = model.anchor_generator(images, features)

    losses = {}
    detections: List[Dict[str, Tensor]] = []

    matched_idxs = []
    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full(
                        (anchors_per_image.size(0),),
                        -1,
                        dtype=torch.int64,
                        device=anchors_per_image.device,
                    )
                )
                continue

            match_quality_matrix = box_ops.box_iou(
                targets_per_image["boxes"], anchors_per_image
            )
            matched_idxs.append(model.proposal_matcher(match_quality_matrix))

        losses = model.compute_loss(targets, head_outputs, anchors, matched_idxs)

    return losses
