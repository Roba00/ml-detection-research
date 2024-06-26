from collections import OrderedDict
import warnings
import torch
from torch import Tensor
from typing import Dict, List, Optional, Tuple
from torchvision.models.detection.retinanet import RetinaNet

def eval_forward(model, images, targets=None):
    # type: (RetinaNet, List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Args:
        images (list[Tensor]): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    model.eval()

    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        for target in targets:
            boxes = target["boxes"]
            torch._assert(isinstance(boxes, torch.Tensor), "Expected target boxes to be of type Tensor.")
            torch._assert(
                len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                "Expected target boxes to be a tensor of shape [N, 4].",
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
    # TODO: Move this to a function
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                # print the first degenerate box
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

    # TODO: Do we want a list or a dict?
    features = list(features.values())

    # compute the retinanet heads outputs using the features
    head_outputs = model.head(features)

    # create the set of anchors
    anchors = model.anchor_generator(images, features)

    losses = {}
    # detections: List[Dict[str, Tensor]] = []
    if targets is None:
        torch._assert(False, "targets should not be none when in training mode")
    else:
        # compute the losses
        losses = model.compute_loss(targets, head_outputs, anchors)

    return losses